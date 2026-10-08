import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
import networkx as nx


class Evolved(Algorithm):
    """
    GGR algorithm optimized for prefix hit count and runtime
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

        self.dep_graph = None
        self.num_rows = 0
        self.num_cols = 0
        self.column_stats = None
        self.val_len = None
        self.row_stop = None
        self.col_stop = None
        self.base = 2000

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        weighted_counts = {val: self.val_len[val] * (count - 1) for val, count in value_counts.items() if count > 1}
        if not weighted_counts:
            return None
        max_group_val, max_weighted_count = max(weighted_counts.items(), key=lambda x: x[1])
        if max_weighted_count < early_stop:
            return None
        return max_group_val

    def reorder_columns_for_value(self, row, value, column_names, grouped_rows_len: int = 1):
        cols_with_value = []
        for col in column_names:
            try:
                if row[col] == value:
                    cols_with_value.append(col)
            except (AttributeError, KeyError):
                continue

        if self.dep_graph is not None and grouped_rows_len > 1:
            reordered_cols = []
            seen_cols = set()
            for col in cols_with_value:
                if col not in seen_cols:
                    reordered_cols.append(col)
                    seen_cols.add(col)
                dependent_cols = self.get_dependent_columns(col)
                for dep_col in dependent_cols:
                    if dep_col in column_names and dep_col not in seen_cols:
                        reordered_cols.append(dep_col)
                        seen_cols.add(dep_col)
            cols_without_value = [col for col in column_names if col not in seen_cols]
            reordered_cols.extend(cols_without_value)
        else:
            cols_with_value_sorted = sorted(
                cols_with_value,
                key=lambda col: (self.val_len.get(row[col], 0), len(row[col]) if isinstance(row[col], str) else 0),
                reverse=True
            )
            cols_without_value_sorted = sorted(
                [col for col in column_names if col not in cols_with_value],
                key=lambda col: (self.val_len.get(row[col], 0), len(row[col]) if isinstance(row[col], str) else 0),
                reverse=True
            )
            reordered_cols = cols_with_value_sorted + cols_without_value_sorted

        assert len(reordered_cols) == len(column_names), f"Reordered cols len mismatch: {len(reordered_cols)} vs {len(column_names)}"
        return [row[col] for col in reordered_cols], cols_with_value

    def get_dependent_columns(self, col: str) -> List[str]:
        if self.dep_graph is None or not self.dep_graph.has_node(col):
            return []
        return list(nx.descendants(self.dep_graph, col))

    @lru_cache(maxsize=None)
    def get_cached_dependent_columns(self, col: str) -> List[str]:
        return self.get_dependent_columns(col)

    def calculate_col_stats(self, df: pd.DataFrame, enable_index: bool = False) -> Tuple[int, List]:
        stats = []
        for col in df.columns:
            if col == "original_index" and enable_index:
                continue
            nunique = df[col].nunique()
            avg_len = df[col].astype(str).apply(len).mean()
            score = (df.shape[0] - nunique) * (avg_len ** 2)
            stats.append((col, nunique, avg_len, score))
        stats.sort(key=lambda x: x[3], reverse=True)
        return len(df), stats

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        num_rows, column_stats = self.calculate_col_stats(df, enable_index=True)
        reordered_columns = [col for col, _, _, _ in column_stats]
        reordered_df = df[reordered_columns].copy()

        if row_sort:
            reordered_df = reordered_df.sort_values(by=reordered_columns, axis=0)

        column_orderings = [reordered_columns] * num_rows
        return reordered_df, column_orderings

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        with ThreadPoolExecutor(max_workers=min(4, len(grouped_rows))) as executor:
            futures = [
                executor.submit(self.reorder_columns_for_value, row, max_value, grouped_rows.columns.tolist(), len(grouped_rows))
                for _, row in grouped_rows.iterrows()
            ]
            results = [future.result() for future in as_completed(futures)]

        result_df = pd.DataFrame([row_data for row_data, _ in results], columns=grouped_rows.columns)
        cols_settled = results[0][1] if results else []

        if not result_df.empty and cols_settled:
            mask = result_df.iloc[:, 0] == max_value
            if mask.any():
                group = result_df[mask].copy()
                if len(cols_settled) < len(group.columns):
                    group_remainder = group.iloc[:, len(cols_settled):]
                    remainder_counts = Counter(group_remainder.stack())
                    reordered_remainder, _ = self.recursive_reorder(
                        group_remainder, remainder_counts, early_stop=early_stop, 
                        row_stop=row_stop, col_stop=col_stop + 1
                    )
                    group.iloc[:, len(cols_settled):] = reordered_remainder.values
                    result_df.loc[mask] = group.values

        grouped_value_counts = Counter(result_df.stack())
        return result_df, grouped_value_counts

    def recursive_reorder(
        self,
        df: pd.DataFrame,
        value_counts: Dict,
        early_stop: int = 0,
        original_columns: List[str] = None,
        row_stop: int = 0,
        col_stop: int = 0,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        if df.empty or len(df.columns) == 0:
            return df, []

        if self.row_stop is not None and row_stop >= self.row_stop:
            return self.fixed_reorder(df, row_sort=False)

        if self.col_stop is not None and col_stop >= self.col_stop:
            return self.fixed_reorder(df, row_sort=False)

        if original_columns is None:
            original_columns = df.columns.tolist()

        max_value = self.find_max_group_value(df, value_counts, early_stop=early_stop)
        if max_value is None:
            return self.fixed_reorder(df, row_sort=False)

        grouped_mask = df.isin([max_value]).any(axis=1)
        grouped_rows = df[grouped_mask].copy()
        remaining_rows = df[~grouped_mask].copy()

        if grouped_rows.empty:
            return self.fixed_reorder(df, row_sort=False)

        result_df, grouped_value_counts = self.column_recursion(
            pd.DataFrame(columns=df.columns), max_value, grouped_rows, row_stop, col_stop, early_stop
        )

        remaining_value_counts = {k: max(0, value_counts[k] - grouped_value_counts.get(k, 0)) for k in value_counts}
        reordered_remaining, _ = self.recursive_reorder(
            remaining_rows, remaining_value_counts, early_stop=early_stop, 
            row_stop=row_stop + 1, col_stop=col_stop
        )

        final_result_df = pd.concat([result_df, reordered_remaining], axis=0, ignore_index=True)
        final_result_df.columns = df.columns

        column_orderings = [final_result_df.columns.tolist()] * len(final_result_df)
        return final_result_df, column_orderings

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack())
            return self.recursive_reorder(df, initial_value_counts, early_stop, original_columns, row_stop=0, col_stop=0)[0]

        mid_index = len(df) // 2
        df_top_half = df.iloc[:mid_index].copy()
        df_bottom_half = df.iloc[mid_index:].copy()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top_half, original_columns, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom_half, original_columns, early_stop)
            reordered_top_half = future_top.result()
            reordered_bottom_half = future_bottom.result()

        reordered_df = pd.concat([reordered_top_half, reordered_bottom_half], axis=0, ignore_index=True)
        assert reordered_df.shape == df.shape, f"Shape mismatch: {reordered_df.shape} vs {df.shape}"
        return reordered_df

    @lru_cache(maxsize=None)
    def calculate_length(self, value):
        if pd.isna(value):
            return 0
        if isinstance(value, bool):
            return 16
        if isinstance(value, (int, float)):
            return len(str(value)) ** 2
        if isinstance(value, str):
            return len(value) ** 2
        return 0

    def merging_columns(self, df: pd.DataFrame, cols_to_merge: List[str], prepended: bool = True) -> pd.DataFrame:
        merged_col_name = '|'.join(cols_to_merge)
        df[merged_col_name] = df[cols_to_merge].astype(str).agg('|'.join, axis=1)
        df = df.drop(columns=cols_to_merge)
        if prepended:
            cols = [merged_col_name] + [col for col in df.columns if col != merged_col_name]
            df = df[cols]
        return df

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        initial_df = df.copy()
        working_df = df.copy()

        if col_merge:
            for cols_to_merge in col_merge:
                valid_cols = [col for col in cols_to_merge if col in working_df.columns]
                if len(valid_cols) >= 2:
                    working_df = self.merging_columns(working_df, valid_cols)

        nunique_threshold = len(working_df) * distinct_value_threshold
        columns_to_discard = [
            col for col in working_df.columns 
            if working_df[col].nunique() > nunique_threshold and col != "original_index"
        ]
        columns_to_recurse = [col for col in working_df.columns if col not in columns_to_discard]

        working_df["original_index"] = range(len(working_df))
        discarded_columns_df = working_df[columns_to_discard + ["original_index"]] if columns_to_discard else None
        recurse_df = working_df[columns_to_recurse + ["original_index"]].copy()

        initial_value_counts = Counter(recurse_df.drop(columns=["original_index"]).stack())
        self.val_len = {val: self.calculate_length(val) for val in initial_value_counts.keys()}

        self.row_stop = row_stop if row_stop is not None else len(recurse_df)
        self.col_stop = col_stop if col_stop is not None else len(columns_to_recurse)

        if one_way_dep:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1_matches = [col for col in working_df.columns if dep[0] in col]
                col2_matches = [col for col in working_df.columns if dep[1] in col]
                if col1_matches and col2_matches:
                    self.dep_graph.add_edge(col1_matches[0], col2_matches[0])

        if parallel:
            reordered_recurse_df = self.recursive_split_and_reorder(
                recurse_df.drop(columns=["original_index"]), 
                original_columns=columns_to_recurse, 
                early_stop=early_stop
            )
            reordered_recurse_df["original_index"] = recurse_df["original_index"].values
        else:
            reordered_recurse_df, _ = self.recursive_reorder(
                recurse_df.drop(columns=["original_index"]),
                initial_value_counts,
                early_stop=early_stop,
            )
            reordered_recurse_df["original_index"] = recurse_df["original_index"].values

        if discarded_columns_df is not None:
            final_df = pd.merge(
                reordered_recurse_df, discarded_columns_df, 
                on="original_index", how="left"
            )
        else:
            final_df = reordered_recurse_df

        final_df = final_df.drop(columns=["original_index"])

        if col_merge:
            original_cols = [col for col in initial_df.columns if col in final_df.columns]
            merged_cols = [col for col in final_df.columns if col not in initial_df.columns]
            final_df = final_df[original_cols + merged_cols]

        column_orderings = []
        for idx in range(len(final_df)):
            prev_row = final_df.iloc[idx-1] if idx > 0 else None
            if prev_row is not None:
                current_row = final_df.iloc[idx]
                matching_cols = []
                non_matching_cols = []
                for col in final_df.columns:
                    if current_row[col] == prev_row[col]:
                        matching_cols.append(col)
                    else:
                        non_matching_cols.append(col)
                
                matching_cols_sorted = sorted(
                    matching_cols,
                    key=lambda col: (self.val_len.get(current_row[col], 0), len(str(current_row[col]))),
                    reverse=True
                )
                non_matching_cols_sorted = sorted(
                    non_matching_cols,
                    key=lambda col: (self.val_len.get(current_row[col], 0), len(str(current_row[col]))),
                    reverse=True
                )
                current_order = matching_cols_sorted + non_matching_cols_sorted
                final_df.iloc[idx] = final_df.iloc[idx][current_order].values
                column_orderings.append(current_order)
            else:
                column_orderings.append(final_df.columns.tolist())

        return final_df, column_orderings