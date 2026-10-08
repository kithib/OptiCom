import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
import networkx as nx


class Evolved(Algorithm):
    """
    Prefix Optimization Algorithm for LLM Prompt Caching
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

    def calculate_col_stats(self, df: pd.DataFrame, enable_index: bool = False) -> Tuple[int, List]:
        num_rows = len(df)
        stats = []
        for col in df.columns:
            if col == "original_index" and not enable_index:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                stats.append((col, 0, 0, 0.0))
                continue
            value_counts = series.value_counts()
            num_groups = len(value_counts)
            avg_len = series.apply(self.calculate_length).mean()
            score = sum(
                (count - 1) * self.calculate_length(val)
                for val, count in value_counts.items()
                if count > 1
            )
            stats.append((col, num_groups, avg_len, score))
        stats.sort(key=lambda x: (-x[3], -x[2]))
        return num_rows, stats

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        weighted_counts = {}
        for val, count in value_counts.items():
            if count <= 1:
                continue
            weight = self.val_len.get(val, 0) * (count - 1)
            if weight >= early_stop:
                weighted_counts[val] = weight
        
        if not weighted_counts:
            return None
        return max(weighted_counts, key=weighted_counts.get)

    def reorder_columns_for_value(self, row, value, column_names, grouped_rows_len: int = 1):
        cols_with_value = []
        cols_without_value = []
        
        for col in column_names:
            current_val = row[col]
            if pd.isna(current_val) and pd.isna(value):
                cols_with_value.append(col)
            elif not pd.isna(current_val) and not pd.isna(value) and current_val == value:
                cols_with_value.append(col)
            else:
                cols_without_value.append(col)

        if self.dep_graph is not None and grouped_rows_len > 1:
            reordered_cols = []
            for col in cols_with_value:
                dependent_cols = self.get_dependent_columns(col)
                valid_deps = [dep for dep in dependent_cols if dep in column_names]
                reordered_cols.append(col)
                reordered_cols.extend(valid_deps)
            
            reordered_cols = list(dict.fromkeys(reordered_cols))
            remaining_cols = [col for col in column_names if col not in reordered_cols]
            reordered_cols.extend(remaining_cols)
        else:
            reordered_cols = cols_with_value + cols_without_value

        assert len(reordered_cols) == len(column_names), "Column count mismatch after reordering"
        reordered_row = [row[col] for col in reordered_cols]
        return reordered_row, cols_with_value

    def get_dependent_columns(self, col: str) -> List[str]:
        if self.dep_graph is None or not self.dep_graph.has_node(col):
            return []
        return list(nx.descendants(self.dep_graph, col))

    @lru_cache(maxsize=None)
    def get_cached_dependent_columns(self, col: str) -> List[str]:
        return self.get_dependent_columns(col)

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        num_rows, column_stats = self.calculate_col_stats(df, enable_index=True)
        reordered_columns = [col for col, _, _, _ in column_stats]
        reordered_df = df[reordered_columns].copy()

        assert reordered_df.shape == df.shape, "Shape mismatch in fixed reorder"
        column_orderings = [reordered_columns.copy() for _ in range(num_rows)]

        if row_sort:
            reordered_df = reordered_df.sort_values(by=reordered_columns, axis=0)

        return reordered_df, column_orderings

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        grouped_rows = grouped_rows.copy().reset_index(drop=True)
        result_df = pd.DataFrame(index=grouped_rows.index, columns=grouped_rows.columns)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.reorder_columns_for_value,
                    row,
                    max_value,
                    grouped_rows.columns.tolist(),
                    len(grouped_rows)
                )
                for _, row in grouped_rows.iterrows()
            ]
            
            for i, future in enumerate(as_completed(futures)):
                reordered_row, cols_settled = future.result()
                result_df.loc[i] = reordered_row

        if result_df.empty:
            return result_df, Counter()

        first_col = result_df.columns[0]
        matching_group = result_df[result_df[first_col] == max_value]
        
        if not matching_group.empty:
            settle_count = len(cols_settled)
            if settle_count < len(result_df.columns):
                group_remainder = matching_group.iloc[:, settle_count:].copy()
                
                if not group_remainder.empty:
                    remainder_counts = Counter(group_remainder.stack())
                    reordered_remainder, _ = self.recursive_reorder(
                        group_remainder,
                        remainder_counts,
                        early_stop=early_stop,
                        row_stop=row_stop,
                        col_stop=col_stop + 1
                    )
                    
                    for col_idx in range(settle_count, len(result_df.columns)):
                        src_col = reordered_remainder.columns[col_idx - settle_count]
                        result_df.loc[matching_group.index, result_df.columns[col_idx]] = matching_group[src_col].values

        flattened_values = result_df.stack().values
        grouped_value_counts = Counter(flattened_values)
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
            return df.copy(), []

        if self.row_stop is not None and row_stop >= self.row_stop:
            return self.fixed_reorder(df)

        if self.col_stop is not None and col_stop >= self.col_stop:
            return self.fixed_reorder(df)

        if original_columns is None:
            original_columns = df.columns.tolist()

        max_value = self.find_max_group_value(df, value_counts, early_stop=early_stop)
        if max_value is None:
            return self.fixed_reorder(df)

        has_value = df.isin([max_value]).any(axis=1)
        grouped_rows = df[has_value].copy().reset_index(drop=True)
        remaining_rows = df[~has_value].copy().reset_index(drop=True)

        if grouped_rows.empty:
            return self.fixed_reorder(df)

        result_df, grouped_value_counts = self.column_recursion(
            pd.DataFrame(), max_value, grouped_rows, row_stop, col_stop, early_stop
        )

        remaining_value_counts = Counter()
        for val, count in value_counts.items():
            remaining_count = count - grouped_value_counts.get(val, 0)
            if remaining_count > 0:
                remaining_value_counts[val] = remaining_count

        reordered_remaining, _ = self.recursive_reorder(
            remaining_rows,
            remaining_value_counts,
            early_stop=early_stop,
            row_stop=row_stop + 1,
            col_stop=col_stop
        )

        final_result = pd.concat([result_df, reordered_remaining], axis=0, ignore_index=True)
        final_result.columns = df.columns

        column_orderings = []
        for _, row in final_result.iterrows():
            current_order = final_result.columns.tolist()
            column_orderings.append(current_order.copy())

        return final_result, column_orderings

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack())
            return self.recursive_reorder(df, initial_value_counts, early_stop, original_columns)[0]

        mid_index = len(df) // 2
        df_top = df.iloc[:mid_index].copy().reset_index(drop=True)
        df_bottom = df.iloc[mid_index:].copy().reset_index(drop=True)

        with ThreadPoolExecutor() as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top, original_columns, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom, original_columns, early_stop)

        reordered_top = future_top.result()
        reordered_bottom = future_bottom.result()

        reordered_df = pd.concat([reordered_top, reordered_bottom], axis=0, ignore_index=True)
        assert reordered_df.shape == df.shape, "Shape mismatch after split reorder"
        return reordered_df

    def merging_columns(self, df: pd.DataFrame, columns: List[str], prepended: bool = True) -> pd.DataFrame:
        merged_col_name = "_merged_".join(columns)
        df[merged_col_name] = df[columns].apply(lambda x: " | ".join(x.astype(str)), axis=1)
        df = df.drop(columns=columns)
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
            for merge_group in col_merge:
                valid_cols = [col for col in merge_group if col in working_df.columns]
                if len(valid_cols) >= 2:
                    working_df = self.merging_columns(working_df, valid_cols)

        nunique_threshold = len(working_df) * distinct_value_threshold
        columns_to_keep = [
            col for col in working_df.columns
            if working_df[col].nunique() <= nunique_threshold or col == "original_index"
        ]
        columns_to_discard = [col for col in working_df.columns if col not in columns_to_keep]

        working_df["original_index"] = range(len(working_df))
        discard_df = working_df[columns_to_discard + ["original_index"]].copy()
        recurse_df = working_df[columns_to_keep + ["original_index"]].copy()

        self.val_len = {}
        if not recurse_df.empty:
            all_values = recurse_df.stack().unique()
            self.val_len = {val: self.calculate_length(val) for val in all_values}

        self.row_stop = row_stop if row_stop is not None else len(recurse_df)
        self.col_stop = col_stop if col_stop is not None else len(recurse_df.columns)

        if parallel and len(recurse_df) > self.base:
            reordered_recurse = self.recursive_split_and_reorder(recurse_df, columns_to_keep, early_stop)
        else:
            initial_value_counts = Counter(recurse_df.stack())
            reordered_recurse, _ = self.recursive_reorder(
                recurse_df, initial_value_counts, early_stop=early_stop
            )

        assert reordered_recurse.shape == recurse_df.shape, "Shape mismatch after reordering"
        assert reordered_recurse["original_index"].is_unique, "Duplicate indices detected"

        if columns_to_discard:
            final_df = pd.merge(
                reordered_recurse, discard_df, on="original_index", how="left"
            )
        else:
            final_df = reordered_recurse.copy()

        final_df = final_df.drop(columns=["original_index"])

        if not col_merge:
            assert final_df.shape == initial_df.shape, "Final shape mismatch with original"
        else:
            assert final_df.shape[0] == initial_df.shape[0], "Row count mismatch"

        final_orderings = []
        for _ in range(len(final_df)):
            final_orderings.append(final_df.columns.tolist())

        return final_df, final_orderings