import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
import networkx as nx
import numpy as np


class Evolved(Algorithm):
    """
    Optimized prefix reuse algorithm
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

    def calculate_col_stats(self, df: pd.DataFrame, enable_index=True) -> Tuple[int, List]:
        stats = []
        num_rows = len(df)
        for col in df.columns:
            if col == "original_index":
                continue
            try:
                unique_vals = df[col].nunique(dropna=True)
                avg_len = df[col].apply(self.calculate_length).mean()
                freq_score = (num_rows - unique_vals) * avg_len
                stats.append((col, unique_vals, avg_len, freq_score))
            except Exception:
                stats.append((col, num_rows, 0.0, 0.0))
        stats.sort(key=lambda x: (-x[3], -x[2]))
        return num_rows, stats

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        if not value_counts:
            return None
        weighted_counts = {}
        for val, count in value_counts.items():
            if count <= 1:
                continue
            weight = self.calculate_length(val) * (count - 1)
            if weight >= early_stop:
                weighted_counts[val] = weight
        if not weighted_counts:
            return None
        return max(weighted_counts, key=weighted_counts.get)

    def reorder_columns_for_value(self, row: pd.Series, value, column_names: List[str]) -> Tuple[pd.Series, List[str]]:
        cols_with_value = []
        cols_without_value = []
        for col in column_names:
            try:
                cell_val = row[col]
                if pd.isna(cell_val) and pd.isna(value):
                    cols_with_value.append(col)
                elif not pd.isna(cell_val) and not pd.isna(value) and cell_val == value:
                    cols_with_value.append(col)
                else:
                    cols_without_value.append(col)
            except Exception:
                cols_without_value.append(col)

        if self.dep_graph is not None:
            reordered_cols = []
            processed_cols = set()
            for col in cols_with_value:
                if col in processed_cols:
                    continue
                reordered_cols.append(col)
                processed_cols.add(col)
                if col in self.dep_graph.nodes:
                    for dep_col in nx.descendants(self.dep_graph, col):
                        if dep_col in column_names and dep_col not in processed_cols:
                            reordered_cols.append(dep_col)
                            processed_cols.add(dep_col)
            remaining = [col for col in cols_with_value if col not in processed_cols]
            reordered_cols.extend(remaining)
            reordered_cols.extend(cols_without_value)
        else:
            reordered_cols = cols_with_value + cols_without_value

        reordered_row = row.reindex(reordered_cols)
        return reordered_row, cols_with_value

    def fixed_reorder(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[str]]]:
        if df.empty:
            return df, []
        _, column_stats = self.calculate_col_stats(df)
        reordered_columns = [col for col, _, _, _ in column_stats]
        reordered_df = df.reindex(columns=reordered_columns).copy()
        column_orderings = [reordered_columns] * len(reordered_df)
        return reordered_df, column_orderings

    def recursive_reorder(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0,
                          row_stop: int = 0, col_stop: int = 0) -> Tuple[pd.DataFrame, List[List[str]]]:
        if df.empty or len(df.columns) == 0:
            return df, []
        
        if self.row_stop is not None and row_stop >= self.row_stop:
            return self.fixed_reorder(df)
        
        if self.col_stop is not None and col_stop >= self.col_stop:
            return self.fixed_reorder(df)

        max_value = self.find_max_group_value(df, value_counts, early_stop)
        if max_value is None:
            return self.fixed_reorder(df)

        has_value_mask = df.isin([max_value]).any(axis=1)
        grouped_rows = df[has_value_mask].copy()
        remaining_rows = df[~has_value_mask].copy()

        if grouped_rows.empty:
            return self.fixed_reorder(df)

        column_orderings = []
        reordered_group = pd.DataFrame(index=grouped_rows.index, columns=df.columns)
        
        for idx, (_, row) in enumerate(grouped_rows.iterrows()):
            reordered_row, _ = self.reorder_columns_for_value(row, max_value, df.columns.tolist())
            reordered_group.loc[row.name] = reordered_row
            column_orderings.append(reordered_row.index.tolist())

        if len(reordered_group.columns) > 1:
            first_col = reordered_group.columns[0]
            if reordered_group[first_col].nunique() == 1:
                remaining_cols = reordered_group.columns[1:]
                if len(remaining_cols) > 0:
                    sub_df = reordered_group[remaining_cols]
                    sub_value_counts = Counter(sub_df.stack(future_stack=True, dropna=False))
                    reordered_sub, sub_orderings = self.recursive_reorder(
                        sub_df, sub_value_counts, early_stop, row_stop + 1, col_stop + 1
                    )
                    reordered_group[remaining_cols] = reordered_sub
                    for i in range(len(sub_orderings)):
                        if i < len(column_orderings):
                            column_orderings[i] = [first_col] + sub_orderings[i]

        remaining_value_counts = Counter(remaining_rows.stack(future_stack=True, dropna=False))
        reordered_remaining, remaining_orderings = self.recursive_reorder(
            remaining_rows, remaining_value_counts, early_stop, row_stop + 1, col_stop
        )

        final_df = pd.concat([reordered_group, reordered_remaining], axis=0).sort_index()
        column_orderings.extend(remaining_orderings)

        return final_df, column_orderings

    def recursive_split_and_reorder(self, df: pd.DataFrame, early_stop: int = 0) -> Tuple[pd.DataFrame, List[List[str]]]:
        if len(df) <= self.base:
            value_counts = Counter(df.stack(future_stack=True, dropna=False))
            return self.recursive_reorder(df, value_counts, early_stop)

        mid_idx = len(df) // 2
        top_half = df.iloc[:mid_idx].copy()
        bottom_half = df.iloc[mid_idx:].copy()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, top_half, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, bottom_half, early_stop)
            
            reordered_top, order_top = future_top.result()
            reordered_bottom, order_bottom = future_bottom.result()

        final_df = pd.concat([reordered_top, reordered_bottom], axis=0, ignore_index=True)
        all_orderings = order_top + order_bottom

        return final_df, all_orderings

    def merging_columns(self, df: pd.DataFrame, cols_to_merge: List[str], prepended: bool = True) -> pd.DataFrame:
        if not cols_to_merge or len(cols_to_merge) < 2:
            return df
        merged_col_name = "|".join(cols_to_merge)
        df[merged_col_name] = df[cols_to_merge].apply(
            lambda x: "|".join(str(v) if not pd.isna(v) else "" for v in x), axis=1
        )
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
            for merge_group in col_merge:
                valid_cols = [col for col in merge_group if col in working_df.columns]
                if len(valid_cols) >= 2:
                    working_df = self.merging_columns(working_df, valid_cols)

        self.row_stop = row_stop if row_stop is not None else len(working_df)
        self.col_stop = col_stop if col_stop is not None else len(working_df.columns)

        if one_way_dep:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1_matches = [col for col in working_df.columns if dep[0] in col]
                col2_matches = [col for col in working_df.columns if dep[1] in col]
                if col1_matches and col2_matches:
                    self.dep_graph.add_edge(col1_matches[0], col2_matches[0])

        nunique_threshold = len(working_df) * distinct_value_threshold
        columns_to_keep = []
        columns_to_discard = []
        for col in working_df.columns:
            try:
                nunique = working_df[col].nunique(dropna=True)
                if nunique <= nunique_threshold:
                    columns_to_keep.append(col)
                else:
                    columns_to_discard.append(col)
            except Exception:
                columns_to_keep.append(col)

        working_df["original_index"] = range(len(working_df))
        recurse_df = working_df[columns_to_keep + ["original_index"]].copy()
        discard_df = working_df[columns_to_discard + ["original_index"]].copy()

        try:
            if parallel and len(recurse_df) > self.base:
                reordered_recurse, orderings = self.recursive_split_and_reorder(
                    recurse_df.drop(columns=["original_index"]), early_stop
                )
            else:
                value_counts = Counter(recurse_df.drop(columns=["original_index"]).stack(future_stack=True, dropna=False))
                reordered_recurse, orderings = self.recursive_reorder(
                    recurse_df.drop(columns=["original_index"]), value_counts, early_stop
                )
        except Exception:
            reordered_recurse, orderings = self.fixed_reorder(recurse_df.drop(columns=["original_index"]))

        reordered_recurse["original_index"] = recurse_df["original_index"].values

        if len(columns_to_discard) > 0:
            final_df = pd.merge(reordered_recurse, discard_df, on="original_index", how="left")
        else:
            final_df = reordered_recurse

        final_df = final_df.drop(columns=["original_index"])

        if not col_merge:
            assert final_df.shape == initial_df.shape, f"Shape mismatch: {final_df.shape} vs {initial_df.shape}"
        else:
            assert final_df.shape[0] == initial_df.shape[0], f"Row count mismatch: {final_df.shape[0]} vs {initial_df.shape[0]}"

        if len(orderings) != len(final_df):
            base_order = final_df.columns.tolist()
            orderings = [base_order] * len(final_df)

        return final_df, orderings