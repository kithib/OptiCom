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
    Optimized Prefix Reuse Algorithm
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
            unique_vals = df[col].nunique(dropna=True)
            freq = df[col].value_counts(normalize=True).max()
            avg_len = df[col].apply(self.calculate_length).mean()
            score = (1.0 - (unique_vals / num_rows)) * avg_len
            stats.append((col, unique_vals, avg_len, score))
        stats.sort(key=lambda x: x[3], reverse=True)
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
        return max(weighted_counts.items(), key=lambda x: x[1])[0]

    def reorder_columns_for_row(self, current_row, prev_row, column_names):
        match_scores = []
        for col in column_names:
            current_val = current_row[col]
            prev_val = prev_row[col] if prev_row is not None else pd.NA
            if pd.isna(current_val) or pd.isna(prev_val):
                match = 0
            else:
                match = 1 if current_val == prev_val else 0
            score = match * self.column_stats[col][2]
            match_scores.append((-score, col))
        match_scores.sort()
        reordered_cols = [col for (_, col) in match_scores]
        return reordered_cols

    def vectorized_prefix_score(self, df: pd.DataFrame, col_order: List[str]) -> pd.Series:
        df_ordered = df[col_order]
        shifted = df_ordered.shift(1)
        matches = (df_ordered == shifted).cumsum(axis=1)
        matches = (matches == matches.cummax(axis=1)).astype(int)
        str_lens = df_ordered.applymap(self.calculate_length)
        prefix_scores = (matches * str_lens).sum(axis=1)
        return prefix_scores

    def greedy_column_order(self, df: pd.DataFrame) -> List[str]:
        remaining_cols = df.columns.tolist()
        if "original_index" in remaining_cols:
            remaining_cols.remove("original_index")
        ordered_cols = []
        current_df = df.copy()

        while remaining_cols:
            best_col = None
            best_score = -1
            for col in remaining_cols:
                temp_order = ordered_cols + [col]
                score = self.vectorized_prefix_score(current_df, temp_order).sum()
                if score > best_score:
                    best_score = score
                    best_col = col
            if best_col is None:
                best_col = remaining_cols[0]
            ordered_cols.append(best_col)
            remaining_cols.remove(best_col)

        if "original_index" in df.columns:
            ordered_cols.append("original_index")
        return ordered_cols

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        reordered_columns = self.greedy_column_order(df)
        reordered_df = df[reordered_columns].copy()

        if row_sort:
            sort_cols = [col for col in reordered_columns if col != "original_index"]
            reordered_df = reordered_df.sort_values(by=sort_cols, axis=0)

        column_orderings = [reordered_columns] * len(reordered_df)
        return reordered_df, column_orderings

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
            return self.fixed_reorder(df)

        if self.col_stop is not None and col_stop >= self.col_stop:
            return self.fixed_reorder(df)

        if original_columns is None:
            original_columns = df.columns.tolist()

        max_value = self.find_max_group_value(df, value_counts, early_stop=early_stop)
        if max_value is None:
            return self.fixed_reorder(df)

        mask = df.isin([max_value]).any(axis=1)
        grouped_rows = df[mask].copy()
        remaining_rows = df[~mask].copy()

        if grouped_rows.empty:
            return self.fixed_reorder(df)

        result_df = pd.DataFrame(columns=df.columns)
        reordered_cols = self.greedy_column_order(grouped_rows)
        grouped_rows_ordered = grouped_rows[reordered_cols].copy()

        result_df = pd.concat([result_df, grouped_rows_ordered], ignore_index=True)

        grouped_value_counts = Counter(grouped_rows_ordered.stack(dropna=False))
        remaining_value_counts = Counter()
        for val, cnt in value_counts.items():
            remaining_cnt = cnt - grouped_value_counts.get(val, 0)
            if remaining_cnt > 0:
                remaining_value_counts[val] = remaining_cnt

        reordered_remaining_rows, _ = self.recursive_reorder(
            remaining_rows, remaining_value_counts, early_stop=early_stop,
            row_stop=row_stop + 1, col_stop=col_stop
        )

        final_result_df = pd.concat([result_df, reordered_remaining_rows], ignore_index=True)
        final_result_df.columns = df.columns

        column_orderings = [reordered_cols] * len(result_df) + _

        return final_result_df, column_orderings

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack(dropna=False))
            reordered_df, _ = self.recursive_reorder(df, initial_value_counts, early_stop, original_columns)
            return reordered_df

        mid_index = len(df) // 2
        df_top_half = df.iloc[:mid_index].copy()
        df_bottom_half = df.iloc[mid_index:].copy()

        with ThreadPoolExecutor() as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top_half, original_columns, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom_half, original_columns, early_stop)

        reordered_top_half = future_top.result()
        reordered_bottom_half = future_bottom.result()

        reordered_df = pd.concat([reordered_top_half, reordered_bottom_half], axis=0, ignore_index=True)
        return reordered_df

    def merging_columns(self, df: pd.DataFrame, cols_to_merge: List[str], prepended: bool = True) -> pd.DataFrame:
        merged_col_name = "_".join(cols_to_merge)
        df[merged_col_name] = df[cols_to_merge].apply(lambda x: "|".join(str(v) for v in x), axis=1)
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
                valid_cols = [col for col in working_df.columns if col in merge_group]
                if len(valid_cols) >= 2:
                    working_df = self.merging_columns(working_df, valid_cols)

        self.num_rows, stats_list = self.calculate_col_stats(working_df)
        self.column_stats = {col: (unique, avg_len, score) for col, unique, avg_len, score in stats_list}

        if one_way_dep:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1_matches = [col for col in working_df.columns if dep[0] in col]
                col2_matches = [col for col in working_df.columns if dep[1] in col]
                if col1_matches and col2_matches:
                    self.dep_graph.add_edge(col1_matches[0], col2_matches[0])

        nunique_threshold = len(working_df) * distinct_value_threshold
        columns_to_discard = []
        for col in working_df.columns:
            if col == "original_index":
                continue
            if working_df[col].nunique(dropna=True) > nunique_threshold:
                columns_to_discard.append(col)

        columns_to_recurse = [col for col in working_df.columns if col not in columns_to_discard]
        working_df["original_index"] = range(len(working_df))
        discarded_columns_df = working_df[columns_to_discard + ["original_index"]].copy()
        recurse_df = working_df[columns_to_recurse + ["original_index"]].copy()

        self.row_stop = row_stop if row_stop is not None else len(recurse_df)
        self.col_stop = col_stop if col_stop is not None else len(columns_to_recurse)

        initial_value_counts = Counter(recurse_df.stack(dropna=False))
        self.val_len = {val: self.calculate_length(val) for val in initial_value_counts.keys()}

        if parallel and len(recurse_df) > self.base:
            reordered_df = self.recursive_split_and_reorder(recurse_df, columns_to_recurse, early_stop)
        else:
            reordered_df, _ = self.recursive_reorder(
                recurse_df, initial_value_counts, early_stop=early_stop
            )

        assert reordered_df.shape == recurse_df.shape, "Shape mismatch after reordering"
        assert recurse_df["original_index"].is_unique, "Original index duplicates"
        assert reordered_df["original_index"].is_unique, "Reordered index duplicates"

        if columns_to_discard:
            final_df = pd.merge(
                reordered_df, discarded_columns_df,
                on="original_index", how="left"
            )
        else:
            final_df = reordered_df.copy()

        final_df = final_df.drop(columns=["original_index"])

        if not col_merge:
            assert final_df.shape == initial_df.shape, "Final shape mismatch"
        else:
            assert final_df.shape[0] == initial_df.shape[0], "Row count mismatch"
            expected_cols = (initial_df.shape[1] - sum(len(g)-1 for g in col_merge))
            assert final_df.shape[1] == expected_cols, "Column count mismatch"

        sort_cols = [col for col in final_df.columns if col != "original_index"]
        final_df = final_df.sort_values(by=sort_cols, axis=0)

        column_orderings = [final_df.columns.tolist()] * len(final_df)
        return final_df, column_orderings