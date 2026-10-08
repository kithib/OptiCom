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
    Prefix-Optimized Column Reordering Algorithm
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
        stats = []
        num_rows = len(df)
        for col in df.columns:
            if df[col].nunique() == 0:
                continue
            avg_len = df[col].apply(self.calculate_length).mean()
            group_count = df[col].nunique()
            score = (num_rows - group_count) * avg_len
            stats.append((col, group_count, avg_len, score))
        stats.sort(key=lambda x: x[3], reverse=True)
        return num_rows, stats

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        weighted_counts = {}
        for val, count in value_counts.items():
            if count <= 1:
                continue
            weight = self.calculate_length(val) * (count - 1)
            if weight >= early_stop:
                weighted_counts[val] = weight
        if not weighted_counts:
            return None
        return max(weighted_counts.items(), key=lambda x: x[1])[0]

    def reorder_columns_for_value(self, row, value, column_names, grouped_rows_len: int = 1):
        cols_with_value = []
        cols_without_value = []
        for col in column_names:
            current_val = row[col]
            if pd.isna(current_val) and pd.isna(value):
                cols_with_value.append(col)
            elif current_val == value:
                cols_with_value.append(col)
            else:
                cols_without_value.append(col)

        if self.dep_graph is not None and grouped_rows_len > 1:
            reordered_cols = []
            for col in cols_with_value:
                dependent_cols = list(self.dep_graph.successors(col)) if self.dep_graph.has_node(col) else []
                valid_deps = [dep for dep in dependent_cols if dep in column_names]
                reordered_cols.extend([col] + valid_deps)
            reordered_cols = list(dict.fromkeys(reordered_cols))
            cols_without_value = [col for col in column_names if col not in reordered_cols]
            reordered_cols.extend(cols_without_value)
        else:
            reordered_cols = cols_with_value + cols_without_value

        assert len(reordered_cols) == len(column_names), f"Reordered cols len: {len(reordered_cols)} != Original: {len(column_names)}"
        return reordered_cols

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        num_rows, column_stats = self.calculate_col_stats(df)
        reordered_columns = [col for col, _, _, _ in column_stats]
        reordered_df = df[reordered_columns].copy()

        if row_sort:
            reordered_df = reordered_df.sort_values(by=reordered_columns, axis=0)

        column_orderings = [reordered_columns] * num_rows
        return reordered_df, column_orderings

    def column_recursion(self, df: pd.DataFrame, max_value, early_stop: int):
        cols_settled = []
        for col in df.columns:
            if (df[col] == max_value).all():
                cols_settled.append(col)
            elif pd.isna(max_value) and df[col].isna().all():
                cols_settled.append(col)
        
        if not cols_settled:
            return df, Counter()
        
        remainder_df = df.drop(columns=cols_settled)
        if remainder_df.empty:
            return df, Counter(df.stack(future_stack=True, dropna=True))
        
        initial_value_counts = Counter(remainder_df.stack(future_stack=True, dropna=True))
        reordered_remainder, _ = self.recursive_reorder(remainder_df, initial_value_counts, early_stop)
        
        final_df = pd.concat([df[cols_settled], reordered_remainder], axis=1)
        return final_df, Counter(final_df.stack(future_stack=True, dropna=True))

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

        max_value = self.find_max_group_value(df, value_counts, early_stop)
        if max_value is None:
            return self.fixed_reorder(df, row_sort=False)

        has_value_mask = df.isin([max_value]).any(axis=1)
        if pd.isna(max_value):
            has_value_mask = df.isna().any(axis=1)
        
        grouped_rows = df[has_value_mask].copy()
        remaining_rows = df[~has_value_mask].copy()

        if grouped_rows.empty:
            return self.fixed_reorder(df, row_sort=False)

        reordered_group, grouped_counts = self.column_recursion(grouped_rows, max_value, early_stop)
        reordered_remaining, _ = self.recursive_reorder(
            remaining_rows, value_counts, early_stop, row_stop=row_stop + 1, col_stop=col_stop
        )

        final_df = pd.concat([reordered_group, reordered_remaining], axis=0, ignore_index=True)
        return final_df, []

    def recursive_split_and_reorder(self, df: pd.DataFrame, early_stop: int = 0):
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack(future_stack=True, dropna=True))
            return self.recursive_reorder(df, initial_value_counts, early_stop)[0]

        mid_index = len(df) // 2
        df_top = df.iloc[:mid_index].copy()
        df_bottom = df.iloc[mid_index:].copy()

        with ThreadPoolExecutor() as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom, early_stop)

        reordered_top = future_top.result()
        reordered_bottom = future_bottom.result()

        final_df = pd.concat([reordered_top, reordered_bottom], axis=0, ignore_index=True)
        return final_df

    def merging_columns(self, df: pd.DataFrame, cols_to_merge: List[str], prepended: bool = True) -> pd.DataFrame:
        merged_col_name = '|'.join(cols_to_merge)
        df[merged_col_name] = df[cols_to_merge].astype(str).agg('|'.join, axis=1)
        return df.drop(columns=cols_to_merge)

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
        columns_to_keep = []
        columns_to_discard = []
        for col in working_df.columns:
            if working_df[col].nunique() <= nunique_threshold:
                columns_to_keep.append(col)
            else:
                columns_to_discard.append(col)

        if not columns_to_keep:
            columns_to_keep = working_df.columns[:1].tolist()
            columns_to_discard = [col for col in working_df.columns if col not in columns_to_keep]

        working_df["original_index"] = range(len(working_df))
        discard_df = working_df[columns_to_discard + ["original_index"]].copy()
        recurse_df = working_df[columns_to_keep + ["original_index"]].copy()

        self.row_stop = row_stop if row_stop is not None else len(recurse_df)
        self.col_stop = col_stop if col_stop is not None else len(recurse_df.columns)

        if one_way_dep:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1 = next((col for col in working_df.columns if dep[0] in col), None)
                col2 = next((col for col in working_df.columns if dep[1] in col), None)
                if col1 and col2:
                    self.dep_graph.add_edge(col1, col2)

        try:
            if parallel and len(recurse_df) > self.base:
                reordered_recurse = self.recursive_split_and_reorder(recurse_df.drop(columns=["original_index"]), early_stop)
                reordered_recurse["original_index"] = recurse_df["original_index"].values
            else:
                initial_value_counts = Counter(recurse_df.drop(columns=["original_index"]).stack(future_stack=True, dropna=True))
                reordered_recurse, _ = self.recursive_reorder(
                    recurse_df.drop(columns=["original_index"]), initial_value_counts, early_stop
                )
                reordered_recurse["original_index"] = recurse_df["original_index"].values
        except Exception:
            reordered_recurse, _ = self.fixed_reorder(recurse_df.drop(columns=["original_index"]), row_sort=False)
            reordered_recurse["original_index"] = recurse_df["original_index"].values

        if columns_to_discard:
            final_df = pd.merge(
                reordered_recurse, discard_df, on="original_index", how="left"
            )
        else:
            final_df = reordered_recurse

        final_df = final_df.drop(columns=["original_index"])

        if col_merge:
            original_cols = [col for col in initial_df.columns if col in final_df.columns]
            merged_cols = [col for col in final_df.columns if col not in initial_df.columns]
            final_df = final_df.reindex(columns=original_cols + merged_cols)

        column_orderings = [final_df.columns.tolist()] * len(final_df)
        return final_df, column_orderings