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
    GGR algorithm
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

        self.dep_graph = None  # NOTE: not used, for one way dependency

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
            return 4**2
        if isinstance(value, (int, float)):
            return len(str(value)) ** 2
        if isinstance(value, str):
            return len(value) ** 2
        return 0

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        weighted_counts = {}
        for val, count in value_counts.items():
            if count <= 1:
                continue
            weight = self.calculate_length(val) * (count - 1)
            weighted_counts[val] = weight
        
        if not weighted_counts:
            return None
        
        max_group_val = max(weighted_counts.items(), key=lambda x: x[1])[0]
        if weighted_counts[max_group_val] < early_stop:
            return None
        return max_group_val

    def reorder_columns_for_value(self, row_series, value, column_names, grouped_rows_len: int = 1):
        cols_with_value = [col for col in column_names if row_series[col] == value]
        
        if self.dep_graph is not None and grouped_rows_len > 1:
            reordered_cols = []
            for col in cols_with_value:
                reordered_cols.append(col)
                dependent_cols = list(nx.descendants(self.dep_graph, col)) if self.dep_graph and col in self.dep_graph else []
                reordered_cols.extend([c for c in dependent_cols if c in column_names])
            reordered_cols = list(dict.fromkeys(reordered_cols))
            cols_without_value = [col for col in column_names if col not in reordered_cols]
            reordered_cols.extend(cols_without_value)
        else:
            cols_without_value = [col for col in column_names if row_series[col] != value]
            reordered_cols = cols_with_value + cols_without_value
        
        assert len(reordered_cols) == len(column_names), "Column count mismatch in reordering"
        return row_series[reordered_cols].values.tolist(), cols_with_value

    def get_dependent_columns(self, col: str) -> List[str]:
        if self.dep_graph is None or not self.dep_graph.has_node(col):
            return []
        return list(nx.descendants(self.dep_graph, col))

    @lru_cache(maxsize=None)
    def get_cached_dependent_columns(self, col: str) -> List[str]:
        return self.get_dependent_columns(col)

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        col_scores = []
        for col in df.columns:
            if col == "original_index":
                continue
            nunique = df[col].nunique()
            avg_len = df[col].apply(self.calculate_length).mean()
            score = avg_len * (len(df) - nunique)
            col_scores.append((-score, col))
        
        col_scores.sort()
        reordered_columns = [col for _, col in col_scores]
        if "original_index" in df.columns:
            reordered_columns.append("original_index")
        
        reordered_df = df[reordered_columns]
        
        if row_sort:
            sort_cols = [col for col in reordered_columns if col != "original_index"]
            reordered_df = reordered_df.sort_values(by=sort_cols, axis=0)
        
        column_orderings = [reordered_columns] * len(reordered_df)
        return reordered_df, column_orderings

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        cols_settled = []
        grouped_cols = grouped_rows.columns.tolist()
        
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.reorder_columns_for_value, row, max_value, grouped_cols, len(grouped_rows))
                for _, row in grouped_rows.iterrows()
            ]
            reordered_rows = []
            for future in as_completed(futures):
                row_vals, cols = future.result()
                reordered_rows.append(row_vals)
                if not cols_settled:
                    cols_settled = cols
        
        result_df = pd.DataFrame(reordered_rows, columns=grouped_cols)
        grouped_value_counts = Counter(result_df.melt()['value'])

        if not result_df.empty and cols_settled:
            settled_count = len(cols_settled)
            if settled_count < len(result_df.columns):
                group_remainder = result_df.iloc[:, settled_count:]
                if not group_remainder.empty:
                    remainder_counts = Counter(group_remainder.melt()['value'])
                    reordered_remainder, _ = self.recursive_reorder(
                        group_remainder, remainder_counts, early_stop=early_stop,
                        row_stop=row_stop, col_stop=col_stop + 1
                    )
                    result_df.iloc[:, settled_count:] = reordered_remainder.values

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

        grouped_rows = df[df.isin([max_value]).any(axis=1)]
        remaining_rows = df[~df.isin([max_value]).any(axis=1)]

        if grouped_rows.empty:
            return self.fixed_reorder(df, row_sort=False)

        result_df, grouped_value_counts = self.column_recursion(
            pd.DataFrame(columns=df.columns), max_value, grouped_rows, row_stop, col_stop, early_stop
        )

        reordered_remaining_rows = pd.DataFrame(columns=df.columns)
        if not remaining_rows.empty:
            remaining_counts = Counter()
            for val, cnt in value_counts.items():
                remaining_cnt = cnt - grouped_value_counts.get(val, 0)
                if remaining_cnt > 0:
                    remaining_counts[val] = remaining_cnt
            
            reordered_remaining_rows, _ = self.recursive_reorder(
                remaining_rows, remaining_counts, early_stop=early_stop,
                row_stop=row_stop + 1, col_stop=col_stop
            )

        final_result_df = pd.concat([result_df, reordered_remaining_rows], ignore_index=True)
        final_result_df.columns = df.columns

        return final_result_df, []

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        if len(df) <= self.base:
            initial_value_counts = Counter(df.melt()['value'])
            return self.recursive_reorder(df, initial_value_counts, early_stop, original_columns, row_stop=0, col_stop=0)[0]

        mid_index = len(df) // 2
        df_top_half = df.iloc[:mid_index]
        df_bottom_half = df.iloc[mid_index:]

        with ThreadPoolExecutor() as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top_half, original_columns, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom_half, original_columns, early_stop)

        reordered_top_half = future_top.result()
        reordered_bottom_half = future_bottom.result()

        reordered_df = pd.concat([reordered_top_half, reordered_bottom_half], axis=0, ignore_index=True)
        assert reordered_df.shape == df.shape, "Split-reorder shape mismatch"

        return reordered_df

    def calculate_col_stats(self, df: pd.DataFrame, enable_index: bool = False) -> Tuple[int, List]:
        stats = []
        for col in df.columns:
            if enable_index and col == df.index.name:
                continue
            nunique = df[col].nunique()
            avg_len = df[col].apply(self.calculate_length).mean()
            score = avg_len * (len(df) - nunique)
            stats.append((col, nunique, avg_len, score))
        
        stats.sort(key=lambda x: -x[3])
        return len(df), stats

    def merging_columns(self, df: pd.DataFrame, columns_to_merge: List[str], prepended: bool = True) -> pd.DataFrame:
        merged_col_name = '|'.join(columns_to_merge)
        df[merged_col_name] = df[columns_to_merge].astype(str).agg('|'.join, axis=1)
        df = df.drop(columns=columns_to_merge)
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

        nunique_threshold = len(working_df) * distinct_value_threshold
        columns_to_discard = [
            col for col in working_df.columns
            if working_df[col].nunique() > nunique_threshold and col != "original_index"
        ]
        columns_to_recurse = [col for col in working_df.columns if col not in columns_to_discard]

        working_df["original_index"] = range(len(working_df))
        discard_df = working_df[columns_to_discard + ["original_index"]]
        recurse_df = working_df[columns_to_recurse + ["original_index"]]

        self.row_stop = row_stop if row_stop is not None else len(recurse_df)
        self.col_stop = col_stop if col_stop is not None else len(recurse_df.columns)

        if parallel and len(recurse_df) > self.base:
            reordered_recurse_df = self.recursive_split_and_reorder(
                recurse_df, original_columns=columns_to_recurse, early_stop=early_stop
            )
        else:
            initial_value_counts = Counter(recurse_df.melt()['value'])
            reordered_recurse_df, _ = self.recursive_reorder(
                recurse_df, initial_value_counts, early_stop=early_stop
            )

        assert reordered_recurse_df.shape == recurse_df.shape, \
            f"Reordered shape {reordered_recurse_df.shape} != original {recurse_df.shape}"
        assert recurse_df["original_index"].is_unique, "Duplicate original indices!"

        if columns_to_discard:
            final_df = pd.merge(
                reordered_recurse_df, discard_df, on="original_index", how="left"
            )
        else:
            final_df = reordered_recurse_df

        final_df = final_df.drop(columns=["original_index"])

        if not col_merge:
            assert final_df.shape == initial_df.shape, \
                f"Final shape {final_df.shape} != initial {initial_df.shape}"
        else:
            assert final_df.shape[0] == initial_df.shape[0], \
                f"Final row count {final_df.shape[0]} != initial {initial_df.shape[0]}"

        column_orderings = []
        final_cols = final_df.columns.tolist()
        for idx in range(len(final_df)):
            current_row = final_df.iloc[idx]
            if idx == 0:
                col_order = sorted(final_cols, key=lambda x: -self.calculate_length(current_row[x]))
            else:
                prev_row = final_df.iloc[idx-1]
                match_length = 0
                for col in final_cols:
                    if pd.isna(current_row[col]) and pd.isna(prev_row[col]):
                        match_length += 1
                    elif not pd.isna(current_row[col]) and current_row[col] == prev_row[col]:
                        match_length += 1
                    else:
                        break
                
                if match_length > 0:
                    matched_cols = final_cols[:match_length]
                    remaining_cols = [col for col in final_cols if col not in matched_cols]
                    remaining_cols.sort(key=lambda x: -self.calculate_length(current_row[x]))
                    col_order = matched_cols + remaining_cols
                else:
                    col_order = sorted(final_cols, key=lambda x: -self.calculate_length(current_row[x]))
            
            column_orderings.append(col_order)

        return final_df, column_orderings