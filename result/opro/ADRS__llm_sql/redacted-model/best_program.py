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

    def calculate_col_stats(self, df: pd.DataFrame, enable_index: bool = False) -> Tuple[int, List]:
        stats = []
        num_rows = len(df)
        for col in df.columns:
            if enable_index and col == df.index.name:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                stats.append((col, 0, 0.0, 0.0))
                continue
            value_counts = series.value_counts()
            num_groups = len(value_counts)
            avg_len = np.mean([self.calculate_length(v) for v in series])
            score = sum(
                self.calculate_length(v) * (count - 1)
                for v, count in value_counts.items()
                if count > 1
            )
            stats.append((col, num_groups, avg_len, score))
        stats.sort(key=lambda x: (-x[3], -x[2], x[1]))
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

    def reorder_columns_for_value(self, row_series, value, column_names):
        cols_with_value = []
        cols_without_value = []
        for col in column_names:
            val = row_series[col]
            if pd.isna(val):
                cols_without_value.append(col)
            else:
                if val == value:
                    cols_with_value.append(col)
                else:
                    cols_without_value.append(col)
        if self.dep_graph is not None:
            ordered_cols = []
            added = set()
            for col in cols_with_value:
                if col not in added:
                    ordered_cols.append(col)
                    added.add(col)
                for dep_col in nx.descendants(self.dep_graph, col):
                    if dep_col in cols_with_value and dep_col not in added:
                        ordered_cols.append(dep_col)
                        added.add(col)
            remaining = [col for col in cols_with_value if col not in ordered_cols]
            ordered_cols.extend(remaining)
            ordered_cols.extend(cols_without_value)
            return ordered_cols
        else:
            return cols_with_value + cols_without_value

    def get_dependent_columns(self, col: str) -> List[str]:
        if self.dep_graph is None or not self.dep_graph.has_node(col):
            return []
        return list(nx.descendants(self.dep_graph, col))

    @lru_cache(maxsize=None)
    def get_cached_dependent_columns(self, col: str) -> List[str]:
        return self.get_dependent_columns(col)

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        _, column_stats = self.calculate_col_stats(df, enable_index=True)
        reordered_columns = [col for col, _, _, _ in column_stats]
        reordered_df = df[reordered_columns].copy()
        if row_sort:
            reordered_df = reordered_df.sort_values(by=reordered_columns, axis=0, na_position='last')
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

        reordered_groups = []
        column_orderings = []

        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, row in grouped_rows.iterrows():
                futures.append(executor.submit(
                    self.reorder_columns_for_value,
                    row,
                    max_value,
                    df.columns.tolist()
                ))

            for idx, future in enumerate(as_completed(futures)):
                col_order = future.result()
                reordered_row = grouped_rows.iloc[idx][col_order].values
                reordered_groups.append(reordered_row)
                column_orderings.append(col_order)

        result_df = pd.DataFrame(reordered_groups, columns=df.columns, index=grouped_rows.index)

        current_col_order = result_df.columns.tolist()
        if len(current_col_order) > 1:
            first_col = current_col_order[0]
            if result_df[first_col].eq(max_value).all():
                remaining_cols = current_col_order[1:]
                if remaining_cols:
                    sub_df = result_df[remaining_cols].copy()
                    sub_value_counts = Counter(sub_df.stack().dropna())
                    reordered_sub, sub_orderings = self.recursive_reorder(
                        sub_df,
                        sub_value_counts,
                        early_stop=early_stop,
                        row_stop=row_stop + 1,
                        col_stop=col_stop + 1
                    )
                    result_df[remaining_cols] = reordered_sub.values
                    for i in range(len(column_orderings)):
                        if i < len(sub_orderings):
                            column_orderings[i] = [first_col] + sub_orderings[i]

        reordered_remaining, remaining_orderings = self.recursive_reorder(
            remaining_rows,
            value_counts,
            early_stop=early_stop,
            row_stop=row_stop + 1,
            col_stop=col_stop
        )

        final_df = pd.concat([result_df, reordered_remaining]).sort_index()
        final_orderings = column_orderings + remaining_orderings

        return final_df, final_orderings

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack().dropna())
            reordered_df, _ = self.recursive_reorder(df, initial_value_counts, early_stop, original_columns)
            return reordered_df

        mid_index = len(df) // 2
        df_top = df.iloc[:mid_index].copy()
        df_bottom = df.iloc[mid_index:].copy()

        with ThreadPoolExecutor() as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top, original_columns, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom, original_columns, early_stop)

        reordered_top = future_top.result()
        reordered_bottom = future_bottom.result()

        return pd.concat([reordered_top, reordered_bottom], axis=0, ignore_index=True)

    def merging_columns(self, df: pd.DataFrame, cols_to_merge: List[str], prepended: bool = True) -> pd.DataFrame:
        merged_col_name = '|'.join(cols_to_merge)
        df[merged_col_name] = df[cols_to_merge].apply(lambda x: '|'.join(x.astype(str)), axis=1)
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
            if working_df[col].nunique() <= nunique_threshold:
                columns_to_keep.append(col)
            else:
                columns_to_discard.append(col)

        working_df['original_index'] = range(len(working_df))
        discard_df = working_df[columns_to_discard + ['original_index']].copy()
        recurse_df = working_df[columns_to_keep + ['original_index']].copy()

        self.row_stop = row_stop if row_stop is not None else len(recurse_df)
        self.col_stop = col_stop if col_stop is not None else len(recurse_df.columns)

        if parallel:
            reordered_recurse = self.recursive_split_and_reorder(recurse_df, columns_to_keep, early_stop)
        else:
            initial_value_counts = Counter(recurse_df[columns_to_keep].stack().dropna())
            reordered_recurse, _ = self.recursive_reorder(
                recurse_df,
                initial_value_counts,
                early_stop=early_stop
            )

        if columns_to_discard:
            final_df = pd.merge(
                reordered_recurse,
                discard_df,
                on='original_index',
                how='left'
            )
        else:
            final_df = reordered_recurse

        final_df = final_df.drop(columns=['original_index'])

        if not col_merge:
            assert final_df.shape == initial_df.shape, \
                f"Shape mismatch: {final_df.shape} vs {initial_df.shape}"
        else:
            assert final_df.shape[0] == initial_df.shape[0], \
                f"Row count mismatch: {final_df.shape[0]} vs {initial_df.shape[0]}"

        final_df = final_df.sort_values(by=final_df.columns.tolist(), axis=0, na_position='last')

        column_orderings = [final_df.columns.tolist()] * len(final_df)
        return final_df, column_orderings