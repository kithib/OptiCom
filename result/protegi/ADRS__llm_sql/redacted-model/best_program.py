import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
import networkx as nx


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
        cols_without_value = []
        for idx, col in enumerate(column_names):
            try:
                current_val = row[col]
            except (AttributeError, KeyError):
                try:
                    current_val = row[col.replace(" ", "_")]
                except (AttributeError, KeyError):
                    attr_name = f"_{idx}"
                    try:
                        current_val = row[attr_name]
                    except (AttributeError, KeyError):
                        cols_without_value.append(col)
                        continue
            if current_val == value:
                cols_with_value.append(col)
            else:
                cols_without_value.append(col)

        if self.dep_graph is not None and grouped_rows_len > 1:
            reordered_cols = []
            for col in cols_with_value:
                dependent_cols = self.get_dependent_columns(col)
                valid_dependent_cols = []
                for dep_col in dependent_cols:
                    if dep_col in column_names:
                        valid_dependent_cols.append(dep_col)
                reordered_cols.extend([col] + valid_dependent_cols)
            cols_without_value = [col for col in column_names if col not in reordered_cols]
            reordered_cols.extend(cols_without_value)
        else:
            reordered_cols = cols_with_value + cols_without_value

        assert len(reordered_cols) == len(column_names), f"Reordered cols len: {len(reordered_cols)}  Original cols len: {len(column_names)}"
        reordered_row = [row[col] if col in row else row[col.replace(" ", "_")] if col.replace(" ", "_") in row else row[f"_{column_names.index(col)}"] for col in reordered_cols]
        return reordered_row, cols_with_value

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
            if col == "original_index" and not enable_index:
                continue
            nunique = df[col].nunique()
            avg_len = df[col].astype(str).apply(len).mean()
            score = (len(df) - nunique) * (avg_len ** 2)
            stats.append((col, nunique, avg_len, score))
        stats.sort(key=lambda x: x[3], reverse=True)
        return len(df), stats

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        num_rows, column_stats = self.calculate_col_stats(df, enable_index=True)
        reordered_columns = [col for col, _, _, _ in column_stats]
        reordered_df = df[reordered_columns]

        assert reordered_df.shape == df.shape
        column_orderings = [reordered_columns] * num_rows

        if row_sort:
            reordered_df = reordered_df.sort_values(by=reordered_columns, axis=0)

        return reordered_df, column_orderings

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        cols_settled = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.reorder_columns_for_value, row, max_value, grouped_rows.columns.tolist(), len(grouped_rows))
                for _, row in grouped_rows.iterrows()
            ]
            for i, future in enumerate(as_completed(futures)):
                reordered_row, cols_settled = future.result()
                result_df.loc[i] = reordered_row

        grouped_value_counts = Counter()

        if not result_df.empty:
            grouped_result_df = result_df.groupby(result_df.columns[0])
            grouped_value_counts = Counter(grouped_rows.stack())

            for _, group in grouped_result_df:
                if group[group.columns[0]].iloc[0] != max_value:
                    continue

                dependent_cols = self.get_cached_dependent_columns(group.columns[0])
                length_of_settle_cols = len(cols_settled)

                if dependent_cols:
                    for col in group.columns[:length_of_settle_cols]:
                        if group[col].nunique() != 1:
                            break
                    else:
                        group_remainder = group.iloc[:, length_of_settle_cols:]
                        grouped_remainder_value_counts = Counter(group_remainder.stack())
                        reordered_group_remainder, _ = self.recursive_reorder(
                            group_remainder, grouped_remainder_value_counts, early_stop=early_stop, row_stop=row_stop, col_stop=col_stop + 1
                        )
                        group.iloc[:, length_of_settle_cols:] = reordered_group_remainder.values
                        result_df.update(group)
                        continue

                group_remainder = group.iloc[:, 1:]
                grouped_remainder_value_counts = Counter(group_remainder.stack())
                reordered_group_remainder, _ = self.recursive_reorder(
                    group_remainder, grouped_remainder_value_counts, early_stop=early_stop, row_stop=row_stop, col_stop=col_stop + 1
                )
                group.iloc[:, 1:] = reordered_group_remainder.values
                result_df.update(group)
                break

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
            return self.fixed_reorder(df)

        if self.col_stop is not None and col_stop >= self.col_stop:
            return self.fixed_reorder(df)

        if original_columns is None:
            original_columns = df.columns.tolist()

        max_value = self.find_max_group_value(df, value_counts, early_stop=early_stop)
        if max_value is None:
            return self.fixed_reorder(df)

        grouped_rows = df[df.isin([max_value]).any(axis=1)]
        remaining_rows = df[~df.isin([max_value]).any(axis=1)]

        if grouped_rows.empty:
            return self.fixed_reorder(df)

        result_df = pd.DataFrame(index=range(len(grouped_rows)), columns=df.columns)
        reordered_remaining_rows = pd.DataFrame(columns=df.columns)

        result_df, grouped_value_counts = self.column_recursion(result_df, max_value, grouped_rows, row_stop, col_stop, early_stop)

        remaining_value_counts = Counter()
        for val, cnt in value_counts.items():
            remaining_cnt = cnt - grouped_value_counts.get(val, 0)
            if remaining_cnt > 0:
                remaining_value_counts[val] = remaining_cnt

        if not remaining_rows.empty:
            reordered_remaining_rows, _ = self.recursive_reorder(
                remaining_rows, remaining_value_counts, early_stop=early_stop, row_stop=row_stop + 1, col_stop=col_stop
            )

        final_result_df = pd.concat([result_df, reordered_remaining_rows], axis=0, ignore_index=True)
        final_result_df.columns = df.columns

        return final_result_df, []

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack())
            return self.recursive_reorder(df, initial_value_counts, early_stop, original_columns, row_stop=0, col_stop=0)[0]

        mid_index = len(df) // 2
        df_top_half = df.iloc[:mid_index]
        df_bottom_half = df.iloc[mid_index:]

        with ThreadPoolExecutor() as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top_half, original_columns, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom_half, original_columns, early_stop)

        reordered_top_half = future_top.result()
        reordered_bottom_half = future_bottom.result()

        assert reordered_bottom_half.shape == df_bottom_half.shape
        reordered_df = pd.concat([reordered_top_half, reordered_bottom_half], axis=0, ignore_index=True)

        assert reordered_df.shape == df.shape

        return reordered_df

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

    def merging_columns(self, df: pd.DataFrame, columns: List[str], prepended: bool = True) -> pd.DataFrame:
        merged_col_name = '_'.join(columns)
        df[merged_col_name] = df[columns].astype(str).agg(' '.join, axis=1)
        return df.drop(columns=columns)

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

        self.num_rows, self.column_stats = self.calculate_col_stats(working_df, enable_index=True)
        self.column_stats = {col: (num_groups, avg_len, score) for col, num_groups, avg_len, score in self.column_stats}

        if one_way_dep and len(one_way_dep) > 0:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1_matches = [col for col in working_df.columns if dep[0] in col]
                col2_matches = [col for col in working_df.columns if dep[1] in col]
                if len(col1_matches) == 1 and len(col2_matches) == 1:
                    self.dep_graph.add_edge(col1_matches[0], col2_matches[0])

        nunique_threshold = len(working_df) * distinct_value_threshold
        columns_to_discard = [
            col for col in working_df.columns
            if col != "original_index" and working_df[col].nunique() > nunique_threshold
        ]
        columns_to_recurse = [col for col in working_df.columns if col not in columns_to_discard]

        working_df["original_index"] = range(len(working_df))
        discarded_columns_df = working_df[columns_to_discard + ["original_index"]]
        recurse_df = working_df[columns_to_recurse + ["original_index"]]

        self.column_stats = {col: stats for col, stats in self.column_stats.items() if col in columns_to_recurse}
        initial_value_counts = Counter(recurse_df.stack())
        self.val_len = {val: self.calculate_length(val) for val in initial_value_counts.keys()}

        self.row_stop = row_stop if row_stop is not None else len(recurse_df)
        self.col_stop = col_stop if col_stop is not None else len(recurse_df.columns)

        if parallel:
            reordered_df = self.recursive_split_and_reorder(recurse_df, original_columns=columns_to_recurse, early_stop=early_stop)
        else:
            reordered_df, _ = self.recursive_reorder(
                recurse_df,
                initial_value_counts,
                early_stop=early_stop,
            )

        assert reordered_df.shape == recurse_df.shape, f"Reordered DataFrame shape {reordered_df.shape} does not match original DataFrame shape {recurse_df.shape}"
        assert recurse_df["original_index"].is_unique, "Passed in recurse index contains duplicates!"
        assert reordered_df["original_index"].is_unique, "Reordered index contains duplicates!"

        if columns_to_discard:
            final_df = pd.merge(reordered_df, discarded_columns_df, on="original_index", how="left")
        else:
            final_df = reordered_df

        final_df = final_df.drop(columns=["original_index"])

        if not col_merge:
            assert final_df.shape == initial_df.shape, f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {initial_df.shape}"
        else:
            assert final_df.shape[0] == initial_df.shape[0], f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {initial_df.shape}"

        final_df = final_df.sort_values(by=final_df.columns.tolist(), axis=0)
        
        column_orderings = [final_df.columns.tolist()] * len(final_df)
        return final_df, column_orderings