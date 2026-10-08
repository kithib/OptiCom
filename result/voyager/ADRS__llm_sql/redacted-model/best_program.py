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
        self.base = 6000  # Increased base size to reduce recursion overhead

    def calculate_col_stats(self, df: pd.DataFrame, enable_index: bool = False) -> Tuple[int, List]:
        stats = []
        for col in df.columns:
            if enable_index and col == df.index.name:
                continue
            num_unique = df[col].nunique()
            avg_len = df[col].apply(lambda x: len(str(x)) if pd.notna(x) else 0).mean()
            score = (df.shape[0] - num_unique) * avg_len
            stats.append((col, num_unique, avg_len, score))
        stats.sort(key=lambda x: x[3], reverse=True)
        return df.shape[0], stats

    def merging_columns(self, df: pd.DataFrame, cols_to_merge: List[str], prepended: bool = True) -> pd.DataFrame:
        merged_col_name = '_'.join(cols_to_merge)
        df[merged_col_name] = df[cols_to_merge].astype(str).agg(' '.join, axis=1)
        return df.drop(columns=cols_to_merge)

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        # Skip recalculating value counts, use provided counts for efficiency
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
                row_val = row[col]
                if pd.notna(row_val) and row_val == value:
                    cols_with_value.append(col)
            except (AttributeError, KeyError):
                continue

        if self.dep_graph is not None and grouped_rows_len > 1:
            reordered_cols = []
            for col in cols_with_value:
                dependent_cols = self.get_dependent_columns(col)
                valid_dependent_cols = [c for c in dependent_cols if c in column_names]
                reordered_cols.extend([col] + valid_dependent_cols)
            cols_without_value = [col for col in column_names if col not in reordered_cols]
            reordered_cols.extend(cols_without_value)
        else:
            cols_without_value = [col for col in column_names if col not in cols_with_value]
            reordered_cols = cols_with_value + cols_without_value

        assert len(reordered_cols) == len(column_names), f"Reordered cols len: {len(reordered_cols)}  Original cols len: {len(column_names)}"
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

        assert reordered_df.shape == df.shape
        column_orderings = [reordered_columns] * num_rows

        if row_sort:
            reordered_df = reordered_df.sort_values(by=reordered_columns, axis=0)

        return reordered_df, column_orderings

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        result_df = grouped_rows.copy()
        column_names = grouped_rows.columns.tolist()

        # Vectorized approach to find columns with max value
        has_max_val = grouped_rows == max_value
        cols_with_value = has_max_val.any(axis=0)[has_max_val.any(axis=0)].index.tolist()
        
        if cols_with_value:
            cols_without_value = [col for col in column_names if col not in cols_with_value]
            reordered_cols = cols_with_value + cols_without_value
            result_df = result_df[reordered_cols]

        grouped_value_counts = Counter(result_df.stack().dropna())

        if not result_df.empty and cols_with_value:
            length_of_settle_cols = len(cols_with_value)
            if length_of_settle_cols < len(result_df.columns):
                group_remainder = result_df.iloc[:, length_of_settle_cols:]
                if not group_remainder.empty and col_stop + 1 < self.col_stop:
                    grouped_remainder_value_counts = Counter(group_remainder.stack().dropna())
                    reordered_group_remainder, _ = self.recursive_reorder(
                        group_remainder, grouped_remainder_value_counts, early_stop=early_stop,
                        row_stop=row_stop, col_stop=col_stop + 1
                    )
                    result_df.iloc[:, length_of_settle_cols:] = reordered_group_remainder.values

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

        # Early termination conditions to prevent deep recursion
        if self.row_stop is not None and row_stop >= self.row_stop:
            return self.fixed_reorder(df, row_sort=False)

        if self.col_stop is not None and col_stop >= self.col_stop:
            return self.fixed_reorder(df, row_sort=False)

        if original_columns is None:
            original_columns = df.columns.tolist()

        max_value = self.find_max_group_value(df, value_counts, early_stop=early_stop)
        if max_value is None:
            return self.fixed_reorder(df, row_sort=False)

        has_max_value = df.isin([max_value]).any(axis=1)
        grouped_rows = df[has_max_value].copy()
        remaining_rows = df[~has_max_value].copy()

        if grouped_rows.empty:
            return self.fixed_reorder(df, row_sort=False)

        result_df, grouped_value_counts = self.column_recursion(result_df=None, max_value=max_value,
                                                               grouped_rows=grouped_rows, row_stop=row_stop,
                                                               col_stop=col_stop, early_stop=early_stop)

        # Efficiently calculate remaining value counts
        remaining_value_counts = Counter()
        for val, count in value_counts.items():
            remaining_count = count - grouped_value_counts.get(val, 0)
            if remaining_count > 0:
                remaining_value_counts[val] = remaining_count

        reordered_remaining_rows, _ = self.recursive_reorder(
            remaining_rows, remaining_value_counts, early_stop=early_stop,
            row_stop=row_stop + 1, col_stop=col_stop
        )

        final_result_df = pd.concat([result_df, reordered_remaining_rows], axis=0, ignore_index=True)
        final_result_df.columns = df.columns

        return final_result_df, []

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack().dropna())
            return self.recursive_reorder(df, initial_value_counts, early_stop, original_columns, row_stop=0, col_stop=0)[0]

        mid_index = len(df) // 2
        df_top_half = df.iloc[:mid_index].copy()
        df_bottom_half = df.iloc[mid_index:].copy()

        # Limit parallelism to avoid thread overhead and timeout
        if len(df) > self.base * 5:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_top = executor.submit(self.recursive_split_and_reorder, df_top_half, original_columns, early_stop)
                future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom_half, original_columns, early_stop)
                reordered_top_half = future_top.result()
                reordered_bottom_half = future_bottom.result()
        else:
            reordered_top_half = self.recursive_split_and_reorder(df_top_half, original_columns, early_stop)
            reordered_bottom_half = self.recursive_split_and_reorder(df_bottom_half, original_columns, early_stop)

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
        return len(str(value)) ** 2

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
        df = df.copy()

        if col_merge:
            for cols_to_merge in col_merge:
                existing_cols = [col for col in cols_to_merge if col in df.columns]
                if len(existing_cols) >= 2:
                    df = self.merging_columns(df, existing_cols)

        nunique_threshold = len(df) * distinct_value_threshold
        columns_to_discard = [
            col for col in df.columns
            if df[col].nunique() > nunique_threshold
        ]
        columns_to_recurse = [col for col in df.columns if col not in columns_to_discard]

        df["original_index"] = range(len(df))
        discarded_columns_df = df[columns_to_discard + ["original_index"]].copy()
        df_to_recurse = df[columns_to_recurse + ["original_index"]].copy()

        # Set strict recursion limits to prevent infinite recursion and timeout
        self.row_stop = row_stop if row_stop is not None else min(2, len(df_to_recurse) // 4000)
        self.col_stop = col_stop if col_stop is not None else min(2, len(df_to_recurse.columns) // 5)

        initial_value_counts = Counter(df_to_recurse.stack().dropna())
        self.val_len = {val: self.calculate_length(val) for val in initial_value_counts.keys()}

        if one_way_dep:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1_matches = [col for col in df.columns if dep[0] in col]
                col2_matches = [col for col in df.columns if dep[1] in col]
                if col1_matches and col2_matches:
                    self.dep_graph.add_edge(col1_matches[0], col2_matches[0])

        # Use parallel processing only for very large datasets to avoid thread overhead
        if parallel and len(df_to_recurse) > self.base * 6:
            reordered_df = self.recursive_split_and_reorder(df_to_recurse, columns_to_recurse, early_stop)
        else:
            reordered_df, _ = self.recursive_reorder(
                df_to_recurse, initial_value_counts, early_stop=early_stop
            )

        assert reordered_df.shape == df_to_recurse.shape, \
            f"Reordered shape {reordered_df.shape} != original {df_to_recurse.shape}"
        assert reordered_df["original_index"].nunique() == len(reordered_df), "Duplicate original indexes found"

        if len(columns_to_discard) > 0:
            final_df = pd.merge(
                reordered_df, discarded_columns_df,
                on="original_index", how="inner"
            )
        else:
            final_df = reordered_df.copy()

        final_df = final_df.drop(columns=["original_index"])

        if not col_merge:
            assert final_df.shape == initial_df.shape, \
                f"Final shape {final_df.shape} != initial {initial_df.shape}"
            assert len(final_df) == len(initial_df), "Row count mismatch after reordering"
        else:
            assert final_df.shape[0] == initial_df.shape[0], \
                f"Final row count {final_df.shape[0]} != initial {initial_df.shape[0]}"
            expected_cols = len(initial_df.columns) - sum(len(g)-1 for g in col_merge)
            assert final_df.shape[1] == expected_cols, \
                f"Final column count {final_df.shape[1]} != expected {expected_cols}"

        # Optimized column ordering calculation using vectorized operations
        column_orderings = [final_df.columns.tolist()]
        if len(final_df) > 1:
            # Compare consecutive rows with proper NA handling
            matches = final_df.iloc[1:] == final_df.iloc[:-1].values
            # Calculate cumulative matches (stop at first mismatch)
            cumulative_matches = matches.cumprod(axis=1)
            # Get prefix lengths - sum of consecutive matches starting from first column
            prefix_lengths = cumulative_matches.sum(axis=1)
            
            for length in prefix_lengths:
                prefix_cols = final_df.columns[:length].tolist()
                remaining_cols = final_df.columns[length:].tolist()
                column_orderings.append(prefix_cols + remaining_cols)

        return final_df, column_orderings