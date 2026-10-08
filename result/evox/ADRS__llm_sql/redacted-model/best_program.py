import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
import networkx as nx


class Evolved(Algorithm):
    """
    Optimized Prefix Reuse Column Reordering Algorithm
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
        for col in df.columns:
            if df[col].dtype == 'object':
                non_na = df[col].dropna()
                if len(non_na) == 0:
                    avg_len = 0
                    score = 0
                else:
                    avg_len = non_na.str.len().mean() ** 2
                    freq = non_na.value_counts(normalize=True).max()
                    score = freq * avg_len
                num_groups = non_na.nunique()
            else:
                non_na = df[col].dropna()
                if len(non_na) == 0:
                    avg_len = 0
                    score = 0
                else:
                    avg_len = non_na.apply(lambda x: len(str(x))).mean() ** 2
                    freq = non_na.value_counts(normalize=True).max()
                    score = freq * avg_len
                num_groups = non_na.nunique()
            stats.append((col, num_groups, avg_len, score))
        stats.sort(key=lambda x: (-x[3], x[2]))
        return len(df), stats

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        weighted_counts = {}
        for val, count in value_counts.items():
            if pd.isna(val):
                continue
            weight = self.val_len[val] * (count - 1)
            if weight >= early_stop:
                weighted_counts[val] = weight
        if not weighted_counts:
            return None
        return max(weighted_counts.items(), key=lambda x: x[1])[0]

    def reorder_columns_for_value(self, row, value, column_names, grouped_rows_len: int = 1):
        cols_with_value = []
        cols_without_value = []
        
        for idx, col in enumerate(column_names):
            attr_col = col.replace(" ", "_")
            current_val = None
            
            if hasattr(row, col):
                current_val = getattr(row, col)
            elif hasattr(row, attr_col):
                current_val = getattr(row, attr_col)
            elif hasattr(row, f"_{idx}"):
                current_val = getattr(row, f"_{idx}")
            
            if pd.isna(current_val) and pd.isna(value):
                cols_with_value.append(col)
            elif current_val == value:
                cols_with_value.append(col)
            else:
                cols_without_value.append(col)
        
        if self.dep_graph is not None and grouped_rows_len > 1:
            reordered_cols = []
            for col in cols_with_value:
                reordered_cols.append(col)
                dependent_cols = self.get_dependent_columns(col)
                for dep_col in dependent_cols:
                    if dep_col in column_names and dep_col not in reordered_cols:
                        reordered_cols.append(dep_col)
            
            remaining_cols = [col for col in column_names if col not in reordered_cols]
            reordered_cols.extend(remaining_cols)
        else:
            reordered_cols = cols_with_value + cols_without_value
        
        assert len(reordered_cols) == len(column_names), f"Column count mismatch: {len(reordered_cols)} vs {len(column_names)}"
        
        reordered_vals = []
        for col in reordered_cols:
            attr_col = col.replace(" ", "_")
            if hasattr(row, col):
                reordered_vals.append(getattr(row, col))
            elif hasattr(row, attr_col):
                reordered_vals.append(getattr(row, attr_col))
            elif hasattr(row, f"_{column_names.index(col)}"):
                reordered_vals.append(getattr(row, f"_{column_names.index(col)}"))
            else:
                reordered_vals.append(None)
        
        return reordered_vals, cols_with_value

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
        return reordered_df.infer_objects(copy=False), column_orderings

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        result_df = pd.DataFrame(index=grouped_rows.index, columns=grouped_rows.columns)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, row in enumerate(grouped_rows.itertuples(index=False)):
                futures.append(executor.submit(
                    self.reorder_columns_for_value,
                    row, max_value, grouped_rows.columns.tolist(), len(grouped_rows)
                ))
            
            for i, future in enumerate(as_completed(futures)):
                reordered_row, cols_settled = future.result()
                result_df.iloc[i] = reordered_row
        
        result_df = result_df.infer_objects(copy=False)
        
        if not result_df.empty and cols_settled:
            first_col = result_df.columns[0]
            mask = result_df[first_col] == max_value
            if mask.any():
                group = result_df[mask].copy()
                if len(cols_settled) < len(result_df.columns):
                    group_remainder = group.iloc[:, len(cols_settled):]
                    if not group_remainder.empty:
                        remainder_counts = Counter(group_remainder.stack(dropna=True))
                        reordered_remainder, _ = self.recursive_reorder(
                            group_remainder, remainder_counts, early_stop=early_stop,
                            row_stop=row_stop + 1, col_stop=col_stop + 1
                        )
                        result_df.loc[mask, result_df.columns[len(cols_settled):]] = reordered_remainder.values
        
        value_counts = Counter(result_df.stack(dropna=True))
        return result_df, value_counts

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
        
        mask = df.isin([max_value]).any(axis=1)
        grouped_rows = df[mask].copy()
        remaining_rows = df[~mask].copy()
        
        if grouped_rows.empty:
            return self.fixed_reorder(df, row_sort=False)
        
        result_grouped, grouped_counts = self.column_recursion(
            pd.DataFrame(), max_value, grouped_rows, row_stop, col_stop, early_stop
        )
        
        remaining_counts = Counter()
        for val, cnt in value_counts.items():
            remaining_counts[val] = cnt - grouped_counts.get(val, 0)
        
        result_remaining, _ = self.recursive_reorder(
            remaining_rows, remaining_counts, early_stop=early_stop,
            row_stop=row_stop + 1, col_stop=col_stop
        )
        
        final_df = pd.concat([result_grouped, result_remaining], axis=0, ignore_index=True)
        final_df.columns = df.columns
        final_df = final_df.infer_objects(copy=False)
        
        column_orderings = []
        for _ in range(len(final_df)):
            column_orderings.append(final_df.columns.tolist())
        
        return final_df, column_orderings

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack(dropna=True))
            return self.recursive_reorder(df, initial_value_counts, early_stop, original_columns, row_stop=0, col_stop=0)[0]
        
        mid_index = len(df) // 2
        df_top = df.iloc[:mid_index].copy()
        df_bottom = df.iloc[mid_index:].copy()
        
        with ThreadPoolExecutor() as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top, original_columns, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom, original_columns, early_stop)
        
        reordered_top = future_top.result()
        reordered_bottom = future_bottom.result()
        
        reordered_df = pd.concat([reordered_top, reordered_bottom], axis=0, ignore_index=True)
        reordered_df.columns = df.columns
        return reordered_df.infer_objects(copy=False)

    def merging_columns(self, df: pd.DataFrame, columns: List[str], prepended: bool = True) -> pd.DataFrame:
        merged_col = '_'.join(columns)
        df[merged_col] = df[columns].astype(str).agg(' '.join, axis=1)
        df = df.drop(columns=columns)
        if prepended:
            cols = [merged_col] + [col for col in df.columns if col != merged_col]
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
        
        _, col_stats = self.calculate_col_stats(recurse_df.drop(columns=['original_index']))
        self.column_stats = {col: (ng, al, sc) for col, ng, al, sc in col_stats}
        
        if one_way_dep:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1_matches = [col for col in working_df.columns if dep[0] in col]
                col2_matches = [col for col in working_df.columns if dep[1] in col]
                if col1_matches and col2_matches:
                    self.dep_graph.add_edge(col1_matches[0], col2_matches[0])
        
        stack_vals = recurse_df.drop(columns=['original_index']).stack(dropna=True)
        initial_value_counts = Counter(stack_vals)
        self.val_len = {val: self.calculate_length(val) for val in initial_value_counts.keys()}
        
        self.row_stop = row_stop if row_stop is not None else len(recurse_df)
        self.col_stop = col_stop if col_stop is not None else len(recurse_df.columns) - 1
        
        if parallel:
            reordered_recurse = self.recursive_split_and_reorder(
                recurse_df.drop(columns=['original_index']),
                original_columns=columns_to_keep,
                early_stop=early_stop
            )
            reordered_recurse['original_index'] = recurse_df['original_index'].values
        else:
            reordered_recurse, _ = self.recursive_reorder(
                recurse_df.drop(columns=['original_index']),
                initial_value_counts,
                early_stop=early_stop
            )
            reordered_recurse['original_index'] = recurse_df['original_index'].values
        
        reordered_recurse = reordered_recurse.infer_objects(copy=False)
        
        if columns_to_discard:
            final_df = pd.merge(
                reordered_recurse, discard_df,
                on='original_index', how='left'
            )
        else:
            final_df = reordered_recurse
        
        final_df = final_df.drop(columns=['original_index'])
        
        if not col_merge:
            assert final_df.shape == initial_df.shape, \
                f"Shape mismatch: final {final_df.shape} vs initial {initial_df.shape}"
        else:
            assert final_df.shape[0] == initial_df.shape[0], \
                f"Row count mismatch: final {final_df.shape[0]} vs initial {initial_df.shape[0]}"
        
        final_df = final_df.sort_values(by=final_df.columns.tolist(), axis=0, na_position='last')
        final_df = final_df.infer_objects(copy=False)
        
        column_orderings = [final_df.columns.tolist()] * len(final_df)
        return final_df, column_orderings