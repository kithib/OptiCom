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
        cols_without_value = []
        for idx, col in enumerate(column_names):
            current_val = None
            if hasattr(row, col):
                current_val = getattr(row, col)
            elif hasattr(row, col.replace(" ", "_")):
                current_val = getattr(row, col.replace(" ", "_"))
            elif hasattr(row, f"_{idx}"):
                current_val = getattr(row, f"_{idx}")
            
            if current_val == value:
                cols_with_value.append(col)
            else:
                cols_without_value.append(col)

        if self.dep_graph is not None and grouped_rows_len > 1:
            reordered_cols = []
            for col in cols_with_value:
                dependent_cols = self.get_cached_dependent_columns(col)
                valid_dependent_cols = [dc for dc in dependent_cols if dc in column_names]
                reordered_cols.extend([col] + valid_dependent_cols)
            
            reordered_cols = list(dict.fromkeys(reordered_cols))
            cols_without_value = [col for col in column_names if col not in reordered_cols]
            reordered_cols.extend(cols_without_value)
        else:
            reordered_cols = cols_with_value + cols_without_value

        assert len(reordered_cols) == len(column_names), f"Reordered cols len: {len(reordered_cols)} != Original cols len: {len(column_names)}"
        
        reordered_vals = []
        for col in reordered_cols:
            if hasattr(row, col):
                reordered_vals.append(getattr(row, col))
            elif hasattr(row, col.replace(" ", "_")):
                reordered_vals.append(getattr(row, col.replace(" ", "_")))
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
        if df.empty:
            return df, []
        
        column_scores = []
        for col in df.columns:
            if col == "original_index":
                continue
            nunique = df[col].nunique()
            avg_len = df[col].astype(str).apply(len).mean()
            score = (len(df) - nunique) * (avg_len ** 2)
            column_scores.append((col, score))
        
        column_scores.sort(key=lambda x: x[1], reverse=True)
        reordered_columns = [col for col, _ in column_scores]
        if "original_index" in df.columns:
            reordered_columns.append("original_index")
        
        reordered_df = df[reordered_columns].copy()
        
        if row_sort:
            sort_cols = [col for col in reordered_columns if col != "original_index"]
            reordered_df = reordered_df.sort_values(by=sort_cols, axis=0)
        
        column_orderings = [reordered_columns] * len(reordered_df)
        return reordered_df, column_orderings

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        if grouped_rows.empty:
            return result_df, Counter()
        
        column_names = grouped_rows.columns.tolist()
        result_df = pd.DataFrame(index=grouped_rows.index, columns=column_names)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, row in enumerate(grouped_rows.itertuples(index=False)):
                futures.append(executor.submit(
                    self.reorder_columns_for_value, 
                    row, 
                    max_value, 
                    column_names, 
                    len(grouped_rows)
                ))
            
            for i, future in enumerate(as_completed(futures)):
                reordered_row, cols_settled = future.result()
                result_df.loc[grouped_rows.index[i]] = reordered_row

        grouped_value_counts = Counter(grouped_rows.stack())

        if not result_df.empty and cols_settled:
            current_settle_cols = cols_settled[:1]
            if current_settle_cols:
                first_col = current_settle_cols[0]
                same_value_mask = result_df[first_col] == max_value
                same_value_group = result_df[same_value_mask].copy()
                
                if len(same_value_group) > 1 and len(result_df.columns) > len(current_settle_cols):
                    group_remainder = same_value_group.iloc[:, len(current_settle_cols):]
                    
                    if not group_remainder.empty:
                        remainder_value_counts = Counter(group_remainder.stack())
                        reordered_group_remainder, _ = self.recursive_reorder(
                            group_remainder,
                            remainder_value_counts,
                            early_stop=early_stop,
                            row_stop=row_stop + 1,
                            col_stop=col_stop + 1
                        )
                        
                        target_cols = result_df.columns[len(current_settle_cols):]
                        result_df.loc[same_value_group.index, target_cols] = reordered_group_remainder.values.astype(result_df[target_cols].dtypes)

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

        has_max_value = df.isin([max_value]).any(axis=1)
        grouped_rows = df[has_max_value].copy()
        remaining_rows = df[~has_max_value].copy()

        if grouped_rows.empty:
            return self.fixed_reorder(df)

        result_df, grouped_value_counts = self.column_recursion(
            pd.DataFrame(), max_value, grouped_rows, row_stop, col_stop, early_stop
        )

        remaining_value_counts = Counter()
        for val, cnt in value_counts.items():
            remaining_cnt = cnt - grouped_value_counts.get(val, 0)
            if remaining_cnt > 0:
                remaining_value_counts[val] = remaining_cnt

        reordered_remaining_rows, _ = self.recursive_reorder(
            remaining_rows, remaining_value_counts, early_stop=early_stop, row_stop=row_stop + 1, col_stop=col_stop
        )

        final_result_df = pd.concat([result_df, reordered_remaining_rows], axis=0).sort_index()
        
        column_orderings = []
        for idx in final_result_df.index:
            if idx in result_df.index:
                column_orderings.append(result_df.columns.tolist())
            else:
                column_orderings.append(reordered_remaining_rows.columns.tolist())

        return final_result_df, column_orderings

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack())
            reordered_df, _ = self.recursive_reorder(df, initial_value_counts, early_stop, original_columns, row_stop=0, col_stop=0)
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

    def calculate_col_stats(self, df: pd.DataFrame, enable_index: bool = False) -> Tuple[int, List]:
        stats = []
        for col in df.columns:
            if enable_index and col == df.index.name:
                continue
            nunique = df[col].nunique()
            avg_len = df[col].astype(str).apply(len).mean()
            score = (len(df) - nunique) * (avg_len ** 2)
            stats.append((col, nunique, avg_len, score))
        
        stats.sort(key=lambda x: x[3], reverse=True)
        return len(df), stats

    def merging_columns(self, df: pd.DataFrame, cols_to_merge: List[str], prepended: bool = True) -> pd.DataFrame:
        if not cols_to_merge or len(cols_to_merge) < 2:
            return df
        
        merged_col_name = '_'.join(cols_to_merge)
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
            for merge_group in col_merge:
                valid_cols = [col for col in working_df.columns if col in merge_group]
                if len(valid_cols) >= 2:
                    working_df = self.merging_columns(working_df, valid_cols)

        self.num_rows, self.column_stats = self.calculate_col_stats(working_df)
        self.column_stats = {col: (nunique, avg_len, score) for col, nunique, avg_len, score in self.column_stats}

        if one_way_dep:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1_matches = [col for col in working_df.columns if dep[0] in col]
                col2_matches = [col for col in working_df.columns if dep[1] in col]
                if col1_matches and col2_matches:
                    self.dep_graph.add_edge(col1_matches[0], col2_matches[0])

        nunique_threshold = len(working_df) * distinct_value_threshold
        columns_to_discard = [
            col for col in working_df.columns 
            if working_df[col].nunique() > nunique_threshold and col != "original_index"
        ]
        columns_to_recurse = [col for col in working_df.columns if col not in columns_to_discard]

        working_df["original_index"] = range(len(working_df))
        discarded_columns_df = working_df[columns_to_discard + ["original_index"]].copy()
        recurse_df = working_df[columns_to_recurse + ["original_index"]].copy()

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

        assert reordered_df.shape == recurse_df.shape, f"Shape mismatch: {reordered_df.shape} vs {recurse_df.shape}"
        assert recurse_df["original_index"].is_unique, "Duplicate original indices in input!"
        assert reordered_df["original_index"].is_unique, "Duplicate original indices after reorder!"

        if columns_to_discard:
            final_df = pd.merge(reordered_df, discarded_columns_df, on="original_index", how="left")
        else:
            final_df = reordered_df.copy()

        final_df = final_df.drop(columns=["original_index"])

        if not col_merge:
            assert final_df.shape == initial_df.shape, f"Final shape {final_df.shape} != initial shape {initial_df.shape}"
        else:
            assert final_df.shape[0] == initial_df.shape[0], f"Row count mismatch: {final_df.shape[0]} vs {initial_df.shape[0]}"

        sort_cols = [col for col in final_df.columns if col != "original_index"]
        final_df = final_df.sort_values(by=sort_cols, axis=0)

        column_orderings = [final_df.columns.tolist() for _ in range(len(final_df))]
        return final_df, column_orderings