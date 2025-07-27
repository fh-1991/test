import pandas as pd
import numpy as np

class Visualizer:
    def style_similarity_table(self, df):
        """類似度テーブルのスタイリング"""
        def highlight_similarity(val):
            if isinstance(val, (int, float)):
                if val >= 0.8:
                    return 'color: black; background-color: #90EE90'  # 薄緑
                elif val >= 0.6:
                    return 'color: black; background-color: #FFFFE0'  # 薄黄
                elif val >= 0.4:
                    return 'color: black; background-color: #FFE4B5'  # 薄オレンジ
                else:
                    return 'color: black; background-color: #FFB6C1'  # 薄赤
            return ''
        
        # 数値列のみにスタイルを適用
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        styled_df = df.style.applymap(highlight_similarity, subset=numeric_cols)
        
        return styled_df
    
    def style_ranking_table(self, df):
        """順位テーブルのスタイリング"""
        def compact_rank(val):
            if isinstance(val, (int, float)):
                return f"{val:.0f}"
            
        def highlight_rank(val):
            if isinstance(val, (int, float)):
                if val <= 3:
                    return 'color: black; background-color: #90EE90'  # 薄緑
                elif val <= 10:
                    return 'color: black; background-color: #FFFFE0'  # 薄黄
                elif val <= 20:
                    return 'color: black; background-color: #FFE4B5'  # 薄オレンジ
                else:
                    return 'color: black; background-color: #FFB6C1'  # 薄赤
            return ''
        
        # 順位列のみにスタイルを適用
        rank_cols = [col for col in df.columns if col != '案件番号']
        styled_df = df.style\
            .format({col: compact_rank for col in rank_cols})\
            .applymap(highlight_rank, subset=rank_cols)
        
        return styled_df
