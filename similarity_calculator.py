import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SimilarityCalculator:
    def __init__(self, data):
        self.data = data
    
    def calculate_similarity(self, comp_name, target_case):
        """特定コンポーネントの類似度を計算"""
        comp_data = self.data[comp_name]
        main_df = comp_data['main_data']
        param_types = comp_data['param_types']
        weights = comp_data['weights']
        
        # 対象案件のデータを取得
        target_row = main_df[main_df['案件番号'] == target_case]
        if target_row.empty:
            return None
        
        # 過去案件のデータ（対象案件より前の案件）
        target_idx = main_df[main_df['案件番号'] == target_case].index[0]
        past_df = main_df.iloc[:target_idx]
        print(target_case, target_idx, past_df)

        if past_df.empty:
            return None
        
        # 類似度計算
        similarities = []
        
        for idx, past_row in past_df.iterrows():
            sim_scores = {}
            total_sim = 0
            total_weight = 0
            
            for param, param_type in param_types.items():
                weight = weights[weights['設計諸元'] == param]['重み'].values[0]
                
                if param_type == 'category':
                    # カテゴリデータの類似度
                    sim = 1.0 if past_row[param] == target_row[param].values[0] else 0.0
                else:
                    # 数値データの類似度
                    # 全データでの最大最小値を使用して正規化
                    all_values = main_df[param].values
                    min_val = all_values.min()
                    max_val = all_values.max()
                    
                    if max_val != min_val:
                        past_norm = (past_row[param] - min_val) / (max_val - min_val)
                        target_norm = (target_row[param].values[0] - min_val) / (max_val - min_val)
                        distance = abs(past_norm - target_norm)
                        sim = 1.0 - distance
                    else:
                        sim = 1.0
                
                sim_scores[param] = sim
                total_sim += sim * weight
                total_weight += weight
            
            # 重み付き平均類似度
            avg_sim = total_sim / total_weight if total_weight > 0 else 0
            
            sim_record = {
                '案件番号': past_row['案件番号'],
                '総合類似度': avg_sim
            }
            sim_record.update(sim_scores)
            similarities.append(sim_record)
        
        # 類似度でソート
        sim_df = pd.DataFrame(similarities)
        sim_df = sim_df.sort_values('総合類似度', ascending=False)
        
        # 順位を追加
        sim_df['順位'] = range(1, len(sim_df) + 1)
        
        return {
            'similarity_df': sim_df,
            'past_cases_df': past_df.merge(
                sim_df[['案件番号', '順位', '総合類似度']], 
                on='案件番号'
            ).sort_values('順位'),
            'target_case_df': target_row
        }
    
    def calculate_all_similarities(self, target_case):
        """すべてのコンポーネントの類似度を計算"""
        results = {}
        
        for comp_name in self.data.keys():
            result = self.calculate_similarity(comp_name, target_case)
            if result:
                results[comp_name] = result
        
        return results
    
    def create_ranking_summary(self, similarities):
        """コンポーネント別の順位一覧を作成"""
        summary_data = []
        
        for comp_name, sim_data in similarities.items():
            sim_df = sim_data['similarity_df']
            for _, row in sim_df.iterrows():
                summary_data.append({
                    '案件番号': row['案件番号'],
                    comp_name: row['順位'],
                })
        # データフレームの作成とマージ
        if summary_data:
            df = pd.DataFrame(summary_data)
            # 案件番号でグループ化して集約
            result_df = df.groupby('案件番号').first().reset_index()
            return result_df
        else:
            return pd.DataFrame()
