import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
    
    def load_all_data(self):
        """すべてのコンポーネントデータを読み込む"""
        data = {}
        
        for excel_file in sorted(self.data_dir.glob('*.xlsx')):
            comp_name = excel_file.stem
            
            # メインデータの読み込み
            main_df = pd.read_excel(excel_file, sheet_name='案件データ')
            
            # メタデータの読み込み
            meta_df = pd.read_excel(excel_file, sheet_name='メタデータ')
            
            # 除外フラグのチェック
            if '除外フラグ' in main_df.columns:
                # X*で始まる案件を除外
                main_df = main_df[~main_df['除外フラグ'].str.startswith('X', na=False)].reset_index(drop=True)
            
            data[comp_name] = {
                'main_data': main_df,
                'param_types': dict(zip(meta_df['設計諸元'], meta_df['型'])),
                'weights': meta_df
            }
        
        return data
    
    def update_data(self, comp_name, new_data):
        """データを更新"""
        excel_file = self.data_dir / f"{comp_name}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            new_data.to_excel(writer, sheet_name='案件データ', index=False)
