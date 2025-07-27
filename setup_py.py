import os
import random
import pandas as pd
import numpy as np
from openpyxl import Workbook
from pathlib import Path

def create_directory_structure():
    """プロジェクトのディレクトリ構造を作成"""
    directories = [
        'src',
        'data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # __init__.pyファイルを作成
    Path('src/__init__.py').touch()
    print("ディレクトリ構造を作成しました。")

def generate_sample_data():
    """サンプルデータを生成"""
    np.random.seed(42)
    
    # コンポーネント定義
    components = {
        '01_component_A': {
            'params': ['材料', '厚さ(mm)', '表面処理', '硬度(HRC)'],
            'types': ['category', 'numeric', 'category', 'numeric'],
            'categories': {
                '材料': ['アルミ', 'ステンレス', '銅', '鉄'],
                '表面処理': ['メッキ', '塗装', 'アルマイト', '無処理']
            }
        },
        '02_component_B': {
            'params': ['構造', '長さ(mm)', '幅(mm)'],
            'types': ['category', 'numeric', 'numeric'],
            'categories': {
                '構造': ['中空', '中実', 'ハニカム']
            }
        },
        '03_component_C': {
            'params': ['接続方式', '耐圧(MPa)', '流量(L/min)', '材質'],
            'types': ['category', 'numeric', 'numeric', 'category'],
            'categories': {
                '接続方式': ['ねじ込み', 'フランジ', 'カプラー'],
                '材質': ['樹脂', '金属', 'ゴム']
            }
        },
        '04_component_D': {
            'params': ['制御方式', '電圧(V)', '応答時間(ms)'],
            'types': ['category', 'numeric', 'numeric'],
            'categories': {
                '制御方式': ['アナログ', 'デジタル', 'PWM', 'PID']
            }
        }
    }
    
    # 各コンポーネントのデータを生成
    for comp_name, comp_info in components.items():
        # 案件データを生成
        data = {'案件番号': [f'P{str(i).zfill(4)}' for i in range(1, 51)]}
        
        for param, param_type in zip(comp_info['params'], comp_info['types']):
            if param_type == 'category':
                data[param] = np.random.choice(comp_info['categories'][param], 50)
            else:
                if '厚さ' in param:
                    data[param] = np.round(np.random.uniform(0.5, 10, 50), 1)
                elif '長さ' in param:
                    data[param] = np.round(np.random.uniform(100, 1000, 50), 0)
                elif '幅' in param:
                    data[param] = np.round(np.random.uniform(50, 500, 50), 0)
                elif '硬度' in param:
                    data[param] = np.round(np.random.uniform(20, 65, 50), 0)
                elif '耐圧' in param:
                    data[param] = np.round(np.random.uniform(1, 20, 50), 1)
                elif '流量' in param:
                    data[param] = np.round(np.random.uniform(10, 200, 50), 0)
                elif '電圧' in param:
                    data[param] = np.random.choice([12, 24, 48, 100, 200], 50)
                elif '応答時間' in param:
                    data[param] = np.round(np.random.uniform(1, 100, 50), 0)
        
        # 不適切案件フラグ（component_Aのみ）
        if comp_name == '01_component_A':
            exclude_flags = [''] * 50
            exclude_indices = np.random.choice(50, 5, replace=False)
            for idx in exclude_indices:
                exclude_flags[idx] = f'X{random.choice(["設計不良", "生産中止", "品質問題"])}'
            data['除外フラグ'] = exclude_flags
        
        df_main = pd.DataFrame(data)
        
        # 型と重みのデータを作成
        meta_data = {
            '設計諸元': comp_info['params'],
            '型': comp_info['types'],
            '重み': [1.0] * len(comp_info['params'])
        }
        df_meta = pd.DataFrame(meta_data)
        
        # Excelファイルに書き込み
        filepath = f'data/{comp_name}.xlsx'
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df_main.to_excel(writer, sheet_name='案件データ', index=False)
            df_meta.to_excel(writer, sheet_name='メタデータ', index=False)
        
        print(f"{filepath} を作成しました。")

def create_source_files():
    """ソースコードファイルを作成"""
    
    # app.py
    app_code = '''import streamlit as st
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.similarity_calculator import SimilarityCalculator
from src.visualization import Visualizer
from src.utils import initialize_session_state

def main():
    st.set_page_config(page_title="製品案件類似度分析", layout="wide")
    st.title("製品案件類似度分析システム")
    
    # セッションステートの初期化
    initialize_session_state()
    
    # データローダーの初期化
    data_loader = DataLoader()
    
    # サイドバーでの設定
    with st.sidebar:
        st.header("設定")
        
        # データの読み込み
        if st.button("データを読み込む"):
            try:
                st.session_state.data = data_loader.load_all_data()
                st.success("データを読み込みました")
            except Exception as e:
                st.error(f"データ読み込みエラー: {str(e)}")
        
        if 'data' in st.session_state and st.session_state.data:
            # 対象案件の選択
            all_cases = []
            for comp_data in st.session_state.data.values():
                all_cases.extend(comp_data['main_data']['案件番号'].tolist())
            unique_cases = sorted(list(set(all_cases)))
            
            target_case = st.selectbox("対象案件を選択", unique_cases)
            st.session_state.target_case = target_case
            
            # 重みの編集
            st.subheader("重みの編集")
            for comp_name, comp_data in st.session_state.data.items():
                st.write(f"**{comp_name}**")
                weights = comp_data['weights'].copy()
                for idx, row in weights.iterrows():
                    param = row['設計諸元']
                    current_weight = row['重み']
                    new_weight = st.number_input(
                        f"{param}",
                        value=float(current_weight),
                        min_value=0.0,
                        max_value=10.0,
                        step=0.1,
                        key=f"weight_{comp_name}_{param}"
                    )
                    weights.loc[idx, '重み'] = new_weight
                st.session_state.data[comp_name]['weights'] = weights
    
    # メインエリア
    if 'data' in st.session_state and st.session_state.data and 'target_case' in st.session_state:
        # 類似度計算
        calculator = SimilarityCalculator(st.session_state.data)
        similarities = calculator.calculate_all_similarities(st.session_state.target_case)
        
        # 結果の表示
        visualizer = Visualizer()
        
        # タブで各コンポーネントの結果を表示
        tabs = st.tabs(list(st.session_state.data.keys()))
        
        for tab, comp_name in zip(tabs, st.session_state.data.keys()):
            with tab:
                if comp_name in similarities:
                    st.subheader(f"{comp_name} の類似度分析結果")
                    
                    # 類似度テーブル
                    st.write("### 類似度テーブル")
                    sim_df = similarities[comp_name]['similarity_df']
                    styled_sim_df = visualizer.style_similarity_table(sim_df)
                    st.dataframe(styled_sim_df, use_container_width=True)
                    
                    # 過去案件データ
                    st.write("### 過去案件データ（上位10件）")
                    past_df = similarities[comp_name]['past_cases_df'].head(10)
                    st.dataframe(past_df, use_container_width=True)
                    
                    # 対象案件データ
                    st.write("### 対象案件データ")
                    target_df = similarities[comp_name]['target_case_df']
                    st.dataframe(target_df, use_container_width=True)
                else:
                    st.warning(f"{comp_name} のデータが見つかりません")
        
        # 全体の順位表
        st.subheader("コンポーネント別順位一覧")
        ranking_df = calculator.create_ranking_summary(similarities)
        styled_ranking_df = visualizer.style_ranking_table(ranking_df)
        st.dataframe(styled_ranking_df, use_container_width=True)
    
    else:
        st.info("サイドバーから「データを読み込む」をクリックしてください")

if __name__ == "__main__":
    main()
'''
    
    # data_loader.py
    data_loader_code = '''import pandas as pd
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
                main_df = main_df[~main_df['除外フラグ'].str.startswith('X', na=False)]
            
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
'''
    
    # similarity_calculator.py
    similarity_calculator_code = '''import pandas as pd
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
                    comp_name: row['順位']
                })
        
        # データフレームの作成とマージ
        if summary_data:
            df = pd.DataFrame(summary_data)
            # 案件番号でグループ化して集約
            result_df = df.groupby('案件番号').first().reset_index()
            return result_df
        else:
            return pd.DataFrame()
'''
    
    # visualization.py
    visualization_code = '''import pandas as pd
import numpy as np

class Visualizer:
    def style_similarity_table(self, df):
        """類似度テーブルのスタイリング"""
        def highlight_similarity(val):
            if isinstance(val, (int, float)):
                if val >= 0.8:
                    return 'background-color: #90EE90'  # 薄緑
                elif val >= 0.6:
                    return 'background-color: #FFFFE0'  # 薄黄
                elif val >= 0.4:
                    return 'background-color: #FFE4B5'  # 薄オレンジ
                else:
                    return 'background-color: #FFB6C1'  # 薄赤
            return ''
        
        # 数値列のみにスタイルを適用
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        styled_df = df.style.applymap(highlight_similarity, subset=numeric_cols)
        
        return styled_df
    
    def style_ranking_table(self, df):
        """順位テーブルのスタイリング"""
        def highlight_rank(val):
            if isinstance(val, (int, float)):
                if val <= 3:
                    return 'background-color: #90EE90'  # 薄緑
                elif val <= 10:
                    return 'background-color: #FFFFE0'  # 薄黄
                elif val <= 20:
                    return 'background-color: #FFE4B5'  # 薄オレンジ
                else:
                    return 'background-color: #FFB6C1'  # 薄赤
            return ''
        
        # 順位列のみにスタイルを適用
        rank_cols = [col for col in df.columns if col != '案件番号']
        styled_df = df.style.applymap(highlight_rank, subset=rank_cols)
        
        return styled_df
'''
    
    # utils.py
    utils_code = '''import streamlit as st

def initialize_session_state():
    """セッションステートの初期化"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'target_case' not in st.session_state:
        st.session_state.target_case = None
    
    if 'weights_updated' not in st.session_state:
        st.session_state.weights_updated = False
'''
    
    # ファイルの書き込み
    files = {
        'src/app.py': app_code,
        'src/data_loader.py': data_loader_code,
        'src/similarity_calculator.py': similarity_calculator_code,
        'src/visualization.py': visualization_code,
        'src/utils.py': utils_code
    }
    
    for filepath, content in files.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"{filepath} を作成しました。")

def create_requirements():
    """requirements.txtを作成"""
    requirements = '''streamlit==1.28.1
pandas==2.1.1
numpy==1.25.2
openpyxl==3.1.2
scikit-learn==1.3.1
'''
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    print("requirements.txt を作成しました。")

def create_readme():
    """README.mdを作成"""
    readme = '''# 製品案件類似度分析システム

## 概要
製品の案件情報から、指定した案件と過去案件との類似度を計算し、コンポーネントごとの順位を表示するStreamlitアプリケーションです。

## フォルダ構成
```
project/
├── src/
│   ├── __init__.py
│   ├── app.py              # メインアプリケーション
│   ├── data_loader.py      # データ読み込みモジュール
│   ├── similarity_calculator.py  # 類似度計算モジュール
│   ├── visualization.py    # 可視化モジュール
│   └── utils.py           # ユーティリティ関数
├── data/
│   ├── 01_component_A.xlsx  # コンポーネントAのデータ
│   ├── 02_component_B.xlsx  # コンポーネントBのデータ
│   ├── 03_component_C.xlsx  # コンポーネントCのデータ
│   └── 04_component_D.xlsx  # コンポーネントDのデータ
├── requirements.txt        # 依存パッケージ
├── README.md              # このファイル
└── setup.py               # セットアップスクリプト
```

## セットアップ

### 1. プロジェクトの初期セットアップ
```bash
python setup.py
```

### 2. 仮想環境の作成と有効化（推奨）
```bash
python -m venv venv
venv\\Scripts\\activate  # Windows
```

### 3. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

## 実行方法

### アプリケーションの起動
```bash
streamlit run src/app.py
```

ブラウザが自動的に開き、アプリケーションが表示されます。

## 使用方法

1. **データの読み込み**
   - サイドバーの「データを読み込む」ボタンをクリック

2. **対象案件の選択**
   - サイドバーのドロップダウンから分析したい案件を選択

3. **重みの調整**（オプション）
   - 各設計諸元の重みを0.0〜10.0の範囲で調整可能

4. **結果の確認**
   - 各コンポーネントタブで類似度分析結果を確認
   - 類似度に応じて色分けされたテーブルが表示される
     - 緑：高い類似度/上位順位
     - 黄：中程度の類似度/順位
     - オレンジ：低い類似度/順位
     - 赤：非常に低い類似度/下位順位

## データ形式

### 案件データシート
- 案件番号：P0001〜P0050
- 設計諸元：コンポーネントごとに異なる
- 除外フラグ：X*で始まる場合は分析から除外（01_component_A.xlsxのみ）

### メタデータシート
- 設計諸元：パラメータ名
- 型：category（カテゴリ）またはnumeric（数値）
- 重み：初期値1.0

## 類似度計算ロジック
- カテゴリデータ：一致で1、不一致で0
- 数値データ：正規化後の距離による類似度（0〜1）
- 総合類似度：重み付き平均

## 実行例
```bash
# Windowsでの実行例
C:\\project> python setup.py
ディレクトリ構造を作成しました。
data/01_component_A.xlsx を作成しました。
data/02_component_B.xlsx を作成しました。
data/03_component_C.xlsx を作成しました。
data/04_component_D.xlsx を作成しました。
...

C:\\project> pip install -r requirements.txt
...

C:\\project> streamlit run src/app.py
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```
'''
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    print("README.md を作成しました。")

def main():
    """メイン処理"""
    print("プロジェクトのセットアップを開始します...")
    
    # 1. ディレクトリ構造の作成
    create_directory_structure()
    
    # 2. サンプルデータの生成
    generate_sample_data()
    
    # 3. ソースファイルの作成
    create_source_files()
    
    # 4. requirements.txtの作成
    create_requirements()
    
    # 5. README.mdの作成
    create_readme()
    
    print("\\nセットアップが完了しました！")
    print("次のコマンドでアプリケーションを起動できます：")
    print("1. pip install -r requirements.txt")
    print("2. streamlit run src/app.py")

if __name__ == "__main__":
    main()
