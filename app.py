import streamlit as st
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
    st.title("仮")
    
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
            unique_cases = sorted(list(set(all_cases)),reverse=True)
            
            target_case = st.selectbox("対象案件を選択", unique_cases)
            st.session_state.target_case = target_case
            
            # # 重みの編集
            # st.subheader("重みの編集")
            # for comp_name, comp_data in st.session_state.data.items():
            #     st.write(f"**{comp_name}**")
            #     weights = comp_data['weights'].copy()
            #     for idx, row in weights.iterrows():
            #         param = row['設計諸元']
            #         current_weight = row['重み']
            #         new_weight = st.number_input(
            #             f"{param}",
            #             value=float(current_weight),
            #             min_value=0.0,
            #             max_value=10.0,
            #             step=0.1,
            #             key=f"weight_{comp_name}_{param}"
            #         )
            #         weights.loc[idx, '重み'] = new_weight
            #     st.session_state.data[comp_name]['weights'] = weights
    
    # メインエリア
    if 'data' in st.session_state and st.session_state.data and 'target_case' in st.session_state:
        # 類似度計算
        calculator = SimilarityCalculator(st.session_state.data)
        similarities = calculator.calculate_all_similarities(st.session_state.target_case)
        
        # 結果の表示
        visualizer = Visualizer()
        
        # 全体の順位表
        st.subheader("コンポーネント別順位一覧")
        ranking_df = calculator.create_ranking_summary(similarities)
        styled_ranking_df = visualizer.style_ranking_table(ranking_df)
        st.dataframe(styled_ranking_df, height=150, use_container_width=True)

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
                    
                    # 対象案件データ
                    st.write("### 対象案件データ")
                    target_df = similarities[comp_name]['target_case_df']
                    st.dataframe(target_df, use_container_width=True)

                    # 過去案件データ
                    st.write("### 過去案件データ")
                    past_df = similarities[comp_name]['past_cases_df']
                    st.dataframe(past_df, use_container_width=True)
                    
                else:
                    st.warning(f"{comp_name} のデータが見つかりません")
        

    
    else:
        st.info("サイドバーから「データを読み込む」をクリックしてください")

if __name__ == "__main__":
    main()
