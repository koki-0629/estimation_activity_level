# ファイル名: data_augmentor.py

import itertools
import pandas as pd
import numpy as np

def create_augmented_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    df:
    - 以下の列が存在することを想定
        zncc_90_dx, zncc_90_dy, zncc_90_dz,
        zncc_50_dx, zncc_50_dy, zncc_50_dz,
        centroid_50, centroid_90, cluster
    - cluster列はラベル(例: 0,1など)を想定
    
    機能:
    - df 内の全ての行ペア(2行)について、数値列の絶対差分を計算
    - 2行の cluster 値が同じかどうかで cluster=1 (同クラスタ) or 0 (別クラスタ)
    - それを新しい行として積み重ね、拡張後の DataFrame を作成
    """
    # 数値列 (cluster以外)
    feature_cols = [
        "zncc_90_dx","zncc_90_dy","zncc_90_dz",
        "zncc_50_dx","zncc_50_dy","zncc_50_dz",
        "centroid_50","centroid_90"
    ]

    rows_to_add = []

    # df.index の全組み合わせからペアを取得
    for idx1, idx2 in itertools.combinations(df.index, 2):
        # 数値列の絶対差分
        diff_values = abs(df.loc[idx1, feature_cols] - df.loc[idx2, feature_cols])
        # cluster値が一致するか判定 (同じなら1, 違うなら0)
        cluster_value = (
            1 if df.loc[idx1, 'cluster'] == df.loc[idx2, 'cluster'] else 0
        )
        # 新しい行（差分 + cluster）
        new_row = list(diff_values.values) + [cluster_value]
        rows_to_add.append(new_row)

    # 新しいDataFrameを作成(列名は feature_cols + ['cluster'])
    new_columns = feature_cols + ['cluster']
    df_aug = pd.DataFrame(rows_to_add, columns=new_columns)
    
    return df_aug
