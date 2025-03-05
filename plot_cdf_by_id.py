import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データ拡張モジュール (既存)
from data_augmentor import create_augmented_dataset

def drop_unwanted_columns(df, cols_to_drop):
    """不要な列を削除 (存在しない列は errors='ignore' で無視)"""
    if cols_to_drop:
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    return df

def compute_discrete_cdf(values: np.ndarray):
    """
    離散データを想定したCDF(経験的累積分布関数)を「x, cdf」の形で返す。
    - x: 昇順にソートされたユニーク値
    - cdf: その値以下となる確率(サンプル比率)
    """
    clean_vals = values[~np.isnan(values)]
    if len(clean_vals) == 0:
        return np.array([]), np.array([])
    
    sorted_vals = np.sort(clean_vals)
    unique_vals = np.unique(sorted_vals)
    cdf_vals = []
    n = len(sorted_vals)
    for uv in unique_vals:
        cdf_vals.append(np.sum(sorted_vals <= uv) / n)
    return unique_vals, np.array(cdf_vals)

def main():
    """
    【実行例】
      python3 plot_cdf_by_id.py test_dataset.csv --id_col ID --drop_cols colA colB --output_dir cdf_by_id_plots

    概要:
    1) テストデータ (CSV) を読み込む
    2) 不要列削除
    3) データ拡張
    4) ID列で groupby し、グループごとに特徴分布のCDFを作成
    5) 同じ特徴列について複数グループのCDFを1枚に描画
    6) 画像出力
    """
    parser = argparse.ArgumentParser(description="Plot discrete CDF of each feature, grouped by ID, after data augmentation.")
    parser.add_argument("test_csv", help="Path to the test dataset CSV.")
    parser.add_argument("--id_col", default="ID", help="Column name representing data group/ID (e.g., for different acquisition times).")
    parser.add_argument("--drop_cols", nargs="*", default=[],
                        help="Columns to drop from the test dataset before augmentation.")
    parser.add_argument("--output_dir", default="cdf_by_id_plots",
                        help="Directory to save the resulting CDF plots.")
    args = parser.parse_args()

    test_csv = args.test_csv
    id_col = args.id_col
    drop_cols = args.drop_cols
    output_dir = args.output_dir

    # 1. CSV読み込み
    if not os.path.exists(test_csv):
        print(f"Error: test_csv='{test_csv}' not found.")
        return
    df_test = pd.read_csv(test_csv)
    print(f"[Test] shape={df_test.shape}")

    # 2. 不要列削除
    df_test = drop_unwanted_columns(df_test, drop_cols)
    print(f"[Test after drop] shape={df_test.shape}")

    # 3. データ拡張
    df_aug = create_augmented_dataset(df_test)
    print(f"[Augmented] shape={df_aug.shape}")

    # ID列が存在するかチェック
    if id_col not in df_aug.columns:
        print(f"Error: ID column '{id_col}' not found in augmented data.")
        return

    # 4. 数値列を抽出 (cluster などCDFに使わない列は除外する場合)
    all_cols = set(df_aug.columns)
    # cluster列など除外したい列を減らす (お好みで調整)
    if "cluster" in all_cols:
        all_cols.remove("cluster")
    # ID列は分布に使わないので除外
    all_cols.discard(id_col)

    # 数値型のみ
    numeric_cols = set(df_aug.select_dtypes(include=[np.number]).columns)
    # augmentedデータにおける特徴列 (IDでもclusterでもない数値列)
    feature_cols = all_cols.intersection(numeric_cols)
    if not feature_cols:
        print("No numeric features found to plot.")
        return

    # 5. groupby で ID ごとのサブセットを用意
    grouped = df_aug.groupby(id_col)

    # 出力先ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # 6. 各特徴について、グループごとのCDFを1枚のグラフに重ね描画
    for feat in sorted(feature_cols):
        plt.figure(figsize=(7,5))
        plt.title(f"CDF of '{feat}' by '{id_col}' group")
        plt.xlabel(feat)
        plt.ylabel("CDF")
        plt.grid(True)

        # グループ名ごとに線を追加
        for group_id, subdf in grouped:
            # subdfには当該IDの拡張済みデータが入る
            vals = subdf[feat].dropna().values
            x_vals, cdf_vals = compute_discrete_cdf(vals)

            # 離散CDFを線で描いてもよいし、点のみでもよい
            # ここではステッププロット or 点のみを選択:
            # 例: 点のみ
            plt.plot(x_vals, cdf_vals, marker='o', linestyle='none', label=f"{group_id} (n={len(vals)})")
            
            # ステッププロットにする場合は下記を使用:
            # plt.step(x_vals, cdf_vals, where='post', label=f"{group_id} (n={len(vals)})")

        plt.legend(loc="best")

        # 画像保存
        out_path = os.path.join(output_dir, f"{feat}_cdf_by_{id_col}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {out_path}")

if __name__ == "__main__":
    main()
