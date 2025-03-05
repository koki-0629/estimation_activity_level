import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_augmentor import create_augmented_dataset

def drop_unwanted_columns(df, cols_to_drop):
    """不要な列を削除 (存在しない列は errors='ignore' で無視)"""
    if cols_to_drop:
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    return df

def compute_discrete_cdf(values: np.ndarray):
    """
    離散データを想定したCDF(累積分布関数)を「点列」として返す関数。
      - unique_vals: 昇順にソートしたユニーク値
      - cdf_vals   : 各ユニーク値 v_i に対して P(X <= v_i)
    """
    # NaNなど除外
    clean_vals = values[~np.isnan(values)]
    if len(clean_vals) == 0:
        return np.array([]), np.array([])

    # 昇順のユニーク値 (離散値)
    unique_vals = np.unique(np.sort(clean_vals))
    cdf_vals = []
    n = len(clean_vals)
    for uv in unique_vals:
        # P(X <= uv) = (uv以下の個数) / n
        cdf_vals.append(np.sum(clean_vals <= uv) / n)

    return unique_vals, np.array(cdf_vals)

def main():
    """
    【実行例】
      python3 plot_discrete_cdf.py train_dataset.csv test_dataset.csv
        --drop_cols1 colA colB
        --drop_cols2 colX colY
        --output_dir cdf_plots

    手順:
      1. トレインCSV, テストCSV を読み込み
      2. 不要列削除(各々)
      3. データ拡張( create_augmented_dataset )
      4. 両方に共通する数値列のみ抽出 (cluster 列は除外)
      5. 各列に対して、トレイン/テストの離散CDF(点のみ)を描画
         加えて、平均値(点線)と±1標準偏差(破線)を垂直線として描画
      6. PNG で保存
    """
    parser = argparse.ArgumentParser(description="Plot discrete CDF with mean & std lines (no line connecting points) for train/test augmented data.")
    parser.add_argument("train_csv", help="Path to train dataset CSV.")
    parser.add_argument("test_csv", help="Path to test dataset CSV.")
    parser.add_argument("--drop_cols1", nargs="*", default=[],
                        help="Columns to drop from the train dataset.")
    parser.add_argument("--drop_cols2", nargs="*", default=[],
                        help="Columns to drop from the test dataset.")
    parser.add_argument("--output_dir", default="cdf_plots",
                        help="Directory to save the resulting CDF plots.")
    args = parser.parse_args()

    train_csv = args.train_csv
    test_csv = args.test_csv
    drop_cols1 = args.drop_cols1
    drop_cols2 = args.drop_cols2
    output_dir = args.output_dir

    if not os.path.exists(train_csv):
        print(f"Error: train_csv='{train_csv}' not found.")
        return
    if not os.path.exists(test_csv):
        print(f"Error: test_csv='{test_csv}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. CSV読み込み
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    print(f"[Train] shape={df_train.shape}")
    print(f"[Test ] shape={df_test.shape}")

    # 2. 不要列削除
    df_train = drop_unwanted_columns(df_train, drop_cols1)
    df_test = drop_unwanted_columns(df_test, drop_cols2)
    print(f"[Train after drop] shape={df_train.shape}")
    print(f"[Test  after drop] shape={df_test.shape}")

    # 3. データ拡張 (train, test)
    train_aug = create_augmented_dataset(df_train)
    test_aug = create_augmented_dataset(df_test)
    print(f"[TrainAug] shape={train_aug.shape}")
    print(f"[TestAug ] shape={test_aug.shape}")

    # 4. 共通の数値列を抽出 (clusterはCDFに不要なら除外)
    train_cols = set(train_aug.columns)
    test_cols = set(test_aug.columns)
    common_cols = train_cols.intersection(test_cols)

    if "cluster" in common_cols:
        common_cols.remove("cluster")

    # 数値列だけに限定
    train_num = set(train_aug.select_dtypes(include=[np.number]).columns)
    test_num = set(test_aug.select_dtypes(include=[np.number]).columns)
    numeric_common_cols = common_cols.intersection(train_num).intersection(test_num)

    if not numeric_common_cols:
        print("No common numeric columns found. Nothing to plot.")
        return

    print(f"Common numeric features: {numeric_common_cols}")

    # 5. 各特徴に対して離散CDFを点のみでプロット + 平均&±1標準偏差の垂直線
    for feat in sorted(numeric_common_cols):
        # --- トレイン ---
        train_vals = train_aug[feat].dropna().values
        x_train, cdf_train = compute_discrete_cdf(train_vals)
        # 統計量
        train_mean = np.mean(train_vals) if len(train_vals) > 0 else 0
        train_std = np.std(train_vals) if len(train_vals) > 0 else 0

        # --- テスト ---
        test_vals = test_aug[feat].dropna().values
        x_test, cdf_test = compute_discrete_cdf(test_vals)
        # 統計量
        test_mean = np.mean(test_vals) if len(test_vals) > 0 else 0
        test_std = np.std(test_vals) if len(test_vals) > 0 else 0

        # 図を作成
        plt.figure(figsize=(7, 5))
        # 離散CDF (トレイン: 青丸, テスト: 赤四角), 点のみ
        plt.plot(x_train, cdf_train, marker='o', linestyle='none', color='blue',  label='Train CDF', markersize=4)
        plt.plot(x_test,  cdf_test,  marker='o', linestyle='none', color='red',   label='Test CDF', markersize=4)

        # 垂直線: 平均 (破線)、±1標準偏差 (点線) を追加
        # Train
        plt.axvline(train_mean, color='blue', linestyle='--', label='Train mean')
        # plt.axvline(train_mean + train_std, color='blue', linestyle=':', label='Train ±1 std')
        # plt.axvline(train_mean - train_std, color='blue', linestyle=':')
        # Test
        plt.axvline(test_mean, color='red', linestyle='--', label='Test mean')
        # plt.axvline(test_mean + test_std, color='red', linestyle=':', label='Test ±1 std')
        # plt.axvline(test_mean - test_std, color='red', linestyle=':')

        plt.xlabel(feat)
        plt.ylabel("CDF")
        # plt.title(f"Discrete CDF of {feat}\n(mean & ±1 std shown)")
        # plt.grid(True)
        plt.legend(loc='best')

        # 6. 保存
        out_path = os.path.join(output_dir, f"{feat}_cdf.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved discrete CDF plot with mean/std for '{feat}' -> {out_path}")

if __name__ == "__main__":
    main()
