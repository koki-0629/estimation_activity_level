import argparse
import pandas as pd
import numpy as np
import os

# XGBoost
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# データ拡張モジュール
from data_augmentor import create_augmented_dataset

# 可視化ライブラリ
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred):
    """
    評価指標(Accuracy, FNR, FPR, Precision, Recall, F1)および混同行列を計算し、辞書で返す。
    FNR, FPRは混同行列から算出:
    TN, FP, FN, TP = confusion_matrix(...).ravel()
    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "Accuracy": accuracy,
        "FNR": fnr,
        "FPR": fpr,
        "Precision": precision,
        "Recall": recall,
        "F1score": f1,
        # 混同行列をリスト形式で保存
        "ConfusionMatrix": cm.tolist()
    }

def drop_unwanted_columns(df, cols_to_drop):
    """不要列を削除 (存在しない列は無視)"""
    if cols_to_drop:
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    return df

def train_and_evaluate_xgb_default(X_train, y_train, X_test, y_test, feature_names):
    """
    XGBoost (デフォルトパラメータ) を学習し、テストデータで評価。
    交差検証(cv=5)のスコア(accuracy)も表示。
    """
    xgb_default = XGBClassifier(
        use_label_encoder=False,  # v1.3以降warning回避
        eval_metric='logloss',
        random_state=42
    )

    # cv=5 で学習データの検証
    scores = cross_val_score(
        xgb_default, X_train, y_train,
        cv=5, scoring='accuracy'
    )
    print(f"[Default XGB] CV=5 accuracy -> {scores}, mean={scores.mean():.4f}")

    # fit → テストセットで評価
    xgb_default.fit(X_train, y_train)
    y_pred = xgb_default.predict(X_test)
    metrics_ = calculate_metrics(y_test, y_pred)

    # 特徴重要度の取得
    feature_importances = xgb_default.feature_importances_

    # コンソールに混同行列を表示
    print("Confusion Matrix:")
    print(np.array(metrics_["ConfusionMatrix"]))
    return metrics_, feature_importances

def train_and_evaluate_xgb_grid(X_train, y_train, X_test, y_test, feature_names):
    """
    XGBoost (グリッドサーチ, cv=5) でパラメータチューニングし、テストデータで評価
    """
    xgb_base = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    gs = GridSearchCV(
        xgb_base,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    print(f"[GridSearch XGB] Best Params: {gs.best_params_}")
    y_pred = best_model.predict(X_test)
    metrics_ = calculate_metrics(y_test, y_pred)

    # 特徴重要度の取得
    feature_importances = best_model.feature_importances_

    # コンソールに混同行列を表示
    print("Confusion Matrix:")
    print(np.array(metrics_["ConfusionMatrix"]))
    return metrics_, feature_importances

def plot_feature_importances(model_type, feature_set, features, importances, output_dir):
    """
    特徴重要度を可視化し、棒グラフとして保存する。
    """
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # データフレームに変換してソート
    fi_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
    plt.title(f'Feature Importances - {model_type} - {feature_set}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')

    # 出力ファイル名の作成
    # filename = f'feature_importances_{model_type}_{feature_set}.png'.replace('/', '_')
    # filepath = os.path.join(output_dir, filename)

    plt.tight_layout()
    # plt.savefig(filepath)
    plt.show()
    plt.close()
    # print(f"[Saved] Feature importances plot -> {filepath}")

def main():
    """
    【実行例】
      # 1ファイルのみ
      python3 train_model_xgb.py dataset1.csv --drop_cols colA colB --output_csv metrics_xgb.csv

      # 2ファイル (学習: dataset1, テスト: dataset2)
      python3 train_model_xgb.py dataset1.csv dataset2.csv \
        --drop_cols1 cA cB --drop_cols2 cX cY --output_csv metrics_xgb.csv

    特徴量セット: (1) zncc_50, (2) zncc_90, (3) centroid_50, (4) centroid_90, (5) all
      いずれも clusterがラベル(0/1)
      デフォルトXGB & グリッドサーチXGB の2種類を評価
    """
    parser = argparse.ArgumentParser(description="XGBoost with various feature subsets, default & grid, for cluster=0/1 classification.")
    parser.add_argument("csv_files", nargs="+",
                        help="1 or 2 input CSVs.")
    parser.add_argument("--drop_cols", nargs="*", default=[],
                        help="Columns to drop if only 1 file is provided.")
    parser.add_argument("--drop_cols1", nargs="*", default=[],
                        help="Columns to drop from the first dataset if 2 files.")
    parser.add_argument("--drop_cols2", nargs="*", default=[],
                        help="Columns to drop from the second dataset if 2 files.")
    parser.add_argument("--output_csv", default="result_metrics_xgb.csv",
                        help="Output CSV for model performance.")
    args = parser.parse_args()

    input_csvs = args.csv_files

    # 特徴量セット定義
    feature_sets_dict = {
        "zncc_50": ["zncc_50_dx","zncc_50_dy","zncc_50_dz"],
        "zncc_90": ["zncc_90_dx","zncc_90_dy","zncc_90_dz"],
        "centroid_50": ["centroid_50"],
        "centroid_90": ["centroid_90"],
        "all": [
            "zncc_50_dx","zncc_50_dy","zncc_50_dz",
            "zncc_90_dx","zncc_90_dy","zncc_90_dz",
            "centroid_50","centroid_90"
        ]
    }

    # 結果保存用リスト
    result_rows = []
    feature_importance_rows = []

    # 特徴重要度プロット用ディレクトリ
    fi_plot_dir = "feature_importances_plots"
    os.makedirs(fi_plot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1ファイル or 2ファイルで処理分岐
    # ------------------------------------------------------------------
    if len(input_csvs) == 1:
        # ==============
        #   1ファイル
        # ==============
        csv_path = input_csvs[0]
        if not os.path.exists(csv_path):
            print(f"Error: file '{csv_path}' not found.")
            return

        df = pd.read_csv(csv_path)
        print(f"Loaded single dataset shape={df.shape}")

        # 不要列削除
        df = drop_unwanted_columns(df, args.drop_cols)
        print(f"After dropping columns: shape={df.shape}")

        # データ拡張
        df_aug = create_augmented_dataset(df)
        print(f"Augmented dataset shape={df_aug.shape}")

        if 'cluster' not in df_aug.columns:
            print("Error: 'cluster' column not found after augmentation.")
            return

        # 特徴量パターンごとに学習/評価
        for feat_name, feat_cols in feature_sets_dict.items():
            print(f"\n--- FeatureSet: {feat_name} ---")
            X_all = df_aug[feat_cols].values
            y_all = df_aug['cluster'].values

            # train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, shuffle=True
            )

            # 特徴名リスト
            feature_names = feat_cols

            # 1) XGBoost(デフォルト)
            print("[Default XGB]")
            met_def, fi_def = train_and_evaluate_xgb_default(X_train, y_train, X_test, y_test, feature_names)
            result_rows.append({
                "ModelType": "XGB_Default",
                "FeatureSet": feat_name,
                **met_def
            })
            # 特徴重要度を保存
            for fname, fi in zip(feature_names, fi_def):
                feature_importance_rows.append({
                    "ModelType": "XGB_Default",
                    "FeatureSet": feat_name,
                    "Feature": fname,
                    "Importance": fi
                })
            # 特徴重要度を可視化
            plot_feature_importances("XGB_Default", feat_name, feature_names, fi_def, fi_plot_dir)

            # 2) XGBoost(グリッドサーチ)
            print("[GridSearch XGB]")
            met_gs, fi_gs = train_and_evaluate_xgb_grid(X_train, y_train, X_test, y_test, feature_names)
            result_rows.append({
                "ModelType": "XGB_Grid",
                "FeatureSet": feat_name,
                **met_gs
            })
            # 特徴重要度を保存
            for fname, fi in zip(feature_names, fi_gs):
                feature_importance_rows.append({
                    "ModelType": "XGB_Grid",
                    "FeatureSet": feat_name,
                    "Feature": fname,
                    "Importance": fi
                })
            # 特徴重要度を可視化
            plot_feature_importances("XGB_Grid", feat_name, feature_names, fi_gs, fi_plot_dir)

    elif len(input_csvs) == 2:
        # ==============
        #   2ファイル
        # ==============
        csv_path1, csv_path2 = input_csvs
        if not os.path.exists(csv_path1) or not os.path.exists(csv_path2):
            print("Error: one of the 2 CSVs not found.")
            return

        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
        print(f"Dataset1 shape={df1.shape}, Dataset2 shape={df2.shape}")

        # 個別に不要列削除
        df1 = drop_unwanted_columns(df1, args.drop_cols1)
        df2 = drop_unwanted_columns(df2, args.drop_cols2)
        print(f"After dropping: df1={df1.shape}, df2={df2.shape}")

        # 拡張
        df1_aug = create_augmented_dataset(df1)
        df2_aug = create_augmented_dataset(df2)
        print(f"Aug df1={df1_aug.shape}, Aug df2={df2_aug.shape}")

        if 'cluster' not in df1_aug.columns or 'cluster' not in df2_aug.columns:
            print("Error: 'cluster' column not found in one of the augmented datasets.")
            return

        # df1_aug → 学習, df2_aug → テスト
        X_train_all, y_train_all = df1_aug.drop(columns=['cluster']), df1_aug['cluster']
        X_test_all, y_test_all = df2_aug.drop(columns=['cluster']), df2_aug['cluster']

        # 特徴量パターンごとに学習/評価
        for feat_name, feat_cols in feature_sets_dict.items():
            print(f"\n--- FeatureSet: {feat_name} ---")

            # 学習データ
            X_train = X_train_all[feat_cols].values
            y_train = y_train_all.values
            # テストデータ
            X_test = X_test_all[feat_cols].values
            y_test = y_test_all.values

            feature_names = feat_cols

            # 1) デフォルトXGB
            print("[Default XGB]")
            met_def, fi_def = train_and_evaluate_xgb_default(X_train, y_train, X_test, y_test, feature_names)
            result_rows.append({
                "ModelType": "XGB_Default",
                "FeatureSet": feat_name,
                **met_def
            })
            # 特徴重要度を保存
            for fname, fi in zip(feature_names, fi_def):
                feature_importance_rows.append({
                    "ModelType": "XGB_Default",
                    "FeatureSet": feat_name,
                    "Feature": fname,
                    "Importance": fi
                })
            # 特徴重要度を可視化
            plot_feature_importances("XGB_Default", feat_name, feature_names, fi_def, fi_plot_dir)

            # 2) グリッドサーチXGB
            print("[GridSearch XGB]")
            met_gs, fi_gs = train_and_evaluate_xgb_grid(X_train, y_train, X_test, y_test, feature_names)
            result_rows.append({
                "ModelType": "XGB_Grid",
                "FeatureSet": feat_name,
                **met_gs
            })
            # 特徴重要度を保存
            for fname, fi in zip(feature_names, fi_gs):
                feature_importance_rows.append({
                    "ModelType": "XGB_Grid",
                    "FeatureSet": feat_name,
                    "Feature": fname,
                    "Importance": fi
                })
            # 特徴重要度を可視化
            plot_feature_importances("XGB_Grid", feat_name, feature_names, fi_gs, fi_plot_dir)

    else:
        print("Error: must provide 1 or 2 CSV files only.")
        return

    # 結果をCSVに保存
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(args.output_csv, index=False)
    print(f"\n[Saved] metrics -> {args.output_csv}")

    # 特徴重要度をCSVに保存
    if feature_importance_rows:
        fi_df = pd.DataFrame(feature_importance_rows)
        fi_output_csv = "feature_importances_xgb.csv"
        fi_df.to_csv(fi_output_csv, index=False)
        print(f"[Saved] feature importances -> {fi_output_csv}")

    print(f"[Saved] Feature importances plots are saved in '{fi_plot_dir}' directory.")

if __name__ == "__main__":
    main()
