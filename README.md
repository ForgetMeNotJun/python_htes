
# 治療効果の異質性の識別：Causal Forestを用いた分析

このレポジトリは、Causal Forestを使用して治療効果の異質性（Heterogenous Treatment Effects, HTEs）を検証するサンプルです。`pandas`、`scikit-learn`、`causalml`、`matplotlib`などのライブラリを使用してPythonで書かれています。

## 必要なライブラリのインストール

以下のコマンドを使用して必要なライブラリをインストールしてください：

```bash
pip install pandas scikit-learn causalml matplotlib openpyxl
```

## 環境のセットアップ

以下のコードを使用してライブラリをインポートし、環境をセットアップします：

```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from causalml.inference.tree import CausalForest
import warnings 
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 400)
```

## データの読み込み

データを読み込むために、以下のコードを使用します：

```python
DATA_PATH = "../../data/"
FILE_NAME = "FILE.csv"

df = pd.read_csv(DATA_PATH + FILE_NAME)
df.head()
```

## 変数の定義

処置変数、結果変数、共変量リストを定義します：

```python
COVARIATES_LIST = ["COVARIATES_LIST"]
OUTCOME = "OUTCOME"
TREATMENT = "TREATMENT"

X = df[COVARIATES_LIST]
Y = df[OUTCOME].values
W = df[TREATMENT].values
```

## クロスバリデーションのセットアップ

クロスバリデーションの設定を行います：

```python
num_rankings = 5
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)
```

## シミュレーションとモデルフィッティング

シミュレーションを行い、モデルをフィットさせます：

```python
n = 1000
results = []

for i in range(n):
    np.random.seed(i)
    cf = CausalForest(n_estimators=5000, random_state=i)
    cf.fit(X.values, W, Y)

    tau_hat = cf.predict(X.values).flatten()
    e_hat = cf.propensity
    m_hat = cf.marginal_outcome

    rankings = np.full(X.shape[0], np.nan)
    for train_index, test_index in kf.split(X):
        tau_hat_quantiles = np.quantile(tau_hat[test_index], np.linspace(0, 1, num_rankings + 1))
        rankings[test_index] = np.digitize(tau_hat[test_index], tau_hat_quantiles) - 1

    mu_hat_0 = m_hat - e_hat * tau_hat
    mu_hat_1 = m_hat + (1 - e_hat) * tau_hat
    aipw_scores = tau_hat + W / e_hat * (Y - mu_hat_1) - (1 - W) / (1 - e_hat) * (Y - mu_hat_0)
```

## AIPWスコアのプロット

推定された処置効果をプロットします：

```python
plt.figure(figsize=(10, 6))
plt.scatter(range(len(tau_hat)), tau_hat, alpha=0.5)
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel("Individual ranking of treatment effects")
plt.ylabel("Estimated individual treatment effects on outcome")
plt.title("Estimated Treatment Effects")
plt.show()
```

## 平均処置効果（ATE）のプロット

各クインタイルの平均処置効果をプロットします：

```python
forest_ate = []
for q in range(num_rankings):
    mask = (rankings == q)
    ate = np.mean(aipw_scores[mask])
    stderr = np.std(aipw_scores[mask]) / np.sqrt(mask.sum())
    forest_ate.append((f"Quintile{q + 1}", ate, stderr))
forest_ate_df = pd.DataFrame(forest_ate, columns=["Quintile", "Estimate", "StdErr"])

plt.figure(figsize=(10, 6))
plt.errorbar(forest_ate_df["Quintile"], forest_ate_df["Estimate"], yerr=1.96 * forest_ate_df["StdErr"], fmt='o', color='black', capsize=5)
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel("Quintile")
plt.ylabel("The treatment effect on outcome")
plt.title("Average CATE within each ranking (as defined by predicted CATE)")
plt.show()
```

## ATTとATEの計算

ATT（処置された対象の平均処置効果）とATE（全体の平均処置効果）を計算して出力します：

```python
att_results = []
ate_results = []
for q in range(num_rankings):
    mask = (rankings == q)
    att = cf.estimate_ate(X.values[mask], W[mask], Y[mask], target="treated")
    ate = cf.estimate_ate(X.values[mask], W[mask], Y[mask])
    att_results.append(att)
    ate_results.append(ate)

print("ATT Results:", att_results)
print("ATE Results:", ate_results)
```

以上の手順で、Causal Forestを使用して異質な処置効果を識別することができます。

## サンプルデータセットの説明

このノートブックでは、異質な処置効果（HTE）を識別するためにサンプルデータセットを使用します。以下のリンクからサンプルデータセットをダウンロードできます。

[サンプルデータセットをダウンロード](sandbox:/mnt/data/sample_data.csv)

### サンプルデータセットの概要

サンプルデータセットには1000件のサンプルが含まれており、以下の変数が含まれています：

- covariate_1, covariate_2, covariate_3, covariate_4, covariate_5: 共変量
- treatment: 処置変数（0または1）
- outcome: 結果変数

### データの生成方法

このデータは、以下の方法で生成されました：

1. 共変量（covariate_1 から covariate_5）を標準正規分布に従って生成。
2. 処置変数（treatment）を0.5の確率で二項分布に従って生成。
3. 結果変数（outcome）は、処置変数に応じて異なる正規分布から生成。

### ノートブックでの使用方法

サンプルデータセットを使用する場合、以下のコードを実行してデータを読み込み、必要な変数を設定します：

```python
import pandas as pd

# データの読み込み
df = pd.read_csv("PATH_TO_YOUR_DOWNLOADED_SAMPLE_DATA/sample_data.csv")

# 変数の定義
COVARIATES_LIST = ["covariate_1", "covariate_2", "covariate_3", "covariate_4", "covariate_5"]
OUTCOME = "outcome"
TREATMENT = "treatment"

X = df[COVARIATES_LIST]
Y = df[OUTCOME].values
W = df[TREATMENT].values