# AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전 · 초보용 코드 e2.py
# 과제 문항 2 「흡연자만, 비흡연자만, 고액만: 그룹별 오차」
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼
GROUP_COLUMN = "smoker"   # [다이얼] 그룹을 나눌 열: "smoker" 또는 "sex" 또는 "region"
TOP_SHARE = 0.10          # [다이얼] 고액으로 볼 비율 (0.10 = 상위 10%)

# ② 데이터 읽기 · 같은 행 1건 빼기
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv")
df = df.drop_duplicates().reset_index(drop=True)
X = pd.get_dummies(df.drop(columns="charges"), drop_first=True).astype(float)
y = df["charges"]

# ③ 1편과 같은 5-Fold 로 "검증 예측"(모든 사람이 한 번씩 시험을 본 예측) 만들기
band = pd.qcut(y, 10, labels=False)
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, band))
df["pred_linear"] = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=folds)
df["pred_gbm"] = cross_val_predict(GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42), X, y, cv=folds)
df["err_linear"] = df["pred_linear"] - df["charges"]
df["err_gbm"] = df["pred_gbm"] - df["charges"]

# ④ 그룹별 MAE · 평균 잔차(편향 = 예측 − 실제, 음수면 적게 잡음) · MAPE
df["abs_linear"] = df["err_linear"].abs()
df["abs_gbm"] = df["err_gbm"].abs()
df["ape_linear"] = df["abs_linear"] / df["charges"]
table = df.groupby(GROUP_COLUMN)[["abs_linear", "abs_gbm", "err_linear", "err_gbm", "ape_linear"]].mean()
table.columns = ["MAE linear", "MAE gbm", "bias linear", "bias gbm", "MAPE linear"]
print(table.round(3))
print("people per group:", df[GROUP_COLUMN].value_counts().to_dict())

# ⑤ 고액 상위 고객만
top = df["charges"] >= df["charges"].quantile(1 - TOP_SHARE)
print("top people:", top.sum(), " smokers among them:", (df["smoker"][top] == "yes").sum())
print("top MAE linear:", round(df["abs_linear"][top].mean()), " bias linear:", round(df["err_linear"][top].mean()))
print("top MAE gbm   :", round(df["abs_gbm"][top].mean()), " bias gbm   :", round(df["err_gbm"][top].mean()))

# ⑥ 전체 MSE = 그룹 비율 x 그룹 MSE 의 합 (선형 모델로 확인)
share = df[GROUP_COLUMN].value_counts(normalize=True)
mse_group = (df["err_linear"] ** 2).groupby(df[GROUP_COLUMN]).mean()
print("MSE all          :", round((df["err_linear"] ** 2).mean()))
print("sum share x MSE  :", round((share * mse_group).sum()))
