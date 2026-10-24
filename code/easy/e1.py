# AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전 · 초보용 코드 e1.py
# 과제 문항 1 「이상치를 지우지 않고 5-Fold로 성능 재기」
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼
N_SPLITS = 5        # [다이얼] fold 수 (과제는 5)
GB_DEPTH = 3        # [다이얼] 그래디언트 부스팅 나무 깊이 (최종 모델은 3)

# ② 데이터 읽기 · 같은 행 1건 빼기 (charges 는 자르지 않는다)
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv")
df = df.drop_duplicates().reset_index(drop=True)
X = pd.get_dummies(df.drop(columns="charges"), drop_first=True).astype(float)
y = df["charges"]
print("rows:", len(df), " features:", list(X.columns))

# ③ charges 10분위로 층화한 5-Fold (섞기, random_state=42)
band = pd.qcut(y, 10, labels=False)
folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(X, band))
score = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}

# ④ 선형 회귀: 원래 charges
lin = make_pipeline(StandardScaler(), LinearRegression())
r = cross_validate(lin, X, y, cv=folds, scoring=score)
print("Linear raw   MAE per fold:", np.round(-r["test_MAE"]), " mean", round(-r["test_MAE"].mean()), " std", round(r["test_MAE"].std(ddof=1)))
print("Linear raw   R2  per fold:", np.round(r["test_R2"], 4), " mean", round(r["test_R2"].mean(), 4))

# ⑤ 선형 회귀: log(charges) 로 학습하고 exp 로 되돌려 달러로 평가
lin_log = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), LinearRegression()), func=np.log, inverse_func=np.exp)
r = cross_validate(lin_log, X, y, cv=folds, scoring=score)
print("Linear log   MAE per fold:", np.round(-r["test_MAE"]), " mean", round(-r["test_MAE"].mean()), " std", round(r["test_MAE"].std(ddof=1)))
print("Linear log   R2 (dollar) per fold:", np.round(r["test_R2"], 4), " mean", round(r["test_R2"].mean(), 4))

# ⑥ 같은 선형 모델을 로그 척도 그대로 채점하면 (log y 를 정답으로)
r = cross_validate(make_pipeline(StandardScaler(), LinearRegression()), X, np.log(y), cv=folds, scoring="r2")
print("Linear log   R2 (log scale) mean:", round(r["test_score"].mean(), 4), " <- 달러로 되돌린 R2 와 비교해 보세요")

# ⑦ 그래디언트 부스팅: 원래 charges
gb = GradientBoostingRegressor(n_estimators=200, max_depth=GB_DEPTH, learning_rate=0.05, random_state=42)
r = cross_validate(gb, X, y, cv=folds, scoring=score)
print("GBM raw      MAE per fold:", np.round(-r["test_MAE"]), " mean", round(-r["test_MAE"].mean()), " std", round(r["test_MAE"].std(ddof=1)))
print("GBM raw      R2  per fold:", np.round(r["test_R2"], 4), " mean", round(r["test_R2"].mean(), 4))

# ⑧ 기준선: 학습 fold 평균만 답하기 (DummyRegressor)
from sklearn.dummy import DummyRegressor
r = cross_validate(DummyRegressor(strategy="mean"), X, y, cv=folds, scoring=score)
print("Baseline     MAE per fold:", np.round(-r["test_MAE"]), " mean", round(-r["test_MAE"].mean()), " R2 mean", round(r["test_R2"].mean(), 4))
