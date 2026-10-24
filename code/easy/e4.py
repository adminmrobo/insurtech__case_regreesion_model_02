# AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전 · 초보용 코드 e4.py
# 4편 「그 모델, HTML에 그대로 넣어도 될까」 (과제 문항 4)
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼
FORGET_INTERACTION = 1   # [다이얼] 1 = 배포할 때 상호작용 규칙을 빼먹었다고 가정, 0 = 제대로 계산
CHANGE_LIMIT = 0.10      # [다이얼] 견적이 이 비율 이상 바뀌면 '바뀐 사람'으로 센다

# ② 데이터 읽기 · 같은 행 1건 빼기 · 원-핫 인코딩 (열 순서를 고정)
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv")
df = df.drop_duplicates().reset_index(drop=True)
X = pd.get_dummies(df.drop(columns="charges"), columns=["sex", "smoker", "region"], drop_first=True).astype(float)
X = X[["age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest"]]
y = df["charges"]

# ③ 상호작용 특성 (흡연 x 비만) 을 붙이고, 표준화 + 선형 회귀를 전체 1,337행으로 학습
X["bmi30"] = (X["bmi"] >= 30).astype(float)
X["smoker_bmi30"] = X["smoker_yes"] * X["bmi30"]
X["smoker_bmi"] = X["smoker_yes"] * X["bmi"]
model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)

# ④ JSON 으로 내보내기: 열 순서 · 평균 · 표준편차 · 계수 · 절편
export = {"columns": list(X.columns), "mean": model[0].mean_.tolist(), "scale": model[0].scale_.tolist(), "coef": model[1].coef_.tolist(), "intercept": float(model[1].intercept_)}
text = json.dumps(export)
print("JSON bytes:", len(text))

# ⑤ JSON 만 보고 다시 계산 (브라우저 JS 가 하는 계산과 같음) -> sklearn 과 비교
J = json.loads(text)
X_used = X.copy()
X_used[["bmi30", "smoker_bmi30", "smoker_bmi"]] = X_used[["bmi30", "smoker_bmi30", "smoker_bmi"]] * (1 - FORGET_INTERACTION)
json_pred = ((X_used[J["columns"]].values - np.array(J["mean"])) / np.array(J["scale"])) @ np.array(J["coef"]) + J["intercept"]
sk_pred = model.predict(X)
print("max |JSON - sklearn|:", np.abs(json_pred - sk_pred).max())
changed = (np.abs(json_pred - sk_pred) / sk_pred) >= CHANGE_LIMIT
print("quotes changed by", CHANGE_LIMIT, "or more:", int(changed.sum()), "of", len(y))
print("mean change ($):", round((json_pred - sk_pred).mean()))

# ⑥ 로그 타깃 모델의 스미어링 계수 (특성 8개, 상호작용 없이)
X8 = X[["age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest"]]
log_model = make_pipeline(StandardScaler(), LinearRegression()).fit(X8, np.log(y))
smear = np.exp(np.log(y) - log_model.predict(X8)).mean()
print("smearing factor:", round(smear, 4))
print("mean exp(pred):", round(np.exp(log_model.predict(X8)).mean()), " actual mean:", round(y.mean()))
