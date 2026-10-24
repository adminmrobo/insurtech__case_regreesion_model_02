# AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전 · 초보용 코드 e3.py
# 과제 문항 3 「그 숫자를 성능이라 부르면 생기는 일」 — 개선 두 가지를 직접 해 봅니다
#  (1) 흡연 x 비만 상호작용 특성 추가  (2) 로그 타깃의 스미어링 보정
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼
BMI_CUT = 30        # [다이얼] 비만으로 볼 BMI 기준 (WHO 성인 기준 30)
COST_UNDER = 3      # [다이얼] 적게 잡은 1달러의 비용 (많이 잡은 1달러 = 1)

# ② 데이터 읽기 · 같은 행 1건 빼기
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv")
df = df.drop_duplicates().reset_index(drop=True)
X = pd.get_dummies(df.drop(columns="charges"), drop_first=True).astype(float)
y = df["charges"]
band = pd.qcut(y, 10, labels=False)
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, band))
smoker = X["smoker_yes"] == 1
top = y >= y.quantile(0.9)

# ③ 개선 전: 선형 회귀 (특성 8개)
before = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=folds)

# ④ 개선 후: 흡연 x 비만 상호작용 특성 3개 추가
X2 = X.copy()
X2["bmi30"] = (X2["bmi"] >= BMI_CUT).astype(float)
X2["smoker_bmi30"] = X2["smoker_yes"] * X2["bmi30"]
X2["smoker_bmi"] = X2["smoker_yes"] * X2["bmi"]
after = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()), X2, y, cv=folds)

# ⑤ 전/후 비교: MAE · 흡연자 MAE · 고액 MAE · 고액 편향 · 3:1 비용
err_b = before - y
err_a = after - y
cost_b = (COST_UNDER * np.clip(-err_b, 0, None) + np.clip(err_b, 0, None)).mean()
cost_a = (COST_UNDER * np.clip(-err_a, 0, None) + np.clip(err_a, 0, None)).mean()
table = pd.DataFrame(index=["Linear", "Linear + interaction"])
table["MAE"] = [err_b.abs().mean(), err_a.abs().mean()]
table["MAE smoker"] = [err_b.abs()[smoker].mean(), err_a.abs()[smoker].mean()]
table["MAE top10"] = [err_b.abs()[top].mean(), err_a.abs()[top].mean()]
table["bias top10"] = [err_b[top].mean(), err_a[top].mean()]
table["cost 3:1"] = [cost_b, cost_a]
print(table.round(0))

# ⑥ 스미어링: fold 1 하나로 보기 (5 fold 전체는 정식 코드 f3.py)
tr, va = folds[0]
log_model = make_pipeline(StandardScaler(), LinearRegression()).fit(X.iloc[tr], np.log(y.iloc[tr]))
plain = np.exp(log_model.predict(X.iloc[va]))
smear = np.mean(np.exp(np.log(y.iloc[tr]) - log_model.predict(X.iloc[tr])))
print("smearing factor (fold 1):", round(smear, 4))
print("mean actual  :", round(y.iloc[va].mean()))
print("mean exp(pred)        :", round(plain.mean()), " <- 평균보다 낮게 (중앙값 쪽)")
print("mean exp(pred) x smear:", round((plain * smear).mean()))
print("MAE exp(pred)         :", round(np.abs(plain - y.iloc[va]).mean()))
print("MAE exp(pred) x smear :", round(np.abs(plain * smear - y.iloc[va]).mean()))
