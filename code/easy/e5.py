# AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전 · 초보용 코드 e5.py
# 5편 「꼬리가 긴 금액은 어디에나 있다」 (과제 문항 5)
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼
TOP_SHARE = 0.01        # [다이얼] 상위 몇 % 의 금액 비율을 볼까 (0.01 = 상위 1%)
GROUP_COLUMN = "smoker"  # [다이얼] 금액 비율을 볼 그룹 열: "smoker" 또는 "sex" 또는 "region"

# ② 데이터 읽기 · 같은 행 1건 빼기
import numpy as np
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv")
df = df.drop_duplicates().reset_index(drop=True)
y = df["charges"].sort_values(ascending=False).values
print("rows:", len(y), " mean:", round(y.mean()), " median:", round(np.median(y)), " mean/median:", round(y.mean() / np.median(y), 2))

# ③ 꼬리 지표: 상위 TOP_SHARE 가 전체 금액에서 차지하는 비율
k = max(1, int(len(y) * TOP_SHARE))
print("top", TOP_SHARE, "people:", k, " share of total charges:", round(y[:k].sum() / y.sum(), 4))
print("top 10% share:", round(y[:int(len(y) * 0.10)].sum() / y.sum(), 4))
print("bottom 50% share:", round(y[len(y) - len(y) // 2:].sum() / y.sum(), 4))

# ④ 그룹별 사람 비율과 금액 비율
table = pd.DataFrame({"people_share": df[GROUP_COLUMN].value_counts(normalize=True), "amount_share": df.groupby(GROUP_COLUMN)["charges"].sum() / df["charges"].sum()})
print(table.round(4))

# ⑤ 그림: 누적 금액 곡선 (사람을 금액 큰 순서로 세웠을 때)
import matplotlib.pyplot as plt
plt.plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(y) / y.sum())
plt.xlabel("share of people (largest first)")
plt.ylabel("share of total charges")
plt.title("How concentrated are medical charges?")
plt.show()
