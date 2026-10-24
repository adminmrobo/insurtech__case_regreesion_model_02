"""
AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 인슈어테크_04_회귀모델_실전 · 공통 코드 (common.py)
글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0

두 시리즈(1부 개념편 · 2부 문제편)가 똑같이 쓰는 파일입니다. 정리 규칙이 두 부에서 같아야 숫자를 비교할 수 있습니다.

하는 일
 ① 건강보험 의료비 데이터 읽기 (공개 학습용 데이터, 1,338행 × 7열)
    읽는 순서: 내 컴퓨터의 data/insurance.csv -> 이 저장소의 raw 주소 -> 원 출처 주소
 ② 정리 규칙: 완전히 같은 행 1건만 뺀다. 목표값(charges)은 자르거나 고치지 않는다.
 ③ 범주형 원-핫 인코딩(drop_first) -> 특성 8개
 ④ 1부의 학습/시험 나누기: charges 10분위로 층화한 80:20, random_state=42 (data/split.csv 와 같음)
 ⑤ 기준선: 모두에게 "학습 데이터 평균"만 답하기
 ⑥ (선택) 흡연 x 비만(BMI 30 이상) 상호작용 특성 만들기 - 2부 3·4편에서 씀

원칙: 데이터 증강(합성 행 · 행 복제 · 정답에 잡음 넣기)은 쓰지 않는다. 원본 행만 쓴다.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

LOCAL_PATHS = ["data/insurance.csv", "../data/insurance.csv", "insurance.csv"]
REPO_URL = "https://raw.githubusercontent.com/adminmrobo/ai-biz-insurtech-03/main/data/insurance.csv"
SOURCE_URL = "https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv"
SOURCE_MD5 = "d5364d06246fb4bfa4cc7d1ee89ebd0d"
SEED = 42
TARGET = "charges"
NUM_COLS = ["age", "bmi", "children"]
CAT_COLS = ["sex", "smoker", "region"]
BMI_OBESE = 30.0   # WHO 성인 비만 기준 (BMI 30 이상)


def load_raw(path=None):
    """① 원본 데이터 읽기. path 를 주지 않으면 로컬 -> 저장소 -> 원 출처 순서로 찾는다."""
    if path is None:
        path = os.environ.get("INS_DATA")
    if path:
        return pd.read_csv(path)
    for p in LOCAL_PATHS:
        if os.path.exists(p):
            return pd.read_csv(p)
    for url in (REPO_URL, SOURCE_URL):
        try:
            return pd.read_csv(url)
        except Exception:
            pass
    raise FileNotFoundError("insurance.csv 를 찾지 못했습니다.")


def clean(df, verbose=False):
    """② 정리 규칙. 바뀌는 행 수를 info 에 남긴다.

    - 완전히 같은 행(7개 열이 모두 같음) 1건 -> 뒤의 것을 뺀다
      (같은 사람이 학습과 시험에 동시에 들어가 점수가 부풀려지는 것을 막기 위해)
    - charges(목표값): 고액도 그대로 둔다. 자르거나(상한) 지우지 않는다.
    - bmi IQR 밖 9명(최대 53.13): 사람에게 가능한 값이라 그대로 둔다.
    - 결측: 없음
    """
    n0 = len(df)
    dup_mask = df.duplicated(keep="first")
    dup_rows = list(df.index[dup_mask])
    out = df.loc[~dup_mask].copy()
    q1, q3 = out[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    tgt_iqr = int(((out[TARGET] < q1 - 1.5 * iqr) | (out[TARGET] > q3 + 1.5 * iqr)).sum())
    b1, b3 = out["bmi"].quantile([0.25, 0.75])
    biqr = b3 - b1
    bmi_iqr = int(((out["bmi"] < b1 - 1.5 * biqr) | (out["bmi"] > b3 + 1.5 * biqr)).sum())
    info = {"rows_raw": n0, "dropped_duplicate_rows": dup_rows, "rows_clean": len(out),
            "missing": int(df.isna().sum().sum()),
            "charges_iqr_flagged_kept": tgt_iqr, "bmi_iqr_flagged_kept": bmi_iqr,
            "charges_changed": 0}
    if verbose:
        print("rule | rows changed")
        print("drop exact duplicate | %d (row %s)" % (len(dup_rows), dup_rows))
        print("charges IQR flagged, kept as is | %d" % tgt_iqr)
        print("bmi IQR flagged, kept as is | %d" % bmi_iqr)
        print("missing values | %d" % info["missing"])
    return out, info


def encode(df):
    """③ X(원-핫, drop_first) 와 y(charges) 만들기. 열 순서는 고정."""
    y = df[TARGET].astype(float)
    X = pd.get_dummies(df.drop(columns=TARGET), columns=CAT_COLS, drop_first=True).astype(float)
    cols = ["age", "bmi", "children", "sex_male", "smoker_yes",
            "region_northwest", "region_southeast", "region_southwest"]
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    return X[cols], y


def add_interactions(X):
    """⑥ 흡연 x 비만 상호작용. 흡연자이면서 BMI 30 이상인 사람의 의료비가 따로 뛰는 것을 직선 모델이 잡게 한다."""
    X = X.copy()
    X["bmi30"] = (X["bmi"] >= BMI_OBESE).astype(float)
    X["smoker_bmi30"] = X["smoker_yes"] * X["bmi30"]
    X["smoker_bmi"] = X["smoker_yes"] * X["bmi"]
    return X


def strata(y, q=10):
    """분위 층화용 꼬리표: 금액을 q 개 구간으로 나눈 번호."""
    return pd.qcut(y, q, labels=False, duplicates="drop")


def split(X, y):
    """④ 1부의 학습/시험 나누기 (charges 10분위 층화, 80:20, random_state=42)."""
    return train_test_split(X, y, test_size=0.2, stratify=strata(y), random_state=SEED)


def folds(y, n_splits=5):
    """2부의 5-Fold: charges 10분위로 층화한 StratifiedKFold(shuffle=True, random_state=42)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), strata(y)))


def baseline(y_train, n):
    """⑤ 기준선: 모두에게 학습 데이터 평균만 답한다."""
    return np.full(n, float(np.mean(y_train)))


def metrics(y, p):
    """주지표 + 보조 지표. MAE 를 주지표로, RMSE · R² · 고액(상위 10%) MAE · 과소 추정 비율을 함께."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = y - p
    top = y >= np.quantile(y, 0.9)
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2": 1 - ss_res / ss_tot, "MAE_top10": float(np.mean(np.abs(e[top]))),
            "under_rate": float(np.mean(p < y)), "bias": float(np.mean(p - y))}


# 최종 모델 설정: 1부 3편에서 "학습 데이터만" 5-Fold 교차검증(MAE)으로 고른 값
# (n_estimators 100/200/300 x max_depth 2/3/4 x learning_rate 0.05/0.1 중 1위)
FINAL_PARAMS = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "random_state": SEED}

MODEL_NAMES = ["Linear", "Ridge", "Decision Tree", "KNN", "Gradient Boosting"]
MODEL_KO = {"Linear": "선형 회귀", "Ridge": "릿지", "Decision Tree": "의사결정나무", "KNN": "KNN",
            "Gradient Boosting": "그래디언트 부스팅"}


def models():
    """두 부가 같이 쓰는 모델 다섯. 표준화가 필요한 모델은 Pipeline 안에서 학습 데이터에만 fit."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    return {
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=SEED),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        "Gradient Boosting": GradientBoostingRegressor(**FINAL_PARAMS),
    }


def final_model():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(**FINAL_PARAMS)


def load_clean_xy():
    """한 줄로: 읽기 -> 정리 -> X, y (원본 행 번호를 index 로 유지)."""
    df, info = clean(load_raw())
    X, y = encode(df)
    return df, X, y, info
