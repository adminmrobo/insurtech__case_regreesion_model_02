"""
AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전 · 정식 코드 f5.py
5편 「꼬리가 긴 금액은 어디에나 있다」 (과제 문항 5)
무엇을, 어떤 순서로:
 ① 우리 데이터(1,337행)의 꼬리 지표: 상위 1% · 10% 가 전체 금액에서 차지하는 비율, 하위 50% 비율,
    흡연자(20.5%)가 차지하는 금액 비율, 평균 / 중앙값, 왜도
 ② 같은 지표를 공개 데이터 두 개에서 직접 계산 (pyreadr 가 있으면)
    - 프랑스 자동차보험 대인배상 손해액 freMTPL2sev (CASdatasets)
    - 덴마크 화재 대형 손해 danish (evir 패키지, Copenhagen Re)
 ③ 결과를 사례 카드(cases.json)의 tail_share 칸과 비교한다
"""
import io
import urllib.request
import numpy as np
import pandas as pd
import common as C

EXTERNAL = {
    "freMTPL2sev": "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/freMTPL2sev.rda",
    "danish": "https://raw.githubusercontent.com/cran/evir/master/data/danish.rda",
}


def tail(x):
    x = np.sort(np.asarray(x, float))[::-1]
    n, s = len(x), x.sum()
    k1, k10, k50 = max(1, int(n * 0.01)), max(1, int(n * 0.10)), n // 2
    return {"n": n, "mean": float(x.mean()), "median": float(np.median(x)), "max": float(x[0]),
            "top1": float(x[:k1].sum() / s), "top10": float(x[:k10].sum() / s), "bottom50": float(x[n - k50:].sum() / s),
            "mean_over_median": float(x.mean() / np.median(x)), "skew": float(pd.Series(x).skew()),
            # 집중 곡선: 금액 큰 사람부터 세웠을 때, 상위 p% 가 차지하는 금액 비율 (p = 0, 1, …, 100%)
            "curve": [0.0] + [float(x[:max(1, int(round(n * p / 100)))].sum() / s) for p in range(1, 101)]}


def load_external(name):
    try:
        import pyreadr
        raw = urllib.request.urlopen(EXTERNAL[name], timeout=30).read()
        path = "/tmp/_%s.rda" % name
        open(path, "wb").write(raw)
        r = pyreadr.read_r(path)
        df = list(r.values())[0]
        return df.select_dtypes("number").iloc[:, -1].values
    except Exception as e:
        print("  (%s 를 읽지 못했습니다: %s)" % (name, type(e).__name__))
        return None


def run(verbose=True):
    df, X, y, info = C.load_clean_xy()
    ours = tail(y)
    smk = df["smoker"] == "yes"
    ours["smoker_people"] = float(smk.mean())
    ours["smoker_amount"] = float(y[smk].sum() / y.sum())
    ext = {}
    for name in EXTERNAL:
        v = load_external(name)
        if v is not None:
            ext[name] = tail(v)
    if verbose:
        print("[ours] n=%d mean=%.0f median=%.0f max=%.0f mean/median=%.2f skew=%.2f" % (
            ours["n"], ours["mean"], ours["median"], ours["max"], ours["mean_over_median"], ours["skew"]))
        print("  top 1%% share=%.4f  top 10%% share=%.4f  bottom 50%% share=%.4f" % (ours["top1"], ours["top10"], ours["bottom50"]))
        print("  smokers: %.1f%% of people, %.1f%% of total charges" % (ours["smoker_people"] * 100, ours["smoker_amount"] * 100))
        for k, t in ext.items():
            print("[%s] n=%d mean=%.2f median=%.2f max=%.2f mean/median=%.2f" % (k, t["n"], t["mean"], t["median"], t["max"], t["mean_over_median"]))
            print("  top 1%% share=%.4f  top 10%% share=%.4f  bottom 50%% share=%.4f" % (t["top1"], t["top10"], t["bottom50"]))
    return {"ours": ours, "ext": ext}


if __name__ == "__main__":
    run()
