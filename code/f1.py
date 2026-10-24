"""
AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전 · 정식 코드 f1.py
과제 문항 1 「이상치를 지우지 않고 5-Fold로 성능 재기」
무엇을, 어떤 순서로:
 ① 데이터를 읽고 정리한다(common.py): 완전 중복 1건만 빼고 1,337행. charges 는 자르지 않는다
 ② 나누는 방법 셋을 비교한다: KFold(섞지 않음) · KFold(섞음) · charges 10분위 층화 StratifiedKFold
 ③ 층화 5-Fold 로 모델 다섯 개를 원래 charges / log(charges) 두 타깃으로 학습한다
    (로그 타깃은 np.exp 로 달러로 되돌려 평가. 표준화는 Pipeline 안에서 학습 fold 에만 fit)
 ④ fold별 MAE · RMSE · R² 5개 + 평균 ± 표준편차(ddof=1), 기준선(학습 fold 평균)과 한 표로
 ⑤ 로그 척도 R² 와 달러 척도 R² 를 나란히 본다
 ⑥ 1부(시험 268명 한 번)의 숫자와 나란히 놓는다
 ⑦ 전체 1,337명의 검증(OOF) 예측을 data/oof_predictions.csv 로 저장한다 (2 · 3편이 씀)
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
import common as C

METRICS = ["MAE", "RMSE", "R2"]


def r2(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def split_compare(df, y):
    """② 나누는 방법별 fold 의 검증 데이터 charges 평균 · 흡연자 수."""
    ways = {
        "KFold (no shuffle)": list(KFold(5).split(df)),
        "KFold (shuffle)": list(KFold(5, shuffle=True, random_state=C.SEED).split(df)),
        "Stratified by charges decile": C.folds(y),
    }
    rows = []
    smk = (df["smoker"] == "yes").values
    for name, fl in ways.items():
        for k, (tr, va) in enumerate(fl, 1):
            rows.append(dict(split=name, fold=k, n=len(va), mean=float(y.values[va].mean()),
                             top10=int((y.values[va] >= np.quantile(y, 0.9)).sum()), smokers=int(smk[va].sum())))
    return pd.DataFrame(rows)


def cv_all(X, y):
    """③ ④ 층화 5-Fold, 모델 다섯 x 타깃 둘 + 기준선."""
    fl = C.folds(y)
    models = C.models()
    rows = []
    oof = pd.DataFrame({"row": X.index, "y": y.values})
    oof["fold"] = 0
    for name in ["Baseline"] + C.MODEL_NAMES:
        for tgt in (["raw"] if name == "Baseline" else ["raw", "log"]):
            col = "%s|%s" % (name, tgt)
            oof[col] = np.nan
    for k, (tr, va) in enumerate(fl, 1):
        Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        oof.loc[oof.index[va], "fold"] = k
        p = C.baseline(ytr, len(va))
        oof.loc[oof.index[va], "Baseline|raw"] = p
        rows.append(dict(model="Baseline", target="raw", fold=k, MAE=np.mean(np.abs(yva - p)),
                         RMSE=np.sqrt(np.mean((yva - p) ** 2)), R2=r2(yva, p), R2_log=np.nan))
        for name in C.MODEL_NAMES:
            m = clone(models[name]).fit(Xtr, ytr)
            p = m.predict(Xva)
            oof.loc[oof.index[va], "%s|raw" % name] = p
            rows.append(dict(model=name, target="raw", fold=k, MAE=np.mean(np.abs(yva - p)),
                             RMSE=np.sqrt(np.mean((yva - p) ** 2)), R2=r2(yva, p), R2_log=np.nan))
            ml = clone(models[name]).fit(Xtr, np.log(ytr))
            pl = ml.predict(Xva)
            p = np.exp(pl)
            oof.loc[oof.index[va], "%s|log" % name] = p
            rows.append(dict(model=name, target="log", fold=k, MAE=np.mean(np.abs(yva - p)),
                             RMSE=np.sqrt(np.mean((yva - p) ** 2)), R2=r2(yva, p), R2_log=r2(np.log(yva), pl)))
    return pd.DataFrame(rows), oof


def summary(per):
    g = per.groupby(["model", "target"], sort=False)
    s = g[METRICS + ["R2_log"]].agg(["mean", "std"])
    return s


def part1_test():
    """⑥ 1부와 같은 나누기(시험 268명 한 번)."""
    df, X, y, info = C.load_clean_xy()
    Xtr, Xte, ytr, yte = C.split(X, y)
    rows = [dict(model="Baseline", **C.metrics(yte, C.baseline(ytr, len(yte))))]
    for name, m in C.models().items():
        rows.append(dict(model=name, **C.metrics(yte, m.fit(Xtr, ytr).predict(Xte))))
    return pd.DataFrame(rows).set_index("model")


def run(verbose=True, save=True):
    df, X, y, info = C.load_clean_xy()
    sp = split_compare(df, y)
    per, oof = cv_all(X, y)
    s = summary(per)
    p1 = part1_test()
    if save:
        oof.to_csv("data/oof_predictions.csv", index=False, float_format="%.4f")
    if verbose:
        print("rows=%d  dropped duplicate rows=%s  charges IQR flagged (kept)=%d" % (
            info["rows_clean"], info["dropped_duplicate_rows"], info["charges_iqr_flagged_kept"]))
        print("\n[split compare] validation-fold charges mean / smokers / top10 count")
        for name, g in sp.groupby("split", sort=False):
            print("%-30s mean: %s  smokers: %s  top10: %s" % (name, [round(v) for v in g["mean"]], list(g["smokers"]), list(g["top10"])))
        for met in METRICS:
            t = per.pivot_table(index=["model", "target"], columns="fold", values=met, sort=False)
            t["mean"] = t.mean(axis=1)
            t["std"] = per.groupby(["model", "target"], sort=False)[met].std(ddof=1)
            print("\n[%s] fold 1~5, mean, std (dollar scale)" % met)
            print(t.round(4 if met == "R2" else 0).to_string())
        print("\n[R2 log scale vs dollar scale] (log-target models, mean of 5 folds)")
        lg = s.xs("log", level="target")
        print(pd.DataFrame({"R2_log_scale": lg[("R2_log", "mean")], "R2_dollar": lg[("R2", "mean")]}).round(4).to_string())
        print("\n[part 1 test (268 people, once)]")
        print(p1[["MAE", "RMSE", "R2"]].round({"MAE": 0, "RMSE": 0, "R2": 4}).to_string())
        print("\noof saved: data/oof_predictions.csv  rows=%d" % len(oof))
    return dict(info=info, split=sp, per=per, summary=s, oof=oof, part1=p1, df=df, X=X, y=y)


if __name__ == "__main__":
    run()
