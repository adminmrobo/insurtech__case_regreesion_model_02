"""
AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전 · 정식 코드 f3.py
과제 문항 3 「그 숫자를 성능이라 부르면 생기는 일」: 문제점 5개와 1:1 개선
무엇을, 어떤 순서로 (모두 1편과 같은 10분위 층화 5-Fold):
 ① 작은 그룹 -> 부트스트랩 95% 구간(2,000번) + fold 범위
 ② MAE 는 방향을 숨긴다 -> 편향(예측 − 실제) · 과소 추정 비율 · 비용(과소 1달러 : 과대 1달러 = 3 : 1)
 ③ 선형 모델은 흡연 x 비만을 못 잡는다 -> common.add_interactions 로 특성 3개 추가, 전/후 비교
 ④ 로그 타깃을 exp 로 되돌리면 평균이 아니라 중앙값 쪽 -> 스미어링 보정(Duan, 1983), 전/후 비교
 ⑤ 같은 fold 로 하이퍼파라미터를 고르고 같은 fold 로 채점하면 낙관 -> 중첩 교차검증과 비교
 ⑥ 4편이 쓸 수 있게 "상호작용 선형 모델"을 전체 1,337행으로 학습해 data/linear_interaction.json 으로 저장
"""
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import common as C

COST_UNDER, COST_OVER = 3.0, 1.0


def lin():
    return make_pipeline(StandardScaler(), LinearRegression())


def summ(y, p, smk, top):
    e = p - y
    under = np.clip(y - p, 0, None)
    over = np.clip(p - y, 0, None)
    return dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)), R2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
                MAE_smoker=np.mean(np.abs(e[smk])), MAE_top10=np.mean(np.abs(e[top])), bias=np.mean(e),
                bias_top10=np.mean(e[top]), under_rate=np.mean(p < y),
                cost_3to1=np.mean(COST_UNDER * under + COST_OVER * over), mean_pred=np.mean(p), mean_true=np.mean(y))


def oof(model, X, y, fl, log=False, smear=False):
    p = np.zeros(len(y))
    for tr, va in fl:
        if log:
            m = clone(model).fit(X.iloc[tr], np.log(y.iloc[tr]))
            z = m.predict(X.iloc[va])
            f = np.mean(np.exp(np.log(y.iloc[tr]) - m.predict(X.iloc[tr]))) if smear else 1.0
            p[va] = np.exp(z) * f
        else:
            p[va] = clone(model).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    return p


def boot_ci(y, p, mask, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    vals = [np.mean(np.abs(p[s] - y[s])) for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


GRID = list(itertools.product([100, 200, 300], [2, 3, 4], [0.05, 0.1]))


def gbm(ne, md, lr):
    return GradientBoostingRegressor(n_estimators=ne, max_depth=md, learning_rate=lr, random_state=C.SEED)


def nested(X, y, fl):
    """⑤ 같은 fold 로 고르고 채점(낙관) vs 안쪽 5-Fold 로 고르고 바깥 fold 로 채점(중첩)."""
    same = {}
    for g in GRID:
        same[g] = np.mean([np.mean(np.abs(gbm(*g).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va]) - y.iloc[va])) for tr, va in fl])
    best = min(same, key=same.get)
    outer = []
    picks = []
    for tr, va in fl:
        Xi, yi = X.iloc[tr], y.iloc[tr]
        inner = list(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(Xi, C.strata(yi)))
        sc = {g: np.mean([np.mean(np.abs(gbm(*g).fit(Xi.iloc[a], yi.iloc[a]).predict(Xi.iloc[b]) - yi.iloc[b])) for a, b in inner]) for g in GRID}
        g = min(sc, key=sc.get)
        picks.append(g)
        outer.append(np.mean(np.abs(gbm(*g).fit(Xi, yi).predict(X.iloc[va]) - y.iloc[va])))
    return dict(same_best=best, same_mae=same[best], nested_mae=float(np.mean(outer)), nested_std=float(np.std(outer, ddof=1)),
                nested_fold_mae=[float(v) for v in outer], nested_picks=picks, grid_size=len(GRID))


def export_linear_interaction(X, y, path="data/linear_interaction.json"):
    """⑥ 4편용: 전체 데이터로 학습한 상호작용 선형 모델의 계산 재료."""
    Xi = C.add_interactions(X)
    m = lin().fit(Xi, y)
    sc, lr = m.named_steps["standardscaler"], m.named_steps["linearregression"]
    d = dict(columns=list(Xi.columns), mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_.tolist(),
             intercept=float(lr.intercept_), bmi_obese=C.BMI_OBESE,
             note="f3.py: LinearRegression on add_interactions(X), all 1,337 rows, StandardScaler inside Pipeline")
    with open(path, "w") as f:
        json.dump(d, f, indent=1)
    return d


def run(verbose=True, do_nested=True, save=True):
    df, X, y, info = C.load_clean_xy()
    fl = C.folds(y)
    yv = y.values
    smk = (X["smoker_yes"] == 1).values
    top = yv >= np.quantile(yv, 0.9)
    Xi = C.add_interactions(X)
    P = {
        "Linear": oof(lin(), X, y, fl),
        "Linear + interaction": oof(lin(), Xi, y, fl),
        "Linear log (exp)": oof(lin(), X, y, fl, log=True),
        "Linear log + smearing": oof(lin(), X, y, fl, log=True, smear=True),
        "Linear log + interaction + smearing": oof(lin(), Xi, y, fl, log=True, smear=True),
        "GBM": oof(C.final_model(), X, y, fl),
        "GBM log (exp)": oof(C.final_model(), X, y, fl, log=True),
        "GBM log + smearing": oof(C.final_model(), X, y, fl, log=True, smear=True),
    }
    base = np.zeros(len(yv))
    for tr, va in fl:
        base[va] = yv[tr].mean()
    P = {"Baseline": base, **P}
    tab = pd.DataFrame({k: summ(yv, v, smk, top) for k, v in P.items()}).T
    # 스미어링 계수 (fold 별)
    smear = {}
    for name, model, XX in [("Linear log", lin(), X), ("GBM log", C.final_model(), X)]:
        fs = []
        for tr, va in fl:
            m = clone(model).fit(XX.iloc[tr], np.log(y.iloc[tr]))
            fs.append(float(np.mean(np.exp(np.log(y.iloc[tr]) - m.predict(XX.iloc[tr])))))
        smear[name] = fs
    # ① 신뢰구간
    ci = {}
    for name in ["Linear", "Linear + interaction", "GBM"]:
        p = P[name]
        fold_mae = [np.mean(np.abs(p[va][smk[va]] - yv[va][smk[va]])) for tr, va in fl]
        ci[name] = dict(smoker_MAE=float(np.mean(np.abs(p[smk] - yv[smk]))), smoker_CI=boot_ci(yv, p, smk).tolist(),
                        smoker_fold_min=float(min(fold_mae)), smoker_fold_max=float(max(fold_mae)),
                        top_MAE=float(np.mean(np.abs(p[top] - yv[top]))), top_CI=boot_ci(yv, p, top).tolist())
    nst = nested(X, y, fl) if do_nested else None
    exp = export_linear_interaction(X, y) if save else None
    out = dict(table=tab, preds=P, smear=smear, ci=ci, nested=nst, export=exp, smk=smk, top=top, y=yv, X=X, df=df)
    if verbose:
        print("smokers=%d  top10%%=%d  cost = %.0f x under$ + %.0f x over$ (per person)" % (smk.sum(), top.sum(), COST_UNDER, COST_OVER))
        print("\n[before / after] 10-decile stratified 5-fold OOF, dollar scale")
        print(tab[["MAE", "RMSE", "R2", "MAE_smoker", "MAE_top10", "bias", "bias_top10", "under_rate", "cost_3to1", "mean_pred"]].round(
            {"MAE": 0, "RMSE": 0, "R2": 4, "MAE_smoker": 0, "MAE_top10": 0, "bias": 0, "bias_top10": 0, "under_rate": 3, "cost_3to1": 0, "mean_pred": 0}).to_string())
        print("mean actual charges = %.0f" % yv.mean())
        print("\n[smearing factor per fold] mean(exp(log residual)) on training fold")
        for k, v in smear.items():
            print("%-12s %s" % (k, [round(x, 4) for x in v]))
        print("\n[95% bootstrap CI, 2000 bootstrap draws] smoker MAE and top10% MAE")
        for k, v in ci.items():
            print("%-22s smoker MAE %.0f  CI [%.0f, %.0f]  fold range [%.0f, %.0f] | top10 MAE %.0f  CI [%.0f, %.0f]" % (
                k, v["smoker_MAE"], v["smoker_CI"][0], v["smoker_CI"][1], v["smoker_fold_min"], v["smoker_fold_max"],
                v["top_MAE"], v["top_CI"][0], v["top_CI"][1]))
        if nst:
            print("\n[hyperparameter choice, GBM grid %d settings]" % nst["grid_size"])
            print("pick & score on same 5 folds : best %s  MAE %.0f  (optimistic)" % (nst["same_best"], nst["same_mae"]))
            print("nested CV (inner 5-fold pick): MAE %.0f +/- %.0f  folds %s" % (nst["nested_mae"], nst["nested_std"], [round(v) for v in nst["nested_fold_mae"]]))
            print("inner picks per outer fold   : %s" % nst["nested_picks"])
        if exp:
            print("\nsaved data/linear_interaction.json  columns=%s" % exp["columns"])
    return out


if __name__ == "__main__":
    run()
