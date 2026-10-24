"""
AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전 · 정식 코드 f2.py
과제 문항 2 「흡연자만, 비흡연자만, 고액만: 그룹별 오차」
무엇을, 어떤 순서로:
 ① 1편(f1.py)이 저장한 검증 예측 data/oof_predictions.csv 를 읽는다 (없으면 f1 을 먼저 실행)
    - 모든 사람이 정확히 한 번 검증 fold 에 들어갔을 때의 예측이라 "학습에 쓰지 않은 예측"이다
 ② 그룹: 전체 · 흡연자만 · 비흡연자만 · 고액 상위 10%만 (실제 charges 기준)
 ③ 그룹마다 MAE · RMSE · MAPE(보조) · 평균 잔차(편향 = 예측 − 실제) · 과소 추정 비율
    - MAPE 는 이 데이터에 0원이 없어서(최소 $1,122) 계산할 수 있다. 0원이 있는 데이터에서는 쓸 수 없다
 ④ 전체 MSE = Σ (그룹 비율 x 그룹 MSE) 를 실제 숫자로 확인한다
 ⑤ fold 별 흡연자 MAE 가 얼마나 흔들리는지 본다 (흡연자는 fold 당 약 55명)
"""
import os
import numpy as np
import pandas as pd
import common as C


def load_oof():
    if not os.path.exists("data/oof_predictions.csv"):
        import f1
        f1.run(verbose=False)
    return pd.read_csv("data/oof_predictions.csv")


def group_metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = p - y
    return dict(n=len(y), MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))), MSE=float(np.mean(e ** 2)),
                MAPE=float(np.mean(np.abs(e) / y)), bias=float(np.mean(e)), under_rate=float(np.mean(p < y)))


def run(verbose=True):
    df, X, y, info = C.load_clean_xy()
    oof = load_oof()
    smk = df.loc[oof["row"], "smoker"].eq("yes").values
    yv = oof["y"].values
    top = yv >= np.quantile(yv, 0.9)
    groups = {"all": np.ones(len(yv), bool), "smoker": smk, "non-smoker": ~smk, "top 10%": top}
    cols = [c for c in oof.columns if "|" in c]
    rows = []
    for c in cols:
        for g, mask in groups.items():
            rows.append(dict(model=c.split("|")[0], target=c.split("|")[1], group=g, **group_metrics(yv[mask], oof[c].values[mask])))
    tab = pd.DataFrame(rows)
    # ④ MSE 분해: 흡연자 / 비흡연자
    dec = []
    for c in cols:
        p = oof[c].values
        mse_all = np.mean((p - yv) ** 2)
        w_s, w_n = smk.mean(), (~smk).mean()
        mse_s, mse_n = np.mean((p[smk] - yv[smk]) ** 2), np.mean((p[~smk] - yv[~smk]) ** 2)
        dec.append(dict(model=c, MSE_all=mse_all, w_smoker=w_s, MSE_smoker=mse_s, w_non=w_n, MSE_non=mse_n,
                        weighted_sum=w_s * mse_s + w_n * mse_n, smoker_share_of_error=w_s * mse_s / mse_all))
    dec = pd.DataFrame(dec).set_index("model")
    # ⑤ fold 별 흡연자 MAE
    fr = []
    for c in cols:
        for k in sorted(oof["fold"].unique()):
            m = (oof["fold"].values == k)
            fr.append(dict(model=c, fold=int(k), n_smoker=int((m & smk).sum()),
                           MAE_smoker=float(np.mean(np.abs(oof[c].values[m & smk] - yv[m & smk]))),
                           MAE_non=float(np.mean(np.abs(oof[c].values[m & ~smk] - yv[m & ~smk])))))
    fr = pd.DataFrame(fr)
    if verbose:
        print("people: all=%d  smoker=%d  non-smoker=%d  top10%%=%d (charges >= %.0f)" % (
            len(yv), smk.sum(), (~smk).sum(), top.sum(), np.quantile(yv, 0.9)))
        print("top10%% that are smokers: %d" % (top & smk).sum())
        show = tab[tab.model.isin(["Baseline", "Linear", "Gradient Boosting", "Decision Tree"])]
        for g in groups:
            print("\n[group = %s]" % g)
            t = show[show.group == g].set_index(["model", "target"])[["n", "MAE", "RMSE", "MAPE", "bias", "under_rate"]]
            print(t.round({"MAE": 0, "RMSE": 0, "MAPE": 3, "bias": 0, "under_rate": 3}).to_string())
        print("\n[MSE decomposition] MSE_all = w_smoker*MSE_smoker + w_non*MSE_non")
        print(dec.loc[["Baseline|raw", "Linear|raw", "Gradient Boosting|raw", "Gradient Boosting|log"]].round(
            {"MSE_all": 0, "w_smoker": 4, "MSE_smoker": 0, "w_non": 4, "MSE_non": 0, "weighted_sum": 0, "smoker_share_of_error": 3}).to_string())
        print("\n[smoker MAE per fold]")
        t = fr[fr.model.isin(["Linear|raw", "Gradient Boosting|raw"])].pivot(index="model", columns="fold", values="MAE_smoker")
        t["min"], t["max"] = t.min(axis=1), t.max(axis=1)
        print(t.round(0).to_string())
        print("smokers per fold:", list(fr[fr.model == "Linear|raw"]["n_smoker"]))
    return dict(table=tab, dec=dec, folds=fr, smk=smk, top=top, y=yv, oof=oof)


if __name__ == "__main__":
    run()
