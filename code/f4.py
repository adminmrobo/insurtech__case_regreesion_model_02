"""
AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전 · 정식 코드 f4.py
4편 「그 모델, HTML에 그대로 넣어도 될까」 (과제 문항 4)
무엇을, 어떤 순서로:
 ① 데이터 1,337행 전체로 배포용 모델 두 개를 학습한다
    (가) 상호작용 선형 회귀: 표준화 + LinearRegression, 특성 11개(흡연 x 비만 포함)
    (나) 최종 그래디언트 부스팅(트리 200개, 깊이 3)
 ② 두 모델을 JSON 으로 내보낸다: 열 순서 · 인코딩 사전 · 표준화 평균/표준편차 · 계수 · 절편 / 트리 노드 전체
 ③ JSON 만으로 다시 계산해(브라우저 JS 와 같은 식) sklearn 예측과의 최대 차이를 잰다
 ④ 규칙 누락 실험: 상호작용 규칙을 빼먹으면 / 지역 원-핫 열 순서가 바뀌면 / 로그 모델의 스미어링 계수를 빼먹으면
    -> 견적이 10% 이상 바뀌는 사람 수와 평균 편향
 ⑤ 로그 역변환 편향: 5-Fold 검증 예측으로 exp(예측) 평균 vs 실제 평균, 스미어링(Duan, 1983) 보정 후
 ⑥ 학습 범위 밖 입력(외삽): 나이 75 · BMI 60 · 지역 '기타' 를 넣으면 두 모델이 무엇을 답하나
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import common as C

ENCODING = {
    "sex": {"male": "sex_male"},
    "smoker": {"yes": "smoker_yes"},
    "region": {"northwest": "region_northwest", "southeast": "region_southeast", "southwest": "region_southwest"},
    "base_levels": {"sex": "female", "smoker": "no", "region": "northeast"},
}


def export_linear(pipe, cols):
    sc, lr = pipe.named_steps["standardscaler"], pipe.named_steps["linearregression"]
    return {"model": "LinearRegression + StandardScaler (상호작용 포함)", "columns": cols,
            "mean": [float(v) for v in sc.mean_], "scale": [float(v) for v in sc.scale_],
            "coef": [float(v) for v in lr.coef_], "intercept": float(lr.intercept_),
            "encoding": ENCODING, "interaction_rules": {"bmi30": "bmi >= 30", "smoker_bmi30": "smoker_yes * bmi30",
                                                        "smoker_bmi": "smoker_yes * bmi"},
            "input_range": {"age": [18, 64], "bmi": [15.96, 53.13], "children": [0, 5],
                            "region": ["northeast", "northwest", "southeast", "southwest"]}}


def export_gbm(gbm, cols):
    trees = []
    for est in gbm.estimators_[:, 0]:
        t = est.tree_
        trees.append({"l": t.children_left.tolist(), "r": t.children_right.tolist(), "f": t.feature.tolist(),
                      "t": [round(float(v), 10) for v in t.threshold], "v": [float(v) for v in t.value[:, 0, 0]]})
    init = float(gbm.init_.constant_[0][0]) if hasattr(gbm.init_, "constant_") else float(gbm._raw_predict_init(np.zeros((1, len(cols))))[0][0])
    return {"model": "GradientBoostingRegressor", "params": C.FINAL_PARAMS, "columns": cols, "init": init,
            "learning_rate": float(gbm.learning_rate), "trees": trees, "encoding": ENCODING}


# JSON 만으로 다시 계산 (페이지의 JS 와 같은 식)
def lin_json_predict(J, Xm):
    Z = (Xm[J["columns"]].values - np.array(J["mean"])) / np.array(J["scale"])
    return Z @ np.array(J["coef"]) + J["intercept"]


def gbm_json_predict(J, Xm):
    A = Xm[J["columns"]].values
    out = np.full(len(A), J["init"])
    for tr in J["trees"]:
        for i in range(len(A)):
            k = 0
            while tr["l"][k] != -1:
                k = tr["l"][k] if A[i, tr["f"][k]] <= tr["t"][k] else tr["r"][k]
            out[i] += J["learning_rate"] * tr["v"][k]
    return out


def run(verbose=True, save=True):
    df, X, y, info = C.load_clean_xy()
    XI = C.add_interactions(X)
    cols_i = list(XI.columns)
    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(XI, y)
    gbm = C.final_model().fit(X, y)
    JL, JG = export_linear(lin, cols_i), export_gbm(gbm, list(X.columns))
    sL = json.dumps(JL, ensure_ascii=False, separators=(",", ":"))
    sG = json.dumps(JG, ensure_ascii=False, separators=(",", ":"))
    if save:
        os.makedirs("data", exist_ok=True)
        open("data/deploy_linear.json", "w", encoding="utf-8").write(sL)
        open("data/deploy_gbm.json", "w", encoding="utf-8").write(sG)
    pL, pG = lin.predict(XI), gbm.predict(X)
    jL, jG = lin_json_predict(JL, XI), gbm_json_predict(JG, X)
    diffL, diffG = float(np.max(np.abs(pL - jL))), float(np.max(np.abs(pG - jG)))
    agreeL = float(np.mean(np.round(pL, 2) == np.round(jL, 2)))
    agreeG = float(np.mean(np.round(pG, 2) == np.round(jG, 2)))

    # ④ 규칙 누락
    X_noint = XI.copy()
    X_noint[["bmi30", "smoker_bmi30", "smoker_bmi"]] = 0.0          # 상호작용 계산을 빼먹음 -> 0 으로 들어감
    p_noint = lin.predict(X_noint)
    X_swap = XI.copy()
    X_swap[["region_northwest", "region_southwest"]] = XI[["region_southwest", "region_northwest"]].values  # 열 순서가 바뀜
    p_swap = lin.predict(X_swap)
    # 로그 모델 (문항 1 의 '로그 타깃 선형 회귀', 특성 8개) + 스미어링
    linlog = make_pipeline(StandardScaler(), LinearRegression()).fit(X, np.log(y))
    res = np.log(y) - linlog.predict(X)
    smear = float(np.mean(np.exp(res)))
    p_log_ok = np.exp(linlog.predict(X)) * smear
    p_log_forgot = np.exp(linlog.predict(X))

    def changed(a, b):
        r = np.abs(b - a) / a
        return int(np.sum(r >= 0.10)), float(np.mean(b - a))

    omit = {"interaction": changed(pL, p_noint), "region_order": changed(pL, p_swap), "smearing": changed(p_log_ok, p_log_forgot)}

    # ⑤ 로그 역변환 편향 (5-Fold 검증 예측: 스미어링 계수도 학습 fold 에서만, 같은 로그 타깃 선형 회귀)
    oof_raw, oof_sm = np.zeros(len(y)), np.zeros(len(y))
    smears = []
    for tr, va in C.folds(y):
        m = make_pipeline(StandardScaler(), LinearRegression()).fit(X.iloc[tr], np.log(y.iloc[tr]))
        s = float(np.mean(np.exp(np.log(y.iloc[tr]) - m.predict(X.iloc[tr]))))
        smears.append(s)
        oof_raw[va] = np.exp(m.predict(X.iloc[va]))
        oof_sm[va] = oof_raw[va] * s
    smk = (X["smoker_yes"] == 1).values
    grp = {}
    for g, mask in (("all", np.ones(len(y), bool)), ("smoker", smk), ("non-smoker", ~smk)):
        a = float(y[mask].mean())
        grp[g] = {"n": int(mask.sum()), "actual": a, "exp": float(oof_raw[mask].mean()), "smear": float(oof_sm[mask].mean()),
                  "exp_gap": float(oof_raw[mask].mean() / a - 1), "smear_gap": float(oof_sm[mask].mean() / a - 1),
                  "mae_exp": float(np.mean(np.abs(y[mask] - oof_raw[mask]))), "mae_smear": float(np.mean(np.abs(y[mask] - oof_sm[mask])))}
    bias = {"groups": grp, "smear_folds": smears, "median_actual": float(np.median(y)), "median_exp": float(np.median(oof_raw)),
            "below_exp": float(np.mean(oof_raw < y))}

    # 배포 후보 두 모델의 5-Fold 검증 MAE (같은 분할)
    oofL, oofG = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in C.folds(y):
        oofL[va] = make_pipeline(StandardScaler(), LinearRegression()).fit(XI.iloc[tr], y.iloc[tr]).predict(XI.iloc[va])
        oofG[va] = C.final_model().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    cv = {"linear_inter": C.metrics(y, oofL), "gbm": C.metrics(y, oofG), "log_linear": C.metrics(y, oof_raw),
          "log_linear_smear": C.metrics(y, oof_sm)}
    JLog = export_linear(linlog, list(X.columns))
    JLog["model"] = "LinearRegression + StandardScaler, 목표 = log(charges)"
    JLog["smearing"] = smear
    JLog.pop("interaction_rules")
    if save:
        open("data/deploy_loglinear.json", "w", encoding="utf-8").write(json.dumps(JLog, ensure_ascii=False, separators=(",", ":")))

    # ⑥ 외삽
    probes = pd.DataFrame([
        dict(case="in range: 45, BMI 34, smoker, 2 kids, southeast", age=45, sex="male", bmi=34.0, children=2, smoker="yes", region="southeast"),
        dict(case="age 75 (max 64)", age=75, sex="male", bmi=34.0, children=2, smoker="yes", region="southeast"),
        dict(case="BMI 60 (max 53.13)", age=45, sex="male", bmi=60.0, children=2, smoker="yes", region="southeast"),
        dict(case="age 75 + BMI 60", age=75, sex="male", bmi=60.0, children=2, smoker="yes", region="southeast"),
        dict(case="region 'other' (not in 4)", age=45, sex="male", bmi=34.0, children=2, smoker="yes", region="other"),
        dict(case="BMI exactly 30.0 (boundary)", age=45, sex="male", bmi=30.0, children=2, smoker="yes", region="southeast"),
        dict(case="BMI 29.9", age=45, sex="male", bmi=29.9, children=2, smoker="yes", region="southeast"),
        dict(case="age 10 (min 18)", age=10, sex="male", bmi=20.0, children=0, smoker="no", region="northeast"),
    ])
    Xp, _ = C.encode(probes.drop(columns="case").assign(charges=0.0))
    ext = probes[["case"]].copy()
    ext["linear"] = lin.predict(C.add_interactions(Xp))
    ext["gbm"] = gbm.predict(Xp)

    # GBM 이 bmi 를 자른 경계값 중 30 근처
    thr = sorted({round(float(t), 4) for tr in JG["trees"] for f, t in zip(tr["f"], tr["t"]) if f == 1 and 29 < t < 31})
    out = dict(cv=cv, JLog=JLog, gbm_bmi_thr=thr, JL=JL, JG=JG, bytes_lin=len(sL.encode()), bytes_gbm=len(sG.encode()), diffL=diffL, diffG=diffG,
               agreeL=agreeL, agreeG=agreeG, omit=omit, smear=smear, bias=bias, ext=ext, df=df, X=X, XI=XI, y=y,
               pL=pL, pG=pG, p_noint=p_noint, p_swap=p_swap, p_log_ok=p_log_ok, p_log_forgot=p_log_forgot,
               n_nodes=int(sum(len(t["v"]) for t in JG["trees"])), lin_mae=float(np.mean(np.abs(y - pL))), gbm_mae=float(np.mean(np.abs(y - pG))))
    if verbose:
        print("rows=%d  deploy models fitted on all rows" % len(y))
        print("[export] linear JSON %d bytes (%d columns) | GBM JSON %d bytes (%d trees, %d nodes)" % (
            out["bytes_lin"], len(cols_i), out["bytes_gbm"], len(JG["trees"]), out["n_nodes"]))
        print("[check] JSON-only vs sklearn  max|diff| linear=%.3e  gbm=%.3e  | same to the cent: linear %.4f  gbm %.4f" % (diffL, diffG, agreeL, agreeG))
        print("[omission] quotes changed >=10%% (of %d) and mean change" % len(y))
        for k, (n, b) in omit.items():
            print("  %-13s changed=%4d (%.1f%%)  mean change=%+.0f" % (k, n, n / len(y) * 100, b))
        print("smearing factor (all rows) = %.4f" % smear)
        print("[5-fold CV, same folds] MAE / RMSE / R2 / MAE_top10 / under_rate")
        for k, m in cv.items():
            print("  %-17s %6.0f %6.0f %.4f %6.0f %.3f" % (k, m["MAE"], m["RMSE"], m["R2"], m["MAE_top10"], m["under_rate"]))
        print("[log back-transform, log-target linear, 5-fold OOF]")
        print("  group        n   actual   exp(pred)  gap     smeared  gap     MAE exp  MAE smeared")
        for g, d in bias["groups"].items():
            print("  %-10s %5d  %7.0f  %7.0f  %+6.1f%%  %7.0f  %+6.1f%%  %7.0f  %7.0f" % (
                g, d["n"], d["actual"], d["exp"], d["exp_gap"] * 100, d["smear"], d["smear_gap"] * 100, d["mae_exp"], d["mae_smear"]))
        print("  smearing factor by fold:", " ".join("%.4f" % s for s in smears))
        print("  median actual=%.0f  median exp(pred)=%.0f  share of people with exp(pred) < actual=%.3f" % (
            bias["median_actual"], bias["median_exp"], bias["below_exp"]))
        print("GBM bmi split thresholds near 30:", thr)
        print("[extrapolation] predictions ($)")
        print(ext.round(0).to_string(index=False))
    return out


if __name__ == "__main__":
    run()
