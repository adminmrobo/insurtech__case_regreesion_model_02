**[한국어](#ko)** · English · 中文 · 日本語 · O'zbekcha · Русский (번역 예정)

---
<a id="ko"></a>
# AI(인공지능) x 비즈니스 #1 인슈어테크_04_회귀모델_실전

건강보험 의료비 회귀 과제의 다섯 문항을 한 편에 한 문항씩 따라 푸는 인터랙티브 튜토리얼입니다. 이상치를 지우지 않고 5-Fold 교차검증으로 성능을 재고, 흡연자만의 오차를 정의하고, 그 숫자를 성능이라 부를 때의 함정을 고치고, 모델을 서버 없는 HTML 하나로 내보내고, 다른 산업의 같은 문제와 비교합니다. 1부 개념편: 인슈어테크_03 회귀모델.

- 글: 글 · 안상선 ((주)M-Robo 대표)
- 라이선스: CC BY-NC 4.0 (글·코드·그림). 데이터는 원 출처의 조건을 따릅니다.
- 공개 주소: insurtech__case_regreesion_model_02/ (  바꾸려면 build/kit.py 의 CFG 와 code/common.py 의 REPO_URL 을 고친 뒤 다시 빌드)
- 1부 개념편: insurtech__case_regreesion_model_01/

## 언어

| 언어 | 폴더 | 상태 |
|---|---|---|
| 한국어 | `/` | 원본 |
| English · 中文 · 日本語 · O'zbekcha · Русский | `en/` `zh/` `ja/` `uz/` `ru/` | 번역 예정 |

페이지 오른쪽 위 언어 상자로 고릅니다. 선택은 브라우저에 기억되고(`it4_lang`), 주소에 `?lang=xx`를 붙여도 됩니다.

## 편 목록

| 편 | 과제 | 파일 | 제목 | 유형 |
|---|---|---|---|---|
| 1 | 문항 1 | `1_kfold_cv.html` | 이상치를 지우지 않고 5-Fold로 성능 재기 | 해설형 · 확인 문제 4 |
| 2 | 문항 2 | `2_group_errors.html` | 흡연자만, 비흡연자만, 고액만: 그룹별 오차 | 해설형 · 확인 문제 4 |
| 3 | 문항 3 | `3_problems_fixes.html` | 그 숫자를 성능이라 부르면 생기는 일 | 방법형 · 확인 문제 4 |
| 4 | 문항 4 | `4_html_deploy.html` | 그 모델, HTML에 그대로 넣어도 될까 | 정리형 · 확인 문제 4 · 가이드라인 |
| 5 | 문항 5 | `5_similar_cases.html` | 꼬리가 긴 금액은 어디에나 있다 | 정리형 · 확인 문제 4 · 가이드라인 |

처음 화면 `index.html` · 교육자용 정답 `answers.html` (#a1 … #a5) · 확인 문제 모두 20개

## 폴더

```
├── index.html · 1_… ~ 5_….html · answers.html · 404.html
├── assets/                   식구 그림(인슈어테크_01·02 와 같은 파일) · hero.webp(키 비주얼) · s{n}.webp(편 끝 장면 1440×800) · c{n}.webp(지도 카드, 장면 가운데 720×660)
├── code/                     정식 코드: common.py(1부·2부 같은 파일), f1.py ~ f5.py
│   └── easy/                 초보용 코드: e1.py ~ e5.py (def·for·if·lambda·들여쓰기 없음)
├── data/                     cases.json · deploy_gbm.json · deploy_linear.json · deploy_loglinear.json · insurance.csv · linear_interaction.json · oof_predictions.csv · split.csv
├── prompts/                  AI 튜터 프롬프트 ai_prompt_1.md ~ ai_prompt_5.md
├── README.md · LICENSE.md · .nojekyll
```

## 데이터

- 원 출처: https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv
- `data/insurance.csv` 는 위 파일을 **그대로 복사**한 것입니다. md5 `d5364d06246fb4bfa4cc7d1ee89ebd0d` (원본과 같음). 1,338행 × 7열(age, sex, bmi, children, smoker, region, charges).
- 코드(`common.load_raw`)는 저장소 안 파일 → 저장소 raw 주소 → 원 출처 주소 순서로 읽습니다. 환경변수 `INS_DATA` 로 경로를 줄 수도 있습니다.
- 정리 규칙(`common.clean`): 완전히 같은 행 1건(행 번호 581)만 뺀다 → 1,337행. 맞힐 금액(charges)은 자르거나 고치지 않는다. IQR 밖 charges 139행 · bmi 9행은 그대로 둔다. 결측 없음.
- **가상 데이터 생성 · 데이터 증강(합성 행 · 행 복제 · 정답에 잡음)을 하지 않습니다.**
- 1부 나누기: charges 10분위 층화 80:20, `random_state=42` → 학습 1,069 / 시험 268 (`data/split.csv`). 2부 검증: 같은 10분위로 층화한 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- 기준선(모두에게 학습 평균 $13,250): 1부 시험 MAE $9,151 · R² −0.0001 · 고액 상위 10% MAE $28,371.
- 재현: `cd code && python f1.py` … (저장소 루트에서 `python code/f1.py` 도 됩니다)

## 주의

이 시리즈는 교육용 설명입니다. 보험 가입 · 보험료 · 보험금 심사 · 투자 · 의료에 대한 조언이 아닙니다. 데이터는 공개 학습 데이터 원본이며, 실제 보험사의 현재 보험료 산정 · 심사 기준을 뜻하지 않습니다. 건강 정보(흡연 · BMI)는 설명용이며 의료 조언이 아닙니다. 실제 보험료 산정에서 성별 · 지역 · 나이 같은 특성의 사용은 감독 규정에 따라 금지되거나 제한될 수 있습니다.
