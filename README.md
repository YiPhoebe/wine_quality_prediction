# 🍷 레드 와인 품질 예측 프로젝트

이 프로젝트는 **레드 와인**과 **화이트 와인**의 품질을 예측하는 머신 러닝 모델을 구축하는 것입니다. 주어진 데이터셋을 사용하여 **회귀** 및 **분류** 모델을 학습시키고, 다양한 모델을 앙상블하여 예측 성능을 향상시키는 방법을 구현합니다.

---

## 📊 데이터셋


- **데이터 출처**:
  - `winequality-red.csv`
  - https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
- - `winequality-white.csv`
  - https://www.kaggle.com/code/gcdatkin/white-wine-quality-prediction/input
  - 
- **설명**: 각 레드 와인과 화이트 와인 샘플의 여러 화학적 특성(`pH`, `alcohol`, `sulphates` 등)과 품질 점수(`quality`)가 포함된 데이터셋입니다.
- **목표**: 주어진 화학적 특성들로부터 와인의 품질을 예측하는 모델을 생성합니다.

---

## 🛠️ 사용된 알고리즘

### 1. **회귀 모델 (Regression Models)**

#### - **선형 회귀 (Linear Regression)**:

- 품질 점수를 예측하기 위해 사용되었습니다.

#### - **랜덤 포레스트 회귀 (Random Forest Regressor)**:

- **비선형** 관계를 모델링하여 품질 예측을 수행합니다.

#### - **XGBoost 회귀 (XGBoost Regressor)**:

- **복잡한 특성 간 상호작용**을 학습하여 예측 성능을 최적화합니다.

---

### 2. **분류 모델 (Classification Models)**

#### - **로지스틱 회귀 (Logistic Regression)**:

- 품질 점수를 **이진 분류** 문제로 변환하여 "좋음(6점 이상)"과 "나쁨(6점 미만)"을 구분하는데 사용되었습니다.

#### - **서포트 벡터 머신 (SVM)**:

- **SVC** 모델을 사용하여 **이진 분류** 문제를 해결했습니다.

#### - **랜덤 포레스트 분류 (Random Forest Classifier)**:

- 여러 개의 **결정 트리**를 결합한 앙상블 모델로, **과적합**을 방지하고 **비선형 관계**를 잘 처리할 수 있습니다.

#### - **XGBoost 분류 (XGBoost Classifier)**:

- **XGBoost**를 사용하여 분류 문제를 해결했습니다. **Gradient Boosting**을 기반으로 여러 모델을 **앙상블**하여 성능을 최적화합니다.

#### - **앙상블 기법 (Voting Classifier)**:

- `XGBoost`, **로지스틱 회귀**, **SVM**을 결합하여 **다수결 투표 방식**으로 최종 예측을 도출합니다.

---

## ⚙️ 설치 방법

1. 프로젝트를 클론하거나 다운로드합니다.
2. 필요한 패키지를 설치합니다:

   ```bash
   pip install -r requirements.txt
   ```

**requirements.txt** 파일에는 다음과 같은 필수 패키지들이 포함됩니다:

• xgboost

• scikit-learn

• pandas

• numpy

• matplotlib

• seaborn

---

## 🤖모델 학습

### 1. 데이터 전처리

**winequality-red.csv** 파일을 로드한 후, **quality** 컬럼을 **이진 분류**로 변환합니다. 6점 이상은 “Good(1)”, 미만은 “Not Good(0)“으로 변환합니다.

### 2.모델훈련

여러 모델을 사용하여 훈련을 진행합니다:

#### 회귀 모델

• 선형 회귀 : **OLS** 회귀 모델을 사용하여 품질 점수를 예측합니다.

• 랜덤 포레스트 회귀 : **RandomForestRegressor** 모델을 사용하여 비선형 관계를 학습합니다.

• XGBoost 회귀 : **XGBoost** 모델을 사용하여 복잡한 상호작용을 모델링합니다

#### 분류 모델

• 로지스틱 회귀 : **LogisticRegression**을 사용하여 이진 분류 문제를 해결합니다.

• SVM : **SVC** 모델을 사용하여 비선형 분류 문제를 학습합니다.

• 랜덤 포레스트 분류 **: **RandomForestClassifier**를 사용하여 다수의 결정 트리를 결합하여 예측합니다.**

• XGBoost 분류 : **XGBoost** 모델을 사용하여 앙상블 방식으로 성능을 최적화합니다.

### 3.앙상블 모델

**VotingClassifier**를 사용하여 **XGBoost**, **로지스틱 회귀**, **SVM** 모델을 결합합니다. 이 방식은 각 모델의 예측을 **다수결 투표 방식**으로 결합하여 최종 예측을 도출합니다.

### 4. 성능 평가

모델 성능은 **정확도(Accuracy)** , **정밀도(Precision)**, **재현율(Recall)** 등을 사용하여 평가합니다.

### 5. 혼동 행렬 시각화

각 모델의 예측 결과에 대한 **혼동 행렬**을 시각화하여 예측 성능을 분석합니다.

---

## ▶️ 실행 방법

1. 코드 파일을 실행하여 모델을 학습하고 평가합니다:

```
python wine_quality_model.py
```

**2. 앙상블 모델**의 예측 결과가 출력되고, **혼동 행렬**이 시각화됩니다.

---

## 🏆 결과

• **앙상블 모델**은 약 76%의 정확도로 예측 성능을 보였으며, **XGBoost**와 **로지스틱 회귀**, **SVM**의 결합으로 좋은 성능을 발휘했습니다.

• 회귀 모델에서는 **랜덤 포레스트 회귀**와 **XGBoost 회귀** 모델이 좋은 성능을 보였으며, **선형 회귀** 모델은 상대적으로 성능이 떨어졌습니다.

---

## 📚 참고

• XGBoost : [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)

• Scikit-learn : [https://scikit-learn.org/](https://scikit-learn.org/)

• SVM**: [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)
