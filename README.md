# Hybrid Predictive Maintenance with Physics-Informed Feature Engineering

**Solo Research Project** | **July 2025** | **2 weeks development time**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-green.svg)](https://xgboost.readthedocs.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

---

## Executive Summary

Developed a hybrid machine learning pipeline combining physics-informed feature engineering with gradient boosting to predict equipment failures and classify specific failure modes in manufacturing systems. Achieved **56% improvement in F1-score** (0.53 → 0.83) by engineering domain-specific features that capture thermal, mechanical, and wear dynamics.

**Key Results:**
- Binary failure prediction: **F1 = 0.83, PR-AUC = 0.85**
- Multi-mode classification: **97% accuracy for heat dissipation failures**
- Novel Weibull-based approach for tool wear failure detection
- Complete end-to-end implementation with model interpretability analysis

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Technical Approach](#technical-approach)
- [Feature Engineering](#feature-engineering)
- [Methodology](#methodology)
- [Results & Analysis](#results--analysis)
- [Implementation Details](#implementation-details)
- [Key Insights](#key-insights)
- [Future Work](#future-work)
- [Technical Stack](#technical-stack)

---

## Problem Statement

### The Challenge

Most predictive maintenance research treats equipment failure as a binary problem - either it breaks or it doesn't. However, understanding **why** something fails is equally important as knowing **when** it will fail. Different failure modes require different maintenance strategies:

- **Heat Dissipation Failure** → Check cooling systems
- **Power Failure** → Inspect electrical components  
- **Overstrain Failure** → Reduce operational loads
- **Tool Wear Failure** → Schedule replacements

### Technical Challenges

1. **Extreme class imbalance** - Only 3.4% failure rate overall
2. **Sparse failure modes** - Some modes have <20 samples in 10,000 observations
3. **Non-exclusive failure types** - Multiple modes can co-occur
4. **Raw sensor limitations** - Basic telemetry doesn't capture underlying physics

### Solution Overview

Developed a hybrid approach combining:
- **Physics-informed feature engineering** (4 new features based on thermodynamics/mechanics)
- **Statistical modeling** (Weibull distribution for tool wear analysis)
- **Advanced ML techniques** (One-vs-rest XGBoost with iterative stratification)

---

## Technical Approach

### Dataset: AI4I 2020 Predictive Maintenance

**Source:** Publicly available manufacturing dataset  
**Size:** 10,000 observations of discrete operating cycles  
**Features:** 6 raw sensor measurements + 5 binary failure mode indicators

#### Raw Features
| Feature | Description | Unit |
|---------|-------------|------|
| `type` | Part category (L/M/H) | Categorical |
| `air_temperature` | Ambient temperature | Kelvin |
| `process_temperature` | Internal process temperature | Kelvin |
| `rotational_speed` | System rotation speed | RPM |
| `torque` | Applied torque | Nm |
| `tool_wear` | Cumulative tool wear | Minutes |

#### Target Variables
| Failure Mode | Support | Description |
|--------------|---------|-------------|
| Tool Wear Failure (TWF) | 46 | Tool degradation beyond limits |
| Heat Dissipation Failure (HDF) | 115 | Thermal management issues |
| Power Failure (PWF) | 95 | Electrical/power system failure |
| Overstrain Failure (OSF) | 98 | Mechanical overload |
| Random Failure (RNF) | 19 | Stochastic/unexplained failures |
| **Total machine_failure** | **339** | **Any failure occurrence** |

---

## Feature Engineering

### Physics-Informed Feature Design

Rather than relying solely on raw sensor data, I engineered 4 features based on first-principles physics that capture the underlying degradation mechanisms.

#### 1. Temperature Difference
```python
temp_diff = process_temperature - air_temperature
```
**Physics rationale:** Internal heat load indicates thermal stress on components. Critical for identifying heat dissipation failures.

#### 2. Heating Risk (Scaled)
```python
heating_risk_scaled = (1 / (temp_diff × rotational_speed)) × 10^5
```
**Physics rationale:** Thermal damage increases with prolonged heat accumulation during high-speed operation. Inverse relationship captures risk threshold.

#### 3. Mechanical Power
```python
power = torque × rotational_speed × (2π/60)
```
**Physics rationale:** Converts RPM and torque to actual mechanical power (Watts). Direct indicator of energy transfer and mechanical stress.

#### 4. Cumulative Work
```python
mechanical_work = torque × tool_wear
```
**Physics rationale:** Total mechanical stress experienced by tool over lifetime. Captures accumulated damage for wear-based failures.

---

## Methodology

### Data Preprocessing

**Class Balancing Strategy:**
- Stratified sampling to preserve failure distribution
- `scale_pos_weight` parameter tuning for XGBoost
- Iterative stratification for multi-label classification

**Validation Strategy:**
- 5-fold stratified cross-validation for binary classification
- Iterative stratification for failure mode classification
- 80/10/10 train/validation/test split

### Modeling Approach

#### 1. Binary Failure Prediction
```python
model = XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss'
)
```

**Two variants compared:**
- **Raw Model:** 6 original features only
- **Engineered Model:** 6 original + 4 engineered features

#### 2. Multi-Mode Classification

**One-vs-Rest Approach:**
- Separate XGBoost classifier for each failure mode
- Individual threshold optimization via precision-recall curves
- F1-score maximization for each mode

#### 3. Tool Wear Failure: Statistical Approach

**Weibull Distribution Analysis:**
```python
from scipy.stats import weibull_min
c, loc, lam = weibull_min.fit(wear_fail, floc=0)
```

**Key implementation details:**
- Forced location parameter `floc=0` (wear cannot be negative)
- Conservative thresholding at 80th percentile
- Complementary approach to ML prediction

### Advanced Techniques Used

1. **Iterative Stratification** - Maintains class balance across all failure modes simultaneously
2. **Precision-Recall Optimization** - F1-maximizing thresholds for each mode
3. **Survival Analysis** - Weibull modeling for time-to-failure prediction
4. **Model Interpretability** - SHAP analysis for feature importance

---

## Results & Analysis

### Binary Classification Performance

| Metric | Raw Features | Engineered Features | **Improvement** |
|--------|--------------|---------------------|-----------------|
| **F1 Score** | 0.532 ± 0.018 | **0.830 ± 0.022** | **+56%** |
| **PR-AUC** | 0.559 ± 0.020 | **0.853 ± 0.019** | **+53%** |


### Failure Mode Classification Results

| Mode | F1 Score | PR-AUC | Support | Status |
|------|----------|--------|---------|--------|
| **HDF** | **0.966** | **0.997** | 58 |  Excellent |
| **PWF** | **0.880** | **0.971** | 48 |  Very Good |
| **OSF** | **0.805** | **0.964** | 49 |  Good |
| **TWF** | 0.042 | 0.080 | 23 |  Challenging |
| **RNF** | 0.000 | 0.002 | 9 |  Impossible |


### Model Interpretability Analysis

**SHAP Feature Importance:**

![SHAP Analysis](images\hdf_global_fi.png)
![SHAP Analysis](images\osf_global_fi.png)
![SHAP Analysis](images\pwf_global_fi.png)


**Key Findings:**
1. **`temp_diff`** - Most important for HDF prediction
2. **`power`** - Critical for PWF and OSF detection  
3. **`heating_risk_scaled`** - Captures thermal-mechanical interactions
4. **`mechanical_work`** - Relevant for long-term degradation

![SHAP Analysis](images\hdf_waterfall.png)
![SHAP Analysis](images\osf_waterfall.png)
![SHAP Analysis](images\pwf_waterfall.png)
*SHAP waterfall plot for individual prediction explanation*

### Tool Wear Failure: Weibull Analysis

**Fitted Parameters:**
- Shape parameter (k): 16.04
- Scale parameter (λ): 222.5 minutes
![SHAP Analysis](images\weibull.png)


**Statistical Validation:**
- Kolmogorov-Smirnov test: p-value = 0.312 (good fit)
- Conservative threshold at 80th percentile for early warning


**Robust Cross-Validation:**
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = IterativeStratification(n_splits=5, order=1)
```

**Dynamic Threshold Optimization:**
```python
precision, recall, thresholds = precision_recall_curve(y_val, probs)
f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
best_thresh = thresholds[np.argmax(f1s)]
```

**Model Persistence:**
```python
joblib.dump(model, "xgb_machine_failure_model.joblib")
joblib.dump(full_features, "features_used.joblib")
joblib.dump(best_threshold, "best_threshold.joblib")
```

---

## Key Insights

### Technical Insights

1. **Feature engineering dominates:** 56% improvement demonstrates that domain knowledge still outperforms raw data approaches

2. **Physics-based features are interpretable:** Unlike deep learning black boxes, these features have clear physical meaning

3. **Mode-specific strategies work:** HDF/PWF/OSF benefit from ML approaches, while TWF requires statistical modeling

4. **Class imbalance requires specialized techniques:** Standard approaches fail on rare failure modes

### Business Insights

1. **Thermal failures are highly predictable** - 97% accuracy enables proactive cooling system maintenance

2. **Power failures show clear patterns** - 88% accuracy allows electrical system scheduling

3. **Random failures are likely noise** - Focus maintenance efforts on predictable modes

4. **Tool wear needs survival analysis** - Traditional ML approaches insufficient for wear processes

### Methodological Insights

1. **Hybrid approaches outperform pure ML** - Combining statistical and ML methods leverages strengths of both

2. **Proper evaluation is critical** - Stratified CV and iterative stratification essential for reliable results

3. **Interpretability enables trust** - SHAP analysis provides actionable insights for maintenance teams

---

## Future Work

### Short-term Improvements

**Enhanced Feature Engineering:**
- Time-series features (rolling averages, trends)
- Interaction terms between engineered features
- Frequency domain analysis of rotational signals

**Advanced Modeling:**
- Ensemble methods beyond XGBoost (Random Forest, CatBoost)
- Multi-task learning for simultaneous mode prediction
- Uncertainty quantification with conformal prediction

### Long-term Applications

**Industrial Implementation:**
- Real-time deployment on edge devices
- Integration with CMMS (Computerized Maintenance Management Systems)
- IoT sensor fusion for multi-modal data

**Research Extensions:**
- Transfer learning to other manufacturing domains
- Federated learning across multiple plants
- Physics-informed neural networks (PINNs)


---

## Technical Stack

### Core Technologies
- **Python 3.8+** - Primary development language
- **XGBoost 1.6+** - Gradient boosting framework
- **Scikit-learn 1.0+** - ML pipeline and evaluation
- **SciPy** - Statistical modeling (Weibull distribution)
- **Pandas/NumPy** - Data manipulation and numerical computing

### Specialized Libraries
- **scikit-multilearn** - Multi-label stratification
- **SHAP** - Model interpretability and explainability
- **Matplotlib/Seaborn** - Data visualization
- **Joblib** - Model serialization and persistence

### Development Tools
- **Jupyter Notebook** - Interactive development
- **Git** - Version control
- **Python virtual environment** - Dependency management

---



**Project Duration:** 2 weeks (July 2025)  
**Development Type:** Solo research project  


*This project demonstrates the practical application of domain expertise in machine learning, combining traditional statistical methods with modern ML techniques to solve real industrial problems.*

---

## Appendix

### Reproduction Instructions

1. **Environment Setup:**
```bash
pip install xgboost scikit-learn scipy pandas numpy matplotlib seaborn shap scikit-multilearn joblib
```

2. **Data Download:**
```bash
# Download AI4I 2020 dataset from UCI ML Repository
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv
```


### Performance Metrics Definitions

- **F1 Score:** Harmonic mean of precision and recall
- **PR-AUC:** Area under precision-recall curve (better for imbalanced data than ROC-AUC)
- **Stratified CV:** Cross-validation maintaining class distribution
- **Iterative Stratification:** Multi-label stratification preserving all label combinations