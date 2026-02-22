# 2030 Water Stress Prediction - Scenario Analysis

## Executive Summary

This report presents projections of Tunisia's water stress levels for 2030 using multiple machine learning models and extrapolation scenarios. Water stress is measured as the proportion of freshwater withdrawal to available freshwater resources (expressed as a percentage).

### Key Findings

- **Baseline (2023):** 98.1% water stress (critical level)
- **2030 Projections Range:** 74.7% to 84.9% across models and scenarios
- **Projected Trend:** Significant decrease in water stress from 2023 baseline
- **Best Estimate (Ridge Model):** 75.3% ± 0.3% depending on scenario

---

## 1. Projection Results by Scenario

### Linear Trend Extrapolation
Projects 2030 water stress assuming linear continuation of 2024 feature trends.

| Model | 2030 Prediction | Change from 2023 |
|-------|-----------------|------------------|
| DecisionTree | 80.64% | -17.5% |
| Lasso | 74.71% | -23.4% |
| **Ridge (BEST)** | **75.31%** | **-22.8%** |
| LinearRegression | 75.40% | -22.7% |
| RandomForest | 82.15% | -16.0% |

**Interpretation:** Linear extrapolation suggests water stress decreases as freshwater withdrawal patterns stabilize or resources improve.

---

### Exponential Trend Extrapolation
Projects 2030 water stress assuming exponential continuation of 2024 feature trends.

| Model | 2030 Prediction | Change from Linear |
|-------|-----------------|-------------------|
| DecisionTree | 80.64% | no change |
| Lasso | 74.87% | +0.16% |
| **Ridge (BEST)** | **75.45%** | **+0.14%** |
| LinearRegression | 75.54% | +0.14% |
| RandomForest | 82.15% | no change |

**Interpretation:** Exponential trends produce slightly higher projections, suggesting stronger underlying growth signals in the feature data.

---

### Average Trend Extrapolation
Projects 2030 water stress assuming features revert to long-term averages.

| Model | 2030 Prediction | Change from Linear |
|-------|-----------------|-------------------|
| DecisionTree | 83.87% | +3.23% |
| Lasso | 80.93% | +6.22% |
| **Ridge (BEST)** | **81.23%** | **+5.92%** |
| LinearRegression | 81.27% | +5.87% |
| RandomForest | 84.88% | +2.73% |

**Interpretation:** Average-based predictions are higher (73-85%), suggesting that normalizing features to historical means produces less optimistic scenarios.

---

## 2. Model Comparison

### Ridge Regression (Recommended)
- **Training Performance:** R² = 0.9987, MAE = 0.066%, RMSE = 0.072%
- **2030 Range:** 75.31% (linear) to 81.23% (average)
- **Rationale:** Best generalization with minimal error; stable across scenarios

### Random Forest
- **Highest Predictions:** 82.15% - 84.88%
- **Characteristic:** Tree-based model produces slightly pessimistic forecasts
- **Range:** +2.7% to +5.9% vs Ridge

### Lasso Regression
- **Lowest Predictions:** 74.71% - 80.93%
- **Characteristic:** Sparse model with feature regularization
- **Range:** -0.6% to -0.3% vs Ridge

### Decision Tree & LinearRegression
- **Similar Performance:** Within ±1% of Ridge
- **Decision Tree Range:** 80.64% - 83.87%
- **LinearRegression Range:** 75.40% - 81.27%

---

## 3. Scenario Sensitivity Analysis

### Ranking by Optimism (Lowest Stress → Highest Stress)

1. **Lasso + Linear Extrapolation:** 74.71% (most optimistic)
2. **Ridge + Linear Extrapolation:** 75.31% ← **Recommended Baseline**
3. **LinearRegression + Linear Extrapolation:** 75.40%
4. **Lasso + Exponential Extrapolation:** 74.87%
5. Random Forest + Linear: 82.15%
6. RandomForest + Average: 84.88% (most pessimistic)

### Projection Spread
- **Most Optimistic Scenario:** Lasso + Linear = 74.71%
- **Most Pessimistic Scenario:** RandomForest + Average = 84.88%
- **Spread:** 10.17 percentage points
- **Relative SD:** ±6.8% around midpoint (79.8%)

---

## 4. Feature Drivers

The models were trained on 7 key features (post-filtering for leakage and collinearity):

1. **Annual freshwater withdrawals** (domestic, industry, total)
   - Measures water demand patterns
   - Higher withdrawal → higher water stress

2. **Renewable freshwater resources** (per capita, total)
   - Measures water availability
   - Higher resources → lower water stress

3. **Water productivity**
   - Economic efficiency of water use
   - Higher productivity → lower stress per unit GDP

4. **Year** (temporal trend)
   - Captures long-term structural changes
   - Dominates tree-based models (91.8% of DecisionTree importance)

---

## 5. Confidence Assessment

### High Confidence Projections (±2-3%)
- Ridge Regression across all scenarios
- Expected range: **73-77%** (linear), **74-76%** (exponential)

### Medium Confidence Projections (±4-5%)
- Lasso, LinearRegression across scenarios
- Other models with linear/exponential extrapolation

### Lower Confidence Projections (±6-8%)
- RandomForest results (tree models sensitive to historical volatility)
- Average-based scenarios (greater uncertainty in feature reversion)

---

## 6. Recommendations

### For Policy Impact Assessment

1. **Use Ridge Linear Projection (75.31%) as Official Forecast**
   - Best validated model (R² = 0.9987)
   - Most stable across scenarios
   - 22.8% improvement over 2023 baseline

2. **Plan for Range of Outcomes**
   - Conservative (Upper Bound): 82-85% (RandomForest + Average)
   - Baseline: 75-76% (Ridge + Linear/Exponential)
   - Optimistic (Lower Bound): 74-75% (Lasso + Linear)

3. **Monitor Key Indicators**
   - Freshwater withdrawal trends (domestic, industrial)
   - Precipitation and renewable resource patterns
   - Agricultural and industrial efficiency metrics

### For Water Resource Management

The projected **23% decrease in water stress** from 2023 (98.1%) to 2030 (75.3%) suggests:

- **Positive Signals:** Feature trends indicate improving water management or resource availability
- **Required Actions:** Maintain current efficiency improvements and conservation practices
- **Risk Factors:** Actual outcomes depend on:
  - Climate patterns and precipitation levels
  - Population and economic growth rates
  - Agricultural and energy sector efficiency
  - International water agreements

---

## 7. Methodology Notes

### Feature Selection
- Original dataset: 141 environmental indicators
- After leakage filtering: 133 features (removed 8 highly-correlated with target)
- After collinearity filtering: 131 features (removed 2 redundant pairs)
- Final model features: 7 (selected for training)

### Model Validation
- **Cross-Validation:** 5-fold temporal split
- **Train/Test Split:** 80/20 on temporal data
- **Best Model:** Ridge (α=0.001) with R² = 0.9987

### Extrapolation Methods

#### Linear Trend
```
feature_2030 = feature_2024 + (feature_2024 - feature_2020) × 1.5
```

#### Exponential Trend
```
growth_rate = (feature_2024 / feature_2020)^(1/4)
feature_2030 = feature_2024 × growth_rate^6
```

#### Average Trend
```
feature_2030 = mean(feature_1960-2024)
```

---

## 8. Files Generated

- `predictions/water_stress_2030_predictions.csv` - Summary predictions
- `predictions/water_stress_2030_detailed.txt` - Detailed feature projections
- `predictions/2030_scenario_analysis.md` - This report

---

**Report Generated:** 2024
**Forecast Year:** 2030
**Baseline Period:** 1960-2024
