# Collinearity & Leakage Filtering Report

**Processed file:** data\processed\processed_tunisia.csv

**Auto-detected target:** Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)

**Selected features (initial):** 11

**Leakage threshold:** 0.99 — features dropped: 8

- Annual freshwater withdrawals, domestic (% of total freshwater withdrawal)
- Annual freshwater withdrawals, industry (% of total freshwater withdrawal)
- Annual freshwater withdrawals, total (% of internal resources)
- Annual freshwater withdrawals, total (billion cubic meters)
- Level of water stress: freshwater withdrawal as a proportion of available freshwater resources
- Renewable internal freshwater resources per capita (cubic meters)
- Renewable internal freshwater resources, total (billion cubic meters)
- Water productivity, total (constant 2015 US$ GDP per cubic meter of total freshwater withdrawal)

**Collinearity threshold:** 0.95 — features flagged to drop: 0

## Metrics

Metrics before (leakage-filtered):

           model      MAE     RMSE        R2
LinearRegression 3.320204 3.612159 -2.261659
    DecisionTree 2.893456 3.517444 -2.092853
    RandomForest 2.990279 3.597512 -2.235261

Metrics after (collinearity-filtered):

           model      MAE     RMSE        R2
LinearRegression 3.320204 3.612159 -2.261659
    DecisionTree 2.893456 3.517444 -2.092853
    RandomForest 2.990279 3.597512 -2.235261