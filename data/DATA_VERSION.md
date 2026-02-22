# Data Versioning and Integrity

This document tracks all data assets used in the Tunisia Water Stress ML project and their versions for reproducibility.

## Data Files

```
data/
├── raw/
│   └── environment_tun.csv              # Raw World Bank water/environment indicators
├── processed/
│   └── processed_tunisia.csv            # Processed wide-form dataset
└── DATA_MANIFEST.json                   # Hash manifest for integrity verification
```

## Raw Data: environment_tun.csv

**Description:** World Bank water and environment indicators for Tunisia (1972-2020)

**Source:**
- World Bank API (https://data.worldbank.org)
- Indicators: Water stress, renewable energy, agricultural land, population, precipitation, etc.

**Date Tracked:** 2024-2-22
**Hash (SHA256):** To be computed on first run
**Size:** ~50KB
**Records:** 49 years × multiple indicators

**Last Updated:** 2024-02-22
**Update Procedure:**
1. Manually downloaded from World Bank API or automated ETL
2. Saved to `data/raw/environment_tun.csv`
3. Hash recorded for integrity tracking

## Processed Data: processed_tunisia.csv

**Description:** Cleaned and wide-form version of raw data

**Transformation Pipeline:**
1. Load raw environment data from World Bank
2. Pivot from long-form to wide-form (years as rows, indicators as columns)
3. Handle missing values (interpolation/forward-fill)
4. Drop sparse columns (>50% missing)
5. Select relevant indicators via keyword matching

**Date Tracked:** 2024-02-22
**Hash (SHA256):** To be computed on first run
**Size:** ~30KB
**Records:** 49 years × selected indicators

**Code Location:** [src/data_loader.py](../src/data_loader.py#L1)

**Last Updated:** 2024-02-22

## Data Integrity Verification

To verify data hasn't changed since last experiment:

```bash
# Initialize/update Data Manifest
python -c "
from src.data_versioning import DataManifest
manifest = DataManifest()
manifest.add_file('data/raw/environment_tun.csv', source='World Bank')
manifest.add_file('data/processed/processed_tunisia.csv', source='Pipeline')
manifest.save('data/DATA_MANIFEST.json')
"

# Verify integrity
python -c "
from src.data_versioning import DataManifest
manifest = DataManifest.load('data/DATA_MANIFEST.json')
results = manifest.verify_all()
for path, is_valid in results.items():
    status = '✓' if is_valid else '✗ CHANGED'
    print(f'{status} {path}')
"
```

## Reproducibility Guarantees

- **Data versioning:** SHA256 hashes ensure exact data lineage
- **Configuration tracking:** All preprocessing parameters in [config/train_config.yaml](../config/train_config.yaml)
- **Experiment tracking:** MLflow logs complete run data in [mlruns/](../mlruns)
- **Seed management:** All random seeds documented in [src/config_train.py](../src/config_train.py)

## Version History

| Date       | Data Source | Raw Hash | Processed Hash | Changes |
|------------|---|---|---|---|
| 2024-02-22 | World Bank | TBD | TBD | Initial version |

## Using Data Versions in Experiments

```yaml
# config/experiment_v1.yaml
data:
  raw_data_path: "data/raw/environment_tun.csv"
  processed_data_path: "data/processed/processed_tunisia.csv"

# Track in MLflow
# The config file is automatically logged to MLflow artifacts
```

## Related Documentation

- [Training Configuration](../config/README.md)
- [MLflow Experiment Tracking](#experiment-tracking)
- [Reproducibility Guide](../docs/REPRODUCIBILITY.md)
