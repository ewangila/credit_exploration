# Credit Exploration – PySpark EDA & Feature Binning

Exploratory data analysis and feature binning on the UCI Credit Card Default dataset using PySpark, pandas, and matplotlib.

## What it does

| Section | Description |
|---------|-------------|
| **Initial exploration** | Load CSV into a Spark DataFrame, inspect schema, show distinct values for `SEX`, `EDUCATION`, and `MARRIAGE` |
| **Feature binning** | Collapse rare / unknown education (0, 5, 6) and marriage (0) codes into an `"Other"` category |
| **Class balance** | Visualize the target (`default`) distribution and default rate broken down by sex |
| **Plots** | Bar charts saved under `presentations/` (raw vs binned categories, class balance, default-by-sex) |

## Files

| Path | Description |
|------|-------------|
| `credit_exploration.py` | Main analysis script |
| `presentations/` | Generated PNG plots |
| `.gitignore` | Ignores CSV data, `__pycache__`, notebook checkpoints |

## Requirements

- Python 3.8+
- Java 8 or higher (required by Spark)
- Packages in `requirements.txt`

```bash
pip install -r requirements.txt
