# Meridian Downstream Energy: Operations and Risk Predictive Support System

**A Real Time Predictive Analytics to Optimize Refining Yields, Prevent Downtime, and Secure Operational Margins**

**Python | pandas | scikit-learn | Streamlit** | Five predictive models for a fictional downstream oil and gas company

**Author:** Adewale Oshinukan | Data Scientist | [GitHub](https://github.com/waleoshin75-sketch) | [Live App](https://meridiandownstreamenergyoperationsandriskpredictivesupportsys.streamlit.app/)

[Live Streamlit App](https://meridiandownstreamenergyoperationsandriskpredictivesupportsys.streamlit.app/) | [GitHub Repository](https://github.com/waleoshin75-sketch/Meridian-Downstream-Energy-Operations-and-Risk-Predictive-Support-System)

---

## At a glance

I took eight messy operational tables from a fictional downstream oil and gas company (refineries, terminals, assets, sensors, maintenance, production, fuel deliveries, quality tests and tank inventory), cleaned them, joined them in a star schema and built one model for each business problem. The saved models run inside a Streamlit app: enter an asset, run, trip, batch or tank reading and get a risk score and a suggested action.

| Business objective | Model that led on unseen data | Result on the test period |
|---|---|---|
| Cut unplanned downtime and maintenance cost | Logistic Regression | Servicing the riskiest 10% of asset-months catches **38%** of breakdowns (3.8x better than random). ROC AUC 0.753 |
| Maximise refinery yield and gross margin | Ridge Regression | Margin forecast off by **0.86 USD/bbl** on average, R² 0.85. Last week's margin is off by 1.77 |
| Raise on-time fuel delivery and cut distribution cost | Random Forest | Flagging the riskiest 20% of trips catches **62%** of late trips. ROC AUC 0.789 |
| Reduce off-spec product and rework cost | Logistic Regression | Checking the riskiest 20% of batches catches **48%** of off-spec batches. ROC AUC 0.733 |
| Detect product loss, theft and metering errors | Isolation Forest, plus a simple rule | Isolation Forest PR AUC 0.401. A plain "gap over 1%" rule scores **0.987** and beats all three models |

**What it is worth:** about **$4.2M a year net** at stated assumptions (sampling range $3.0M to $5.7M), 90% of it from maintenance. These are indicative figures built on assumed unit costs, not company records.

The data is synthetic and was built with realistic quality problems. Two of the five models (breakdowns and off-spec batches) are ranking tools with wide uncertainty, and for inventory the simple rule beat machine learning. I report all of that as it is.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Dictionary](#3-data-dictionary)
4. [Data Quality Audit](#4-data-quality-audit)
5. [Data Cleaning and Error Checking](#5-data-cleaning-and-error-checking)
6. [Feature Engineering](#6-feature-engineering)
7. [Create Relationship Table](#7-create-relationship-table)
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)
9. [Analytical Base Table](#9-analytical-base-table)
10. [Encode Categorical Columns](#10-encode-categorical-columns)
11. [Separate X and Y](#11-separate-x-and-y)
12. [Train Test Split](#12-train-test-split)
13. [Feature Scaling of Numerical Columns](#13-feature-scaling-of-numerical-columns)
14. [Train Model](#14-train-model)
15. [Predict X_test](#15-predict-x_test)
16. [Compare X and Y](#16-compare-x-and-y)
17. [Evaluate Model](#17-evaluate-model)
18. [Business Analysis](#18-business-analysis)
19. [Save Model, Scaler and Features](#19-save-model-scaler-and-features)
20. [Download Model, Scaler and Features](#20-download-model-scaler-and-features)
21. [Key Findings and Insights](#21-key-findings-and-insights)
22. [Recommendations](#22-recommendations)
23. [Impact](#23-impact)
24. [Conclusion](#24-conclusion)

---

## 1. Project Overview

Meridian Downstream Energy is a fictional multinational refiner and fuel distributor. Its data sits in eight separate, untidy tables, and five business goals depend on it:

| # | Business objective | ML task | Target | Owner |
|---|---|---|---|---|
| 1 | Cut unplanned downtime and maintenance cost | Classification | Asset breaks down in the next 30 days (4.6% of asset-months) | Asset Integrity and Maintenance |
| 2 | Maximise refinery yield and gross margin | Regression | Gross margin in USD per barrel (average 11.51) | Refining and Planning |
| 3 | Raise on-time fuel delivery and cut distribution cost | Classification | Trip arrives late (10.5% of trips) | Supply and Distribution |
| 4 | Reduce off-spec product and rework cost | Classification | Batch fails specification (7.2% of batches) | Quality and Blending |
| 5 | Detect product loss, theft and metering errors | Anomaly detection | None. An answer key is used only to grade the result | Terminal Operations and Security |

**How I built it, in order:** data quality audit, cleaning, feature engineering, a star schema, EDA, then one analytical base table per objective, encoding, X/y split, a date-based train/test split, scaling, four candidate models per objective (three detectors for inventory), prediction, evaluation, business analysis, saving the model, scaler and feature list with joblib, and a Streamlit app.

**Two rules I held to throughout**
- **No peeking at the future.** Every split is by date: earlier rows train, later rows test. Imputation and scaling are fitted on training rows only.
- **No leakage.** Columns that contain or reveal the answer are left out on purpose (listed in section 9).

### Repository structure

```
├── README.md
├── app.py                          the Streamlit app (section 20)
├── requirements.txt
├── images/                         every chart shown in this README
├── Meridian_Downstream_Consolidated.xlsx   the raw workbook (8 data sheets + plan sheets)
├── model/                          the saved models, scalers and feature lists (about 2.7 MB)
└── notebook                        all the code shown below
```

---

## 2. Dataset Overview

Eight tables arrive dirty. After cleaning, **35,139 rows** remain and every fact row finds its parent.

| Table | Role | Rows (raw) | Rows (clean) | What it holds |
|---|---|---|---|---|
| `dim_sites` | Dimension | 27 | 24 | Refineries, terminals and depots: country, type, capacity |
| `dim_assets` | Dimension | 366 | 356 | Pumps, compressors, furnaces and more: maker, install date, design life, criticality |
| `fact_sensor_readings` | Fact | 8,779 | 8,477 | Monthly vibration, temperature, pressure, operating hours, lubricant condition |
| `fact_maintenance` | Fact | 1,736 | 1,658 | Work orders: breakdown or preventive, downtime, cost, failure mode |
| `fact_production` | Fact | 4,457 | 4,349 | Daily refinery runs: crude grade, throughput, temperatures, yields, margin |
| `fact_deliveries` | Fact | 6,120 | 5,941 | Fuel trips: product, distance, traffic, weather, driver, promised and actual arrival |
| `fact_quality_tests` | Fact | 4,271 | 4,152 | Blend batches: water, additive dose, sulphur, octane, spec result, rework cost |
| `fact_inventory` | Fact | 10,412 | 10,182 | Daily tank stock: opening, receipts, dispatches, book vs measured closing |

The workbook also holds plan sheets (business objectives, ML objectives, leakage rules, data splitting, dirty-data summary) that guided the build.

### Code: load the eight tables

```python
import pandas as pd

file_path = "Meridian_Downstream_Consolidated.xlsx"

tables = ["dim_sites", "dim_assets", "fact_sensor_readings", "fact_maintenance",
          "fact_production", "fact_deliveries", "fact_quality_tests", "fact_inventory"]

raw = {t: pd.read_excel(file_path, sheet_name=t) for t in tables}

for t, df in raw.items():
    print(t, df.shape)
    display(df.head(3))
```

---

## 3. Data Dictionary

The workbook's `Data_Dictionary` sheet documents all 99 columns. Here it is condensed to one row per table.

| Table | Columns | What it holds | Known problems planted in the data |
|---|---|---|---|
| `dim_sites` (8) | `site_id`, `site_name`, `site_type`, `country`, `region`, `capacity_bpd`, `storage_capacity_kl`, `commissioned_year` | One row per site | Duplicate rows and IDs, country variants (UK / U.K.), capacity stored as text like "120,000 bpd", missing values |
| `dim_assets` (7) | `asset_id`, `site_id`, `asset_type`, `manufacturer`, `install_date`, `design_life_years`, `criticality` | One row per piece of equipment | Duplicates, orphan site codes, five date formats, "25 yrs" text, type variants (HX, Compr.) |
| `fact_sensor_readings` (8) | `reading_id`, `asset_id`, `reading_month`, `avg_vibration_mm_s`, `avg_temperature_c`, `avg_pressure_bar`, `operating_hours`, `lubricant_condition_score` | One row per asset per month | About 3% of vibration in in/s, sentinels (-999, 9999), orphan asset codes, junk dates |
| `fact_maintenance` (10) | `work_order_id`, `asset_id`, `work_type`, `opened_date`, `closed_date`, `downtime_hours`, `cost_usd`, `failure_mode`, `technician_hours`, `status` | One row per work order | Closed date before opened date, negative and 9999 downtime, work-type variants (Brkdwn, PM, CM) |
| `fact_production` (18) | `run_id`, `site_id`, `run_date`, `crude_grade`, `crude_api_gravity`, `crude_sulphur_pct`, `throughput_bbl`, `crude_cost_usd_per_bbl`, `furnace_outlet_temp_c`, `column_pressure_bar`, `energy_use_gj_per_bbl`, `unplanned_downtime_hours`, `crack_spread_usd_per_bbl`, four `*_yield_pct` columns, `gross_margin_usd_per_bbl` | One row per refinery run-day | About 2% of furnace temperatures in Fahrenheit, sentinels, duplicates, mixed dates |
| `fact_deliveries` (17) | `trip_id`, `origin_site_id`, `destination_station_id`, `destination_region`, `product`, `volume_litres`, `distance_km`, `traffic_index`, `weather`, `driver_experience_years`, `truck_age_years`, `dispatch_time`, `promised_arrival`, `actual_arrival`, `freight_cost_usd`, `trip_status`, `unit_label_distance` | One row per fuel trip | About 4% of distances in miles, blank arrival for cancelled trips, impossible driver experience |
| `fact_quality_tests` (18) | `test_id`, `batch_id`, `site_id`, `product`, `test_date`, `crude_grade`, `feed_crude_sulphur_pct`, `additive_dose_ppm`, `blend_temp_c`, `storage_days`, `water_content_ppm`, `sulphur_ppm`, `density_kg_m3`, `octane_rating`, `flash_point_c`, `lab_name`, `spec_result`, `rework_cost_usd` | One row per blend batch test | Censored lab text (<10, ND, BDL), spec labels in many spellings, sentinels, duplicates |
| `fact_inventory` (13) | `record_id`, `site_id`, `tank_id`, `product`, `record_date`, `opening_stock_kl`, `receipts_kl`, `dispatches_kl`, `book_closing_kl`, `measured_closing_kl`, `tank_temp_c`, `days_since_meter_calibration`, `unit` | One row per tank per day | About 6% recorded in barrels instead of kilolitres, 10x data-entry errors, duplicates |

**Leakage rules** (from the `Leakage_Rules` sheet). These columns exist on purpose and are kept out of the models:

| Objective | Never use as features | Why |
|---|---|---|
| 1. Maintenance | closed_date, downtime_hours, cost_usd, technician_hours, failure_mode | Only known after the breakdown |
| 2. Margin | the four `*_yield_pct` columns | Margin is built from the yields, so they are outcomes of the same run |
| 3. Delivery | actual_arrival, trip_status, freight_cost_usd | Contain or reveal the answer (freight cost includes the late penalty) |
| 4. Off-spec | sulphur_ppm, density_kg_m3, octane_rating, flash_point_c, spec_result, rework_cost_usd | Lab results decide the label; rework only exists after a fail |
| 5. Inventory | the answer key | Used only to judge the result, never to train |

---

## 4. Data Quality Audit

Before touching anything I profiled every table. The audit found problems in all eight.

| Check | What it found |
|---|---|
| Exact duplicate rows | 3 to 164 per table, 571 in total |
| Duplicate IDs | More than the exact duplicates in production and quality tables (71 each), so some IDs carry conflicting values |
| Missing values | `capacity_bpd` is 78% blank. `octane_rating` 62%, `flash_point_c` 38% and `failure_mode` 54% are blank because they only apply to some rows |
| Broken keys | Orphan, mis-cased or padded codes (` AST-0088`, `sit-001`, `UNKNOWN`, `SIT-999`): 307 sensor rows, 57 trips and more |
| Mixed units | Miles in the distance column, barrels in the inventory column, Fahrenheit in furnace temperature, in/s in vibration |
| Sentinels and impossible values | -999, 9999, 99 and negatives in temperature, pressure, downtime, water, driver experience |
| Date logic | 68 maintenance rows closed before they were opened. Five date formats plus junk text ("TBD", "00/00/0000") |
| Censored lab results | 310 sulphur readings written as `<10`, `ND` or `BDL` |

<p align="center">
  <img src="images/eda_0_missing.png" alt="Missing values by table, before cleaning" width="100%">
</p>
<p align="center"><em>Missing values by table, before cleaning</em></p>

### Code: data quality audit

```python
# STEP 1: DATA QUALITY AUDIT (read-only: nothing is changed or saved)
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_rows", 200)

FILE = "Meridian_Downstream_Consolidated.xlsx"      # change the path if the file is elsewhere
TABLES = ["dim_sites", "dim_assets", "fact_sensor_readings", "fact_maintenance",
          "fact_production", "fact_deliveries", "fact_quality_tests", "fact_inventory"]
ID_COL = {"dim_sites": "site_id", "dim_assets": "asset_id", "fact_sensor_readings": "reading_id",
          "fact_maintenance": "work_order_id", "fact_production": "run_id", "fact_deliveries": "trip_id",
          "fact_quality_tests": "test_id", "fact_inventory": "record_id"}
FOREIGN = {"dim_assets": [("site_id", "dim_sites")], "fact_sensor_readings": [("asset_id", "dim_assets")],
           "fact_maintenance": [("asset_id", "dim_assets")], "fact_production": [("site_id", "dim_sites")],
           "fact_deliveries": [("origin_site_id", "dim_sites")], "fact_quality_tests": [("site_id", "dim_sites")],
           "fact_inventory": [("site_id", "dim_sites")]}
SENTINELS = [-999, -99, -5, -1, 999, 9999]

df = {t: pd.read_excel(FILE, sheet_name=t) for t in TABLES}


def text_cols(d):
    """Columns that hold text (works on old and new pandas)."""
    return [c for c in d.columns if d[c].dtype == object or pd.api.types.is_string_dtype(d[c])]


def as_number(s):
    """Strip $, USD, commas, 'bpd', 'yrs' and convert to number (text that is not numeric becomes NaN)."""
    return pd.to_numeric(s.astype(str).str.replace(r"[$,]|USD|bpd|yrs|\s", "", regex=True), errors="coerce")


# 1. Overview ---------------------------------------------------------------------------------
rows = []
for t, d in df.items():
    rows.append({"table": t, "rows": len(d), "columns": d.shape[1],
                 "missing_cells_%": round(d.isna().sum().sum() / d.size * 100, 2),
                 "exact_duplicate_rows": int(d.duplicated().sum()),
                 "duplicate_ids": int(d[ID_COL[t]].astype(str).str.strip().str.upper().duplicated().sum())})
print("1. OVERVIEW\n", pd.DataFrame(rows).to_string(index=False))

# 2. Missing values -----------------------------------------------------------------------------
print("\n2. MISSING VALUES (columns with at least 1 missing)")
for t, d in df.items():
    m = d.isna().sum()
    m = m[m > 0]
    if len(m):
        print(f"\n{t}\n", pd.DataFrame({"missing": m, "%": (m / len(d) * 100).round(1)}).to_string())

# 3. Key integrity (orphan and malformed foreign keys) -----------------------------------------
print("\n3. FOREIGN KEY CHECK")
out = []
for t, links in FOREIGN.items():
    for col, parent in links:
        valid = set(df[parent][ID_COL[parent]].astype(str))
        bad = ~df[t][col].astype(str).isin(valid)
        out.append({"table": t, "column": col, "parent": parent, "rows_not_matching": int(bad.sum()),
                    "examples": list(df[t].loc[bad, col].astype(str).unique()[:4])})
print(pd.DataFrame(out).to_string(index=False))

# 4. Numbers stored as text -------------------------------------------------------------------
print("\n4. NUMBERS STORED AS TEXT")
out = []
for t, d in df.items():
    for c in text_cols(d):
        s = d[c].dropna()
        if len(s) == 0 or c == ID_COL[t]:
            continue
        num = as_number(s)
        if num.notna().mean() > 0.5:                          # mostly numeric, but stored as text
            pure = pd.to_numeric(s, errors="coerce").notna().mean()
            out.append({"table": t, "column": c, "text_cells": int(len(s) - pure * len(s)),
                        "cannot_convert": int(num.isna().sum()), "examples": list(s[pd.to_numeric(s, errors="coerce").isna()].astype(str).head(3))})
print(pd.DataFrame(out).to_string(index=False))

# 5. Dates: how many do not follow YYYY-MM-DD ---------------------------------------------------
print("\n5. DATE COLUMNS")
out = []
for t, d in df.items():
    for c in d.columns:
        if re.search(r"date|time|month|arrival", c) and c in text_cols(d):
            s = d[c].dropna().astype(str)
            iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
            parsed = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
            out.append({"table": t, "column": c, "non_ISO_format": int((~iso).sum()),
                        "unparseable_junk": int(parsed.isna().sum()),
                        "junk_examples": list(s[parsed.isna()].unique()[:3])})
print(pd.DataFrame(out).to_string(index=False))

# 6. Category variants: same thing spelled many ways ----------------------------------------------
print("\n6. CATEGORY VARIANTS (raw unique values vs unique after trimming and lower-casing)")
out = []
for t, d in df.items():
    for c in text_cols(d):
        if c == ID_COL[t] or d[c].nunique() > 80:
            continue
        raw, clean = d[c].dropna().nunique(), d[c].dropna().astype(str).str.strip().str.lower().nunique()
        if as_number(d[c].dropna()).notna().mean() < 0.5:     # skip numeric-looking text
            out.append({"table": t, "column": c, "raw_unique": raw, "after_trim_lower": clean})
print(pd.DataFrame(out).to_string(index=False))
print("\nValues in the columns that matter most:")
for t, c in [("dim_sites", "country"), ("fact_deliveries", "product"), ("fact_quality_tests", "spec_result"),
             ("fact_maintenance", "work_type")]:
    print(f"\n{t}.{c}:", sorted(df[t][c].dropna().astype(str).unique().tolist()))

# 7. Sentinels and impossible numbers ------------------------------------------------------------
print("\n7. SENTINEL AND IMPOSSIBLE VALUES")
out = []
for t, d in df.items():
    for c in d.columns:
        if c == ID_COL[t] or "id" in c.split("_") or c in ("commissioned_year", "days_since_meter_calibration"):
            continue
        n = as_number(d[c]) if c in text_cols(d) else d[c]
        if not pd.api.types.is_numeric_dtype(n) or n.notna().mean() < 0.5:
            continue
        sent = int(n.isin(SENTINELS).sum())
        neg = int((n < 0).sum())
        if sent or neg:
            out.append({"table": t, "column": c, "sentinel_hits": sent, "negative_values": neg,
                        "min": round(n.min(), 2), "max": round(n.max(), 2)})
print(pd.DataFrame(out).to_string(index=False))

# 8. Unit mixing and logic checks ------------------------------------------------------------------
print("\n8. UNIT AND LOGIC CHECKS")
print("fact_deliveries unit_label_distance:\n", df["fact_deliveries"]["unit_label_distance"].str.strip().str.lower().value_counts().to_string())
print("\nfact_inventory unit:\n", df["fact_inventory"]["unit"].str.strip().str.lower().value_counts().to_string())
mt = df["fact_maintenance"]
op, cl = pd.to_datetime(mt["opened_date"], errors="coerce", format="mixed"), pd.to_datetime(mt["closed_date"], errors="coerce", format="mixed")
print("\nmaintenance rows where closed_date is before opened_date:", int((cl < op).sum()))
ft = as_number(df["fact_production"]["furnace_outlet_temp_c"])
print("production furnace temp above 500 (probably Fahrenheit):", int((ft > 500).sum()))
cens = df["fact_quality_tests"]["sulphur_ppm"].astype(str).str.contains(r"<|ND|BDL", regex=True).sum()
print("deliveries with driver experience above 40 years (impossible):", int((df["fact_deliveries"]["driver_experience_years"] > 40).sum()))
print("production column pressure above 10 bar (sentinel 99):", int((df["fact_production"]["column_pressure_bar"] > 10).sum()))
print("quality tests with censored sulphur result (<10, ND, BDL):", int(cens))

# 9. Scorecard ------------------------------------------------------------------------------------
print("\n9. AUDIT SUMMARY: fix these in the cleaning step")
print("""- remove exact duplicates, then resolve duplicate IDs (keep one row per ID)
- repair or drop orphan and malformed foreign keys (trim, upper-case, match to parent)
- convert text numbers to numeric; standardise dates to one format; junk dates become missing
- map category variants to one standard label
- turn sentinels and impossible values into missing, then impute
- convert miles to km, barrels to kl, Fahrenheit to Celsius, in/s to mm/s
- fix closed_date before opened_date; treat censored lab results (<10, ND) as half the limit""")
```

---

## 5. Data Cleaning and Error Checking

Each fix follows the audit. Every change is counted in a cleaning log, so nothing happens silently.

| Fix | Example |
|---|---|
| Remove exact duplicates, then one row per ID | 571 exact duplicate rows removed; 1,029 rows removed in total once orphans and unplaceable dates are included |
| Repair keys, drop true orphans | 269 sensor asset codes repaired by trimming and upper-casing; 133 rows with no real parent dropped |
| Standardise categories | 1,154 failure-mode and 582 work-type variants mapped to one label each |
| Convert units | 308 vibration values in/s to mm/s; miles to km; barrels to kl; Fahrenheit to Celsius |
| Sentinels and impossible values to missing | 35 temperatures, 20 pressures and similar |
| Fix dates | Five formats to one. Junk dates become missing. 24 closed-before-opened dates set to missing |
| Censored lab results | `<10`, `ND`, `BDL` treated as half the detection limit |
| Missing values | Left in place and filled after the train/test split, so test data never influences the fill |

**Before and after**

| Table | Rows before | Rows after | Removed |
|---|---|---|---|
| `dim_sites` | 27 | 24 | 3 |
| `dim_assets` | 366 | 356 | 10 |
| `fact_sensor_readings` | 8,779 | 8,477 | 302 |
| `fact_maintenance` | 1,736 | 1,658 | 78 |
| `fact_production` | 4,457 | 4,349 | 108 |
| `fact_deliveries` | 6,120 | 5,941 | 179 |
| `fact_quality_tests` | 4,271 | 4,152 | 119 |
| `fact_inventory` | 10,412 | 10,182 | 230 |

Duplicate IDs after cleaning: **0** in every table.

### Code: data cleaning and error checking

```python
# STEP 2: DATA CLEANING
# Reads the dirty workbook, fixes every problem found in the audit, and saves the clean tables.
# Remaining gaps in numeric columns are left blank on purpose: they are filled AFTER the train/test split
# (using training data only) so no information leaks from the test set.
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
FILE = "Meridian_Downstream_Consolidated.xlsx"       # same file you uploaded in Colab
TABLES = ["dim_sites", "dim_assets", "fact_sensor_readings", "fact_maintenance",
          "fact_production", "fact_deliveries", "fact_quality_tests", "fact_inventory"]
raw = {t: pd.read_excel(FILE, sheet_name=t) for t in TABLES}
clean, LOG = {}, []


def log(table, action, n):
    if n:
        LOG.append({"table": table, "action": action, "rows_or_cells": int(n)})


# ---------------------------------------------------------------- helpers
def num(s):
    """Text such as '$1,250.00', 'USD 99', '120,000 bpd', '25 yrs' -> number."""
    return pd.to_numeric(s.astype(str).str.replace(r"[$,]|USD|bpd|yrs|\s", "", regex=True), errors="coerce")


DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%B %d, %Y", "%m-%d-%Y"]
DATE_FORMATS += [f + t for f in DATE_FORMATS for t in (" %H:%M", " %H:%M:%S")]


def parse_dates(s, table):
    """Five different date styles -> one datetime column. Junk such as TBD becomes missing."""
    text = s.astype(str).str.strip().where(s.notna())
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        out = out.fillna(pd.to_datetime(text, format=fmt, errors="coerce"))
    log(table, f"{s.name}: junk dates (TBD, unknown, 00/00/0000) set to missing", text.notna().sum() - out.notna().sum())
    return out


def std_cat(s, groups, table, fill="Unknown"):
    """Map every spelling variant to one standard label. groups = {standard: [variants]}."""
    lookup = {v.lower(): k for k, vs in groups.items() for v in [k] + vs}
    key = s.astype(str).str.strip().str.lower().where(s.notna())
    out = key.map(lookup)
    unmapped = key[out.isna() & key.notna()].unique()
    if len(unmapped):
        print(f"  WARNING {table}.{s.name}: unmapped values {list(unmapped)[:6]}")
    log(table, f"{s.name}: spelling/case variants standardised", (s.astype(str).str.strip() != out).sum())
    return out.fillna(fill)


def fix_ids(s):
    return s.astype(str).str.strip().str.upper().where(s.notna())


def fix_fk(d, col, parent_ids, table):
    """Repair case and spacing, then drop rows whose key still has no parent (UNKNOWN, SIT-999 ...)."""
    before = d[col].copy()
    d[col] = fix_ids(d[col])
    log(table, f"{col}: case/space repaired so key matches parent", ((before != d[col]) & d[col].isin(parent_ids)).sum())
    bad = ~d[col].isin(parent_ids)
    log(table, f"{col}: rows dropped, key has no parent", bad.sum())
    return d[~bad].copy()


def limit(d, col, lo, hi, table):
    """Values outside the physically possible range (sentinels such as -999 or 9999) become missing."""
    d[col] = num(d[col])
    bad = d[col].notna() & ~d[col].between(lo, hi)
    d.loc[bad, col] = np.nan
    log(table, f"{col}: impossible values/sentinels set to missing (valid {lo} to {hi})", bad.sum())


def dedupe(d, id_col, table):
    n0 = len(d)
    d = d.drop_duplicates()
    log(table, "exact duplicate rows removed", n0 - len(d))
    n1 = len(d)
    d = d.assign(_m=d.isna().sum(axis=1)).sort_values("_m", kind="stable").drop_duplicates(id_col).drop(columns="_m")
    log(table, f"{id_col}: repeated IDs with conflicting values removed (kept the more complete row)", n1 - len(d))
    return d.sort_values(id_col).reset_index(drop=True)


def drop_missing(d, col, table):
    bad = d[col].isna()
    log(table, f"rows dropped, {col} missing (cannot be placed in time)", bad.sum())
    return d[~bad].copy()


# ---------------------------------------------------------------- standard labels
COUNTRY = {"Nigeria": ["ng"], "Ghana": ["gh"], "United Kingdom": ["uk", "u.k.", "great britain"],
           "United States": ["us", "usa", "u.s.a"], "Netherlands": ["holland", "nl", "the netherlands"],
           "UAE": ["u.a.e", "united arab emirates"], "South Africa": ["rsa", "s. africa"],
           "Singapore": ["sg"], "India": ["in"], "Brazil": ["brasil"]}
REGION_OF = {"Nigeria": "West Africa", "Ghana": "West Africa", "Netherlands": "Europe", "United Kingdom": "Europe",
             "Singapore": "Asia Pacific", "UAE": "Middle East", "South Africa": "Southern Africa",
             "United States": "North America", "India": "South Asia", "Brazil": "South America"}
CRUDE = {"Bonny Light": ["bonny-light", "bonny lt"], "Arab Light": ["arab lt", "arabian light"], "Brent": ["brent blend"],
         "WTI": ["west texas intermediate"], "Forcados": [], "Azeri": [], "Basrah Medium": []}
PRODUCT = {"PMS": ["petrol", "premium motor spirit", "gasoline"], "AGO": ["diesel", "automotive gas oil"],
           "DPK": ["kerosene"], "Jet A1": ["jet a-1", "jet fuel"], "LPG": ["cooking gas"]}

# ================================================================ dim_sites
d = raw["dim_sites"].copy()
d["site_id"] = fix_ids(d["site_id"])
d["country"] = std_cat(d["country"], COUNTRY, "dim_sites")
d["site_type"] = std_cat(d["site_type"], {"Refinery": [], "Terminal": ["tank terminal", "storage terminal"], "Depot": ["fuel depot"]}, "dim_sites")
d["site_name"] = d["site_name"].str.strip().str.title()
d["region"] = d["country"].map(REGION_OF)                     # also fills the missing regions
d["capacity_bpd"] = num(d["capacity_bpd"])
d["storage_capacity_kl"] = num(d["storage_capacity_kl"])
d["commissioned_year"] = num(d["commissioned_year"])
clean["dim_sites"] = dedupe(d, "site_id", "dim_sites")
SITES = set(clean["dim_sites"]["site_id"])

# ================================================================ dim_assets
d = raw["dim_assets"].copy()
d["asset_id"] = fix_ids(d["asset_id"])
d = fix_fk(d, "site_id", SITES, "dim_assets")
d["asset_type"] = std_cat(d["asset_type"], {"Pump": [], "Compressor": ["compr."], "Heat Exchanger": ["hx", "heat-exchanger"],
                                            "Furnace": [], "Valve Station": ["valve stn"], "Boiler": []}, "dim_assets")
d["manufacturer"] = std_cat(d["manufacturer"], {"Siemens": ["siemens ag"], "Honeywell": ["honeywell intl"], "Baker Hughes": [],
                                                "Flowserve": [], "ABB": ["a.b.b"], "Sulzer": []}, "dim_assets")
d["criticality"] = std_cat(d["criticality"], {"A": ["critical"], "B": ["medium"], "C": ["low"]}, "dim_assets")
d["install_date"] = parse_dates(d["install_date"], "dim_assets")
d["design_life_years"] = num(d["design_life_years"])
clean["dim_assets"] = dedupe(d, "asset_id", "dim_assets")
ASSETS = set(clean["dim_assets"]["asset_id"])

# ================================================================ fact_sensor_readings
d = raw["fact_sensor_readings"].copy()
d["reading_id"] = fix_ids(d["reading_id"])
d = fix_fk(d, "asset_id", ASSETS, "fact_sensor_readings")
d["reading_month"] = parse_dates(d["reading_month"], "fact_sensor_readings")
d = drop_missing(d, "reading_month", "fact_sensor_readings")
v = num(d["avg_vibration_mm_s"])
inch = (v < 0.35) & (v != 0.2)                               # in/s values are tiny (under 0.35); 0.2 mm/s is the sensor's floor reading
d["avg_vibration_mm_s"] = np.where(inch, v * 25.4, v)
log("fact_sensor_readings", "avg_vibration_mm_s: values in in/s converted to mm/s", inch.sum())
limit(d, "avg_temperature_c", 1, 250, "fact_sensor_readings")
limit(d, "avg_pressure_bar", 1, 60, "fact_sensor_readings")
clean["fact_sensor_readings"] = dedupe(d, "reading_id", "fact_sensor_readings")

# ================================================================ fact_maintenance
d = raw["fact_maintenance"].copy()
d["work_order_id"] = fix_ids(d["work_order_id"])
d = fix_fk(d, "asset_id", ASSETS, "fact_maintenance")
d["work_type"] = std_cat(d["work_type"], {"Breakdown": ["brkdwn", "unplanned", "emergency"], "Preventive": ["pm", "planned"],
                                          "Corrective": ["cm", "repair"]}, "fact_maintenance")
d["status"] = std_cat(d["status"], {"Closed": ["complete", "done"], "Open": ["in progress", "wip"]}, "fact_maintenance")
d["failure_mode"] = std_cat(d["failure_mode"], {"Bearing failure": ["bearing fail", "bearing"], "Seal leak": ["seal leakage"],
                                                "Electrical fault": ["elec fault"], "Overheating": [], "Fouling": [], "Corrosion": []},
                            "fact_maintenance", fill="None")
d["opened_date"] = parse_dates(d["opened_date"], "fact_maintenance")
d["closed_date"] = parse_dates(d["closed_date"], "fact_maintenance")
d = drop_missing(d, "opened_date", "fact_maintenance")
wrong = d["closed_date"] < d["opened_date"]
d.loc[wrong, "closed_date"] = pd.NaT
log("fact_maintenance", "closed_date earlier than opened_date set to missing", wrong.sum())
d["cost_usd"] = num(d["cost_usd"])
limit(d, "downtime_hours", 0, 400, "fact_maintenance")
clean["fact_maintenance"] = dedupe(d, "work_order_id", "fact_maintenance")

# ================================================================ fact_production
d = raw["fact_production"].copy()
d["run_id"] = fix_ids(d["run_id"])
d = fix_fk(d, "site_id", SITES, "fact_production")
d["run_date"] = parse_dates(d["run_date"], "fact_production")
d = drop_missing(d, "run_date", "fact_production")
d["crude_grade"] = std_cat(d["crude_grade"], CRUDE, "fact_production")
d["throughput_bbl"] = num(d["throughput_bbl"])
t = num(d["furnace_outlet_temp_c"])
fahr = t > 500                                               # real furnace outlet is about 365 C
d["furnace_outlet_temp_c"] = np.where(fahr, (t - 32) / 1.8, t)
log("fact_production", "furnace_outlet_temp_c: Fahrenheit converted to Celsius", fahr.sum())
limit(d, "column_pressure_bar", 0.5, 5, "fact_production")
limit(d, "gross_margin_usd_per_bbl", -40, 80, "fact_production")
for c in ["crude_cost_usd_per_bbl", "energy_use_gj_per_bbl", "unplanned_downtime_hours"]:
    d[c] = num(d[c])
clean["fact_production"] = dedupe(d, "run_id", "fact_production")

# ================================================================ fact_deliveries
d = raw["fact_deliveries"].copy()
d["trip_id"] = fix_ids(d["trip_id"])
d = fix_fk(d, "origin_site_id", SITES, "fact_deliveries")
for c in ["dispatch_time", "promised_arrival", "actual_arrival"]:
    d[c] = parse_dates(d[c], "fact_deliveries")
d = drop_missing(d, "dispatch_time", "fact_deliveries")
d["product"] = std_cat(d["product"], PRODUCT, "fact_deliveries")
d["weather"] = std_cat(d["weather"], {"Clear": [], "Rain": ["rainy", "heavy rain"], "Storm": ["thunderstorm"],
                                      "Harmattan dust": ["dust", "harmattan"]}, "fact_deliveries")
d["destination_region"] = std_cat(d["destination_region"], {r: [] for r in set(REGION_OF.values())}, "fact_deliveries")
miles = d["unit_label_distance"].astype(str).str.strip().str.lower().isin(["mi", "miles"])
d["distance_km"] = np.where(miles, num(d["distance_km"]) * 1.609, num(d["distance_km"])).round(1)
log("fact_deliveries", "distance_km: miles converted to km", miles.sum())
d = d.drop(columns="unit_label_distance")
d["volume_litres"] = num(d["volume_litres"])
d["freight_cost_usd"] = num(d["freight_cost_usd"])
limit(d, "driver_experience_years", 0, 45, "fact_deliveries")
clean["fact_deliveries"] = dedupe(d, "trip_id", "fact_deliveries")

# ================================================================ fact_quality_tests
d = raw["fact_quality_tests"].copy()
d["test_id"] = fix_ids(d["test_id"])
d = fix_fk(d, "site_id", SITES, "fact_quality_tests")
d["test_date"] = parse_dates(d["test_date"], "fact_quality_tests")
d = drop_missing(d, "test_date", "fact_quality_tests")
d["spec_result"] = std_cat(d["spec_result"], {"Pass": ["passed", "ok", "within spec"],
                                              "Off-spec": ["off spec", "offspec", "fail", "failed"]}, "fact_quality_tests")
d["lab_name"] = std_cat(d["lab_name"], {"Central Lab": ["central laboratory"], "Quality Control Lab": ["qc lab"],
                                        "Third-Party Lab": ["3rd party lab", "external lab"]}, "fact_quality_tests")
d["product"] = std_cat(d["product"], {k: v for k, v in PRODUCT.items() if k != "LPG"}, "fact_quality_tests")
d["crude_grade"] = std_cat(d["crude_grade"], CRUDE, "fact_quality_tests")
s = d["sulphur_ppm"].astype(str).str.strip()
limit_val = pd.to_numeric(s.str.extract(r"<\s*([\d.]+)")[0], errors="coerce")      # "<10" -> half the limit = 5
cens = s.str.contains(r"<|ND|BDL", regex=True) & d["sulphur_ppm"].notna()
d["sulphur_ppm"] = np.where(cens, (limit_val / 2).fillna(2.5), num(d["sulphur_ppm"]))   # ND and BDL -> 2.5
log("fact_quality_tests", "sulphur_ppm: censored results (<10, <5, ND, BDL) replaced by half the detection limit", cens.sum())
d["rework_cost_usd"] = num(d["rework_cost_usd"])
limit(d, "water_content_ppm", 0, 1500, "fact_quality_tests")
clean["fact_quality_tests"] = dedupe(d, "test_id", "fact_quality_tests")

# ================================================================ fact_inventory
d = raw["fact_inventory"].copy()
d["record_id"] = fix_ids(d["record_id"])
d = fix_fk(d, "site_id", SITES, "fact_inventory")
d["tank_id"] = fix_ids(d["tank_id"])
d["record_date"] = parse_dates(d["record_date"], "fact_inventory")
d = drop_missing(d, "record_date", "fact_inventory")
d["product"] = std_cat(d["product"], {k: v for k, v in PRODUCT.items() if k in ("PMS", "AGO", "DPK", "Jet A1")}, "fact_inventory")
bbl = d["unit"].astype(str).str.strip().str.lower().eq("bbl")
for c in ["opening_stock_kl", "receipts_kl", "dispatches_kl", "book_closing_kl", "measured_closing_kl"]:
    d[c] = np.where(bbl, num(d[c]) / 6.2898, num(d[c])).round(2)
log("fact_inventory", "volume columns: barrels converted to kilolitres", bbl.sum())
d = d.drop(columns="unit")
limit(d, "tank_temp_c", 0, 70, "fact_inventory")
d["days_since_meter_calibration"] = num(d["days_since_meter_calibration"])
clean["fact_inventory"] = dedupe(d, "record_id", "fact_inventory")

# ================================================================ results
print("\nCLEANING LOG (what was changed)")
print(pd.DataFrame(LOG).to_string(index=False))

print("\nBEFORE vs AFTER")
rows = []
for t in TABLES:
    c = clean[t]
    rows.append({"table": t, "rows_before": len(raw[t]), "rows_after": len(c), "rows_removed": len(raw[t]) - len(c),
                 "missing_cells_%_after": round(c.isna().sum().sum() / c.size * 100, 2),
                 "duplicate_ids_after": int(c.iloc[:, 0].duplicated().sum())})
print(pd.DataFrame(rows).to_string(index=False))

print("\nREMAINING MISSING VALUES (filled after the train/test split)")
for t in TABLES:
    m = clean[t].isna().sum()
    m = m[m > 0]
    if len(m):
        print(f"{t}: " + ", ".join(f"{k} {v}" for k, v in m.items()))

print("\nSANITY CHECK ON KEY COLUMNS")
print(clean["fact_sensor_readings"]["avg_vibration_mm_s"].describe().round(2).to_frame().T.to_string())
print(clean["fact_production"]["furnace_outlet_temp_c"].describe().round(1).to_frame().T.to_string())
print(clean["fact_inventory"][["opening_stock_kl", "measured_closing_kl"]].describe().round(0).to_string())
print("spec_result:", clean["fact_quality_tests"]["spec_result"].value_counts().to_dict())
print("product:", clean["fact_deliveries"]["product"].value_counts().to_dict())

pd.to_pickle(clean, "meridian_clean.pkl")
print("\nSaved: meridian_clean.pkl (dictionary of 8 clean tables: clean['fact_deliveries'] and so on)")
```

---

## 6. Feature Engineering

New columns turn raw readings into signals a model can use. Each one answers a plain operations question.

| Table | New columns | Examples |
|---|---|---|
| `fact_sensor_readings` (+18) | Trends and history | `vib_avg_3m`, `temp_avg_3m`, `lube_change_1m`, `life_used_ratio`, `breakdowns_last_6m`, `months_since_last_breakdown`, flags `vib_alert` and `lube_poor`, and the target `breakdown_next_30d` |
| `fact_maintenance` (+2) | Repair time | `repair_hours`, `month` |
| `fact_production` (+11) | Run context | `capacity_utilisation`, `sour_crude`, `heavy_crude`, `crack_spread_7d_avg`, `margin_prev_7d_avg`, `energy_x_sulphur` |
| `fact_deliveries` (+13) | Trip context | `promised_speed_kmh`, `peak_hour`, `long_haul`, `old_truck`, `novice_driver`, `bad_weather`, `traffic_x_distance`, and the target `late_delivery` |
| `fact_quality_tests` (+9) | Blend control | `dose_deviation_ppm`, `blend_temp_deviation`, `high_water`, `long_storage`, `sour_feed`, and the target `off_spec` |
| `fact_inventory` | Stock gaps | `gap_kl`, `gap_pct`, `gap_zscore`, 14-day rolling gap mean and spread, `book_arithmetic_gap` |

Rolling features look backward only, so a row never sees its own future.

### Code: feature engineering

```python
# STEP 3: FEATURE ENGINEERING
# Builds new columns (and the 3 supervised targets) from the clean tables. Nothing is dropped or scaled here.
# Leakage rule used throughout: a feature may only use information known BEFORE the thing we predict.
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    clean                                            # still in memory from Step 2
except NameError:
    clean = pd.read_pickle("meridian_clean.pkl")     # or reload what Step 2 saved
fe = {k: v.copy() for k, v in clean.items()}
NEW = {}                                             # new columns per table, for the summary


def add(table, df, cols):
    fe[table] = df
    NEW[table] = cols


def count_between(events, lo, hi):
    """How many sorted event times fall in [lo, hi)."""
    return np.searchsorted(events, hi, side="left") - np.searchsorted(events, lo, side="left")


# =========================================================== OBJECTIVE 1: predictive maintenance
# One row per asset per month. Target = a Breakdown work order opens in the FOLLOWING month.
s = fe["fact_sensor_readings"].sort_values(["asset_id", "reading_month"]).reset_index(drop=True)
a = clean["dim_assets"][["asset_id", "install_date", "design_life_years"]]
s = s.merge(a, on="asset_id", how="left")
s["asset_age_years"] = (s["reading_month"] - s["install_date"]).dt.days / 365.25
s["life_used_ratio"] = s["asset_age_years"] / s["design_life_years"]
s["utilisation_pct"] = s["operating_hours"] / (s["reading_month"].dt.days_in_month * 24) * 100
s["month_of_year"] = s["reading_month"].dt.month

g = s.groupby("asset_id")
for col, short in [("avg_vibration_mm_s", "vib"), ("avg_temperature_c", "temp"), ("avg_pressure_bar", "press"),
                   ("lubricant_condition_score", "lube")]:
    s[f"{short}_avg_3m"] = g[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    s[f"{short}_change_1m"] = g[col].diff()
s["vib_alert"] = (s["avg_vibration_mm_s"] >= 4.5).astype(float).where(s["avg_vibration_mm_s"].notna())   # 4.5 mm/s = typical alarm level
s["lube_poor"] = (s["lubricant_condition_score"] < 60).astype(float).where(s["lubricant_condition_score"].notna())

# History from the maintenance table (only events up to the end of the reading month) and the target
mt = clean["fact_maintenance"]
ev = {k: {w: np.sort(x["opened_date"].values) for w, x in grp.groupby("work_type")} for k, grp in mt.groupby("asset_id")}
empty = np.array([], dtype="datetime64[ns]")
nxt = (s["reading_month"] + pd.offsets.MonthBegin(1)).values          # first day of the following month
nxt2 = (s["reading_month"] + pd.offsets.MonthBegin(2)).values
back6 = (s["reading_month"] + pd.offsets.MonthBegin(1) - pd.DateOffset(months=6)).values
tgt, bd6, pm6, since = np.zeros(len(s)), np.zeros(len(s)), np.zeros(len(s)), np.full(len(s), 36.0)
for asset, idx in s.groupby("asset_id").indices.items():
    b = ev.get(asset, {}).get("Breakdown", empty)
    p = ev.get(asset, {}).get("Preventive", empty)
    tgt[idx] = count_between(b, nxt[idx], nxt2[idx]) > 0
    bd6[idx] = count_between(b, back6[idx], nxt[idx])
    pm6[idx] = count_between(p, back6[idx], nxt[idx])
    for i in idx:                                                      # months since the last breakdown (36 = none)
        k = np.searchsorted(b, nxt[i], side="left")
        if k > 0:
            since[i] = (nxt[i] - b[k - 1]) / np.timedelta64(1, "D") / 30.4
s["breakdowns_last_6m"], s["preventive_jobs_last_6m"] = bd6, pm6
s["months_since_last_breakdown"] = since
s["breakdown_next_30d"] = tgt.astype(int)
s = s.drop(columns=["install_date", "design_life_years"])
add("fact_sensor_readings", s, ["asset_age_years", "life_used_ratio", "utilisation_pct", "month_of_year", "vib_avg_3m",
    "vib_change_1m", "temp_avg_3m", "temp_change_1m", "press_avg_3m", "press_change_1m", "lube_avg_3m", "lube_change_1m",
    "vib_alert", "lube_poor", "breakdowns_last_6m", "preventive_jobs_last_6m", "months_since_last_breakdown",
    "breakdown_next_30d (TARGET)"])

m = fe["fact_maintenance"].copy()                                      # used later in business analysis
m["repair_hours"] = (m["closed_date"] - m["opened_date"]).dt.total_seconds() / 3600
m["month"] = m["opened_date"].dt.to_period("M").astype(str)
add("fact_maintenance", m, ["repair_hours", "month"])

# =========================================================== OBJECTIVE 2: refinery margin
p = fe["fact_production"].sort_values(["site_id", "run_date"]).reset_index(drop=True)
cap = clean["dim_sites"][["site_id", "capacity_bpd"]]
p = p.merge(cap, on="site_id", how="left")
p["capacity_utilisation"] = p["throughput_bbl"] / p["capacity_bpd"]
p["month"], p["quarter"], p["day_of_week"] = p["run_date"].dt.month, p["run_date"].dt.quarter, p["run_date"].dt.dayofweek
p["sour_crude"] = (p["crude_sulphur_pct"] > 1.5).astype(int)           # high-sulphur crude is costlier to process
p["heavy_crude"] = (p["crude_api_gravity"] < 32).astype(int)           # low API = heavy crude
p["had_downtime"] = (p["unplanned_downtime_hours"] > 0).astype(float).where(p["unplanned_downtime_hours"].notna())
p["energy_x_sulphur"] = p["energy_use_gj_per_bbl"] * p["crude_sulphur_pct"]
gp = p.groupby("site_id")
p["crack_spread_7d_avg"] = gp["crack_spread_usd_per_bbl"].transform(lambda x: x.rolling(7, min_periods=1).mean())
p["crude_cost_change_7d"] = gp["crude_cost_usd_per_bbl"].transform(lambda x: x - x.shift(7))
p["margin_prev_7d_avg"] = gp["gross_margin_usd_per_bbl"].transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())  # past margins only
p = p.drop(columns="capacity_bpd")
add("fact_production", p, ["capacity_utilisation", "month", "quarter", "day_of_week", "sour_crude", "heavy_crude",
    "had_downtime", "energy_x_sulphur", "crack_spread_7d_avg", "crude_cost_change_7d", "margin_prev_7d_avg"])

# =========================================================== OBJECTIVE 3: late delivery
d = fe["fact_deliveries"].copy()
d["promised_hours"] = (d["promised_arrival"] - d["dispatch_time"]).dt.total_seconds() / 3600
d["promised_speed_kmh"] = d["distance_km"] / d["promised_hours"].where(d["promised_hours"] > 0)   # tight promises are risky
d["dispatch_hour"], d["dispatch_day"], d["dispatch_month"] = d["dispatch_time"].dt.hour, d["dispatch_time"].dt.dayofweek, d["dispatch_time"].dt.month
d["is_weekend"] = (d["dispatch_day"] >= 5).astype(int)
d["peak_hour"] = d["dispatch_hour"].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)
d["long_haul"] = (d["distance_km"] > 200).astype(int)
d["old_truck"] = (d["truck_age_years"] > 10).astype(float).where(d["truck_age_years"].notna())
d["novice_driver"] = (d["driver_experience_years"] < 3).astype(float).where(d["driver_experience_years"].notna())
d["bad_weather"] = d["weather"].isin(["Rain", "Storm", "Harmattan dust"]).astype(int)
d["traffic_x_distance"] = d["traffic_index"] * d["distance_km"]
delivered = d["trip_status"].eq("Delivered") & d["actual_arrival"].notna() & d["promised_arrival"].notna()
late = d["actual_arrival"] > d["promised_arrival"] + pd.Timedelta(minutes=30)
d["late_delivery"] = np.where(delivered, late.astype(int), np.nan)     # cancelled or untimed trips have no label
add("fact_deliveries", d, ["promised_hours", "promised_speed_kmh", "dispatch_hour", "dispatch_day", "dispatch_month",
    "is_weekend", "peak_hour", "long_haul", "old_truck", "novice_driver", "bad_weather", "traffic_x_distance",
    "late_delivery (TARGET)"])

# =========================================================== OBJECTIVE 4: off-spec batches
q = fe["fact_quality_tests"].copy()
cut = q["test_date"].quantile(0.8)                                     # typical settings come from the training period only
typ_dose = q.loc[q["test_date"] <= cut, "additive_dose_ppm"].median()
typ_temp = q.loc[q["test_date"] <= cut, "blend_temp_c"].median()
q["dose_deviation_ppm"] = (q["additive_dose_ppm"] - typ_dose).abs()
q["blend_temp_deviation"] = q["blend_temp_c"] - typ_temp
q["high_water"] = (q["water_content_ppm"] > 80).astype(float).where(q["water_content_ppm"].notna())
q["long_storage"] = (q["storage_days"] > 14).astype(float).where(q["storage_days"].notna())
q["sour_feed"] = (q["feed_crude_sulphur_pct"] > 1.0).astype(float).where(q["feed_crude_sulphur_pct"].notna())
q["product_group"] = np.where(q["product"].eq("PMS"), "Gasoline", "Middle distillate")
q["test_month"], q["test_quarter"] = q["test_date"].dt.month, q["test_date"].dt.quarter
q["off_spec"] = q["spec_result"].eq("Off-spec").astype(int)
add("fact_quality_tests", q, ["dose_deviation_ppm", "blend_temp_deviation", "high_water", "long_storage", "sour_feed",
    "product_group", "test_month", "test_quarter", "off_spec (TARGET)"])
print(f"Typical settings from training period: dose {typ_dose:.1f} ppm, blend temp {typ_temp:.1f} C")

# =========================================================== OBJECTIVE 5: inventory anomalies (no target)
v = fe["fact_inventory"].sort_values(["tank_id", "record_date"]).reset_index(drop=True)
v["gap_kl"] = v["measured_closing_kl"] - v["book_closing_kl"]
v["gap_pct"] = v["gap_kl"] / v["book_closing_kl"] * 100
v["abs_gap_pct"] = v["gap_pct"].abs()
v["measured_to_book_ratio"] = v["measured_closing_kl"] / v["book_closing_kl"]
v["book_arithmetic_gap"] = v["opening_stock_kl"] + v["receipts_kl"] - v["dispatches_kl"] - v["book_closing_kl"]
v["dispatch_rate_pct"] = v["dispatches_kl"] / v["opening_stock_kl"] * 100
v["received_today"] = (v["receipts_kl"] > 0).astype(float).where(v["receipts_kl"].notna())
gv = v.groupby("tank_id")["gap_pct"]
v["gap_pct_mean_14d"] = gv.transform(lambda x: x.shift(1).rolling(14, min_periods=5).mean())     # this tank's normal level
v["gap_pct_std_14d"] = gv.transform(lambda x: x.shift(1).rolling(14, min_periods=5).std())
v["gap_zscore"] = (v["gap_pct"] - v["gap_pct_mean_14d"]) / v["gap_pct_std_14d"].clip(lower=0.05)
v["record_month"], v["record_day"] = v["record_date"].dt.month, v["record_date"].dt.dayofweek
add("fact_inventory", v, ["gap_kl", "gap_pct", "abs_gap_pct", "measured_to_book_ratio", "book_arithmetic_gap",
    "dispatch_rate_pct", "received_today", "gap_pct_mean_14d", "gap_pct_std_14d", "gap_zscore", "record_month", "record_day"])

# =========================================================== site feature
st = fe["dim_sites"].copy()
st["site_age_years"] = 2024 - st["commissioned_year"]
add("dim_sites", st, ["site_age_years"])

# =========================================================== summary
print("\nNEW FEATURES PER TABLE")
for t, cols in NEW.items():
    print(f"\n{t}  (+{len(cols)} columns, {len(fe[t]):,} rows)")
    for c in cols:
        name = c.split(" ")[0]
        print(f"   {c:<42} missing: {fe[t][name].isna().sum():>5,}")

print("\nTARGET BASE RATES")
print(f"  breakdown_next_30d : {fe['fact_sensor_readings']['breakdown_next_30d'].mean():.1%} of asset-months")
print(f"  late_delivery      : {fe['fact_deliveries']['late_delivery'].mean():.1%} of delivered trips "
      f"({fe['fact_deliveries']['late_delivery'].notna().sum():,} labelled)")
print(f"  off_spec           : {fe['fact_quality_tests']['off_spec'].mean():.1%} of batches")
print(f"  gross_margin target: mean {fe['fact_production']['gross_margin_usd_per_bbl'].mean():.2f} USD/bbl")

print("\nQUICK SIGNAL CHECK (average of the target by a key new feature)")
print(fe["fact_sensor_readings"].groupby(pd.cut(fe["fact_sensor_readings"]["life_used_ratio"], [0, .5, .8, 1.0, 2.0]))["breakdown_next_30d"].mean().round(3).to_string())
print(fe["fact_deliveries"].groupby(pd.cut(fe["fact_deliveries"]["promised_speed_kmh"], [0, 20, 25, 30, 60]))["late_delivery"].mean().round(3).to_string())
print(fe["fact_quality_tests"].groupby("high_water")["off_spec"].mean().round(3).to_string())

pd.to_pickle(fe, "meridian_features.pkl")
print("\nSaved: meridian_features.pkl  (dictionary 'fe' holds the 8 tables with the new columns)")
```

---

## 7. Create Relationship Table

The eight tables are joined in a **star schema**: two dimensions (`dim_sites`, `dim_assets`) plus a generated `dim_date`, with the fact tables around them. The integrity check passed: every fact row finds its parent (100% match) and every parent key is unique.

<p align="center">
  <img src="images/star_schema.png" alt="Star schema: dimensions in the centre, fact tables around them" width="100%">
</p>
<p align="center"><em>Star schema: dimensions in the centre, fact tables around them</em></p>

| Relationship | Child rows | Match |
|---|---|---|
| sensor readings to assets | 8,477 | 100% |
| maintenance to assets | 1,658 | 100% |
| assets to sites | 356 | 100% |
| production, deliveries, quality, inventory to sites | 4,349 / 5,941 / 4,152 / 10,182 | 100% |
| every fact table to `dim_date` | all | 100% |

### Code: relationship table (star schema)

```python
# STEP 4: RELATIONSHIP TABLE (STAR SCHEMA)
# Documents how the tables join, proves every join works (referential integrity), adds a date dimension,
# and draws the schema. Nothing in the data tables is changed except one new table, dim_date.
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")
try:
    fe
except NameError:
    fe = pd.read_pickle("meridian_features.pkl")

# ---------------------------------------------------------------- 1. date dimension
DATE_COL = {"fact_sensor_readings": "reading_month", "fact_maintenance": "opened_date", "fact_production": "run_date",
            "fact_deliveries": "dispatch_time", "fact_quality_tests": "test_date", "fact_inventory": "record_date"}
all_dates = pd.concat([fe[t][c].dt.normalize() for t, c in DATE_COL.items()])
dim_date = pd.DataFrame({"date": pd.date_range(all_dates.min(), all_dates.max(), freq="D")})
dim_date["year"], dim_date["quarter"], dim_date["month"] = dim_date["date"].dt.year, dim_date["date"].dt.quarter, dim_date["date"].dt.month
dim_date["month_name"], dim_date["day_name"] = dim_date["date"].dt.month_name(), dim_date["date"].dt.day_name()
dim_date["week_of_year"], dim_date["day_of_week"] = dim_date["date"].dt.isocalendar().week.astype(int), dim_date["date"].dt.dayofweek
dim_date["is_weekend"] = (dim_date["day_of_week"] >= 5).astype(int)
dim_date["year_month"] = dim_date["date"].dt.to_period("M").astype(str)
fe["dim_date"] = dim_date

# ---------------------------------------------------------------- 2. relationship table
LINKS = [  # (fact/child table, key column, parent table, parent key, what it supports)
    ("fact_sensor_readings", "asset_id", "dim_assets", "asset_id", "Obj 1: sensor history per asset"),
    ("fact_maintenance", "asset_id", "dim_assets", "asset_id", "Obj 1: breakdown and repair history"),
    ("dim_assets", "site_id", "dim_sites", "site_id", "Obj 1: where each asset sits"),
    ("fact_production", "site_id", "dim_sites", "site_id", "Obj 2: refinery runs by site"),
    ("fact_deliveries", "origin_site_id", "dim_sites", "site_id", "Obj 3: trips by origin terminal or depot"),
    ("fact_quality_tests", "site_id", "dim_sites", "site_id", "Obj 4: batches by site"),
    ("fact_inventory", "site_id", "dim_sites", "site_id", "Obj 5: tanks by site"),
] + [(t, c, "dim_date", "date", "Time analysis and chronological splits") for t, c in DATE_COL.items()]


def key(df, col):
    return df[col].dt.normalize() if pd.api.types.is_datetime64_any_dtype(df[col]) else df[col]


rows = []
for child, ccol, parent, pcol, purpose in LINKS:
    ck, pk = key(fe[child], ccol), key(fe[parent], pcol)
    matched = ck.isin(set(pk.dropna()))
    rows.append({"child_table (many)": child, "child_key": ccol, "parent_table (one)": parent, "parent_key": pcol,
                 "child_rows": len(ck), "rows_without_parent": int((~matched & ck.notna()).sum()) + int(ck.isna().sum()),
                 "match_%": round(matched.mean() * 100, 2), "parent_keys_used_%": round(pk.isin(set(ck.dropna())).mean() * 100, 1),
                 "parent_key_unique": bool(pk.is_unique), "supports": purpose})
rel = pd.DataFrame(rows)
rel.insert(0, "relationship", rel["child_table (many)"] + " -> " + rel["parent_table (one)"])
rel["cardinality"] = "many-to-one"
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 45)
print("RELATIONSHIP TABLE")
print(rel[["relationship", "child_key", "parent_key", "cardinality", "child_rows", "rows_without_parent", "match_%",
           "parent_keys_used_%", "parent_key_unique"]].to_string(index=False))
assert (rel["rows_without_parent"] == 0).all() and rel["parent_key_unique"].all(), "A join is broken, check Step 2"
print("\nIntegrity check passed: every fact row finds its parent, and every parent key is unique.")

print("\nTABLE ROLES")
roles = pd.DataFrame([
    ("dim_date", "Dimension", "date", len(fe["dim_date"]), "One row per calendar day"),
    ("dim_sites", "Dimension", "site_id", len(fe["dim_sites"]), "Refineries, terminals and depots"),
    ("dim_assets", "Dimension (child of dim_sites)", "asset_id", len(fe["dim_assets"]), "Critical equipment"),
    ("fact_sensor_readings", "Fact", "reading_id", len(fe["fact_sensor_readings"]), "Asset-month sensor averages"),
    ("fact_maintenance", "Fact", "work_order_id", len(fe["fact_maintenance"]), "Work orders"),
    ("fact_production", "Fact", "run_id", len(fe["fact_production"]), "Refinery-day runs"),
    ("fact_deliveries", "Fact", "trip_id", len(fe["fact_deliveries"]), "Fuel trips"),
    ("fact_quality_tests", "Fact", "test_id", len(fe["fact_quality_tests"]), "Batch lab tests"),
    ("fact_inventory", "Fact", "record_id", len(fe["fact_inventory"]), "Tank-day reconciliations")],
    columns=["table", "role", "primary_key", "rows", "grain"])
print(roles.to_string(index=False))

# ---------------------------------------------------------------- 3. star schema diagram
fig, ax = plt.subplots(figsize=(15, 7.5))
ax.set_xlim(0, 15); ax.set_ylim(0.6, 8.1); ax.axis("off")
BOX = {  # name: (x, y, width, height, colour)
    "dim_date": (6.0, 6.9, 2.6, 1.0, "#F4A261"), "dim_sites": (4.7, 3.4, 2.6, 1.5, "#2A9D8F"), "dim_assets": (8.7, 3.4, 2.6, 1.5, "#2A9D8F"),
    "fact_production": (0.1, 6.1, 2.6, 0.95, "#264653"), "fact_deliveries": (0.1, 4.7, 2.6, 0.95, "#264653"),
    "fact_quality_tests": (0.1, 3.3, 2.6, 0.95, "#264653"), "fact_inventory": (0.1, 1.9, 2.6, 0.95, "#264653"),
    "fact_sensor_readings": (12.4, 4.4, 2.5, 0.95, "#264653"), "fact_maintenance": (12.4, 2.9, 2.5, 0.95, "#264653")}
SUB = {"dim_date": "date (PK)", "dim_sites": "site_id (PK)\n24 sites", "dim_assets": "asset_id (PK)\nsite_id (FK), 356 assets",
       "fact_production": "Obj 2 margin", "fact_deliveries": "Obj 3 late delivery", "fact_quality_tests": "Obj 4 off-spec",
       "fact_inventory": "Obj 5 anomalies", "fact_sensor_readings": "Obj 1 breakdown", "fact_maintenance": "Obj 1 history"}
for name, (x, y, w, h, col) in BOX.items():
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=col, ec="white", lw=2))
    ax.text(x + w / 2, y + h * 0.62, name, ha="center", va="center", color="white", fontsize=11.5, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.27, SUB[name], ha="center", va="center", color="white", fontsize=8.5)


def point(name, side, shift=0.0):
    x, y, w, h, _ = BOX[name]
    return {"l": (x, y + h / 2 + shift), "r": (x + w, y + h / 2 + shift),
            "t": (x + w / 2 + shift, y + h), "b": (x + w / 2 + shift, y)}[side]


def edge(a, sa, b, sb, label="", shift_b=0.0, color="#555", ls="-", lab_pos=0.3, dy=0.12):
    p1, p2 = point(a, sa), point(b, sb, shift_b)
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color=color, lw=1.4, ls=ls, shrinkA=2, shrinkB=2))
    if label:
        ax.text(p1[0] + (p2[0] - p1[0]) * lab_pos, p1[1] + (p2[1] - p1[1]) * lab_pos + dy, label, fontsize=8, color="#333",
                ha="center", bbox=dict(fc="white", ec="none", pad=0.8))


for f, sh in [("fact_production", 0.45), ("fact_deliveries", 0.15), ("fact_quality_tests", -0.15), ("fact_inventory", -0.45)]:
    edge(f, "r", "dim_sites", "l", "origin_site_id" if f == "fact_deliveries" else "site_id", shift_b=sh, lab_pos=0.35)
edge("fact_sensor_readings", "l", "dim_assets", "r", "asset_id", shift_b=0.25, lab_pos=0.5)
edge("fact_maintenance", "l", "dim_assets", "r", "asset_id", shift_b=-0.25, lab_pos=0.5)
edge("dim_assets", "l", "dim_sites", "r", "site_id", lab_pos=0.5)
edge("fact_production", "t", "dim_date", "l", "date", color="#F4A261", ls="--", lab_pos=0.5)
edge("fact_sensor_readings", "t", "dim_date", "r", "date", color="#F4A261", ls="--", lab_pos=0.5)
ax.text(7.5, 1.0, "Dashed orange lines: every fact table also joins dim_date on its own date column (only two are drawn to keep the picture clear)",
        ha="center", fontsize=9.5, color="#444")
ax.set_title("Meridian Downstream Energy: star schema (many-to-one arrows point to the 'one' side)", fontsize=13, pad=10)
plt.tight_layout()
plt.savefig("star_schema.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------- 4. save
rel.to_csv("relationship_table.csv", index=False)
pd.to_pickle(fe, "meridian_features.pkl")        # fe now also contains dim_date
print("\nSaved: relationship_table.csv, star_schema.png, and meridian_features.pkl (now with dim_date)")
```

---

## 8. Exploratory Data Analysis

One figure per business objective. Five findings stood out:

- **Breakdowns follow sensor trends.** Assets that have used over 100% of design life break down in 6.4% of months against 3.9% under 50%. The 3-month vibration average is 3.95 mm/s before a breakdown against 3.11 otherwise.
- **Crude choice moves margin.** Average margin is 11.5 USD/bbl. WTI earns 13.4 and Basrah Medium 8.1. Sour crude earns 8.9 against 12.4 for sweet. Each 1 USD of crack spread adds about 1.03 USD of margin.
- **Late trips come from tight promises and weather.** 10.5% of trips are late. When the promise needs over 30 km/h it is 20%, under 20 km/h it is 6%. Storm is the worst weather at 34%.
- **Water decides off-spec batches.** 7.2% of batches fail, 39% when water is above 120 ppm against 3% under 25 ppm. A failed batch costs about $11,699 in rework.
- **Inventory gaps are rare and concentrated.** 3.3% of tank-days have a gap above 1% of book volume. Tank TNK-010 is the most affected.

### Charts

<p align="center">
  <img src="images/eda_1_maintenance.png" alt="Objective 1: what comes before a breakdown" width="100%">
</p>
<p align="center"><em>Objective 1: what comes before a breakdown</em></p>

<p align="center">
  <img src="images/eda_2_margin.png" alt="Objective 2: margin by crude grade, sulphur and crack spread" width="100%">
</p>
<p align="center"><em>Objective 2: margin by crude grade, sulphur and crack spread</em></p>

<p align="center">
  <img src="images/eda_3_delivery.png" alt="Objective 3: late rate by promised speed, weather and traffic" width="100%">
</p>
<p align="center"><em>Objective 3: late rate by promised speed, weather and traffic</em></p>

<p align="center">
  <img src="images/eda_4_quality.png" alt="Objective 4: off-spec rate by water, dose and storage" width="100%">
</p>
<p align="center"><em>Objective 4: off-spec rate by water, dose and storage</em></p>

<p align="center">
  <img src="images/eda_5_inventory.png" alt="Objective 5: tank-day gaps" width="100%">
</p>
<p align="center"><em>Objective 5: tank-day gaps</em></p>

### Code: exploratory data analysis

```python
# STEP 5: EXPLORATORY DATA ANALYSIS (read-only). One figure per business objective, plus a short list of findings.
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
try:
    fe
except NameError:
    fe = pd.read_pickle("meridian_features.pkl")

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
C1, C2, C3 = "#264653", "#E76F51", "#2A9D8F"
FINDINGS = []


def rate_bar(ax, series, target, title, xlabel, order=None, rot=0, pct=True):
    """Bar chart of the average of a 0/1 target for each group, with the overall rate as a dashed line."""
    r = target.groupby(series, observed=True).mean()
    r = r.reindex(order) if order is not None else r
    ax.bar(r.index.astype(str), r.values * (100 if pct else 1), color=C1)
    ax.axhline(target.mean() * (100 if pct else 1), color=C2, ls="--", lw=1.5, label="overall rate")
    ax.set(title=title, xlabel=xlabel, ylabel="% of rows" if pct else "")
    ax.tick_params(axis="x", rotation=rot); ax.legend(frameon=False)
    return r


# ============================================================ OBJECTIVE 1: predictive maintenance
s = fe["fact_sensor_readings"].merge(fe["dim_assets"][["asset_id", "asset_type", "criticality"]], on="asset_id", how="left")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
r = rate_bar(ax[0], s["asset_type"], s["breakdown_next_30d"], "Breakdown rate by asset type", "asset type", rot=25)
band = pd.cut(s["life_used_ratio"], [0, .5, .8, 1.0, 2.0], labels=["under 50%", "50-80%", "80-100%", "over 100%"])
r2 = rate_bar(ax[1], band, s["breakdown_next_30d"], "Older assets break down more", "share of design life used")
for lab, col in [(0, C1), (1, C2)]:
    ax[2].hist(s.loc[s["breakdown_next_30d"] == lab, "vib_avg_3m"].dropna(), bins=40, density=True, alpha=.6, color=col,
               label="no breakdown next month" if lab == 0 else "breakdown next month")
ax[2].set(title="3-month vibration before a breakdown", xlabel="average vibration (mm/s)", ylabel="density"); ax[2].legend(frameon=False)
fig.suptitle("Objective 1: predictive maintenance", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout(); plt.savefig("eda_1_maintenance.png", dpi=140, bbox_inches="tight"); plt.show()
vib0, vib1 = s.groupby("breakdown_next_30d")["vib_avg_3m"].mean()
FINDINGS.append(f"Obj 1: assets that used over 100% of design life break down {r2.iloc[-1]*100:.1f}% of months vs {r2.iloc[0]*100:.1f}% under 50%; "
                f"3-month vibration averages {vib1:.2f} mm/s before a breakdown vs {vib0:.2f} otherwise.")

# ============================================================ OBJECTIVE 2: refinery margin
p = fe["fact_production"].merge(fe["dim_sites"][["site_id", "country"]], on="site_id", how="left")
p = p[p["gross_margin_usd_per_bbl"].notna()]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].hist(p["gross_margin_usd_per_bbl"], bins=50, color=C1)
ax[0].axvline(p["gross_margin_usd_per_bbl"].mean(), color=C2, ls="--", label=f"mean {p['gross_margin_usd_per_bbl'].mean():.1f}")
ax[0].set(title="Gross margin per barrel", xlabel="USD per barrel", ylabel="refinery-days"); ax[0].legend(frameon=False)
g = p.groupby("crude_grade")["gross_margin_usd_per_bbl"].mean().sort_values()
ax[1].barh(g.index, g.values, color=[C2 if v == g.min() else C1 for v in g.values])
ax[1].set(title="Average margin by crude grade", xlabel="USD per barrel")
smp = p.sample(min(1500, len(p)), random_state=1)
ax[2].scatter(smp["crack_spread_usd_per_bbl"], smp["gross_margin_usd_per_bbl"], s=8, alpha=.35, color=C1)
ok = smp[["crack_spread_usd_per_bbl", "gross_margin_usd_per_bbl"]].dropna()
k, b = np.polyfit(ok.iloc[:, 0], ok.iloc[:, 1], 1)
xs = np.linspace(ok.iloc[:, 0].min(), ok.iloc[:, 0].max(), 10); ax[2].plot(xs, k * xs + b, color=C2, lw=2)
ax[2].set(title="Margin follows the crack spread", xlabel="crack spread (USD per barrel)", ylabel="margin (USD per barrel)")
fig.suptitle("Objective 2: refinery margin", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout(); plt.savefig("eda_2_margin.png", dpi=140, bbox_inches="tight"); plt.show()
sour = p.groupby("sour_crude")["gross_margin_usd_per_bbl"].mean()
FINDINGS.append(f"Obj 2: average margin {p['gross_margin_usd_per_bbl'].mean():.1f} USD/bbl; best crude {g.index[-1]} ({g.iloc[-1]:.1f}), worst {g.index[0]} ({g.iloc[0]:.1f}); "
                f"sour crude earns {sour.get(1, np.nan):.1f} vs {sour.get(0, np.nan):.1f} for sweet; each 1 USD of crack spread adds about {k:.2f} USD margin.")

# ============================================================ OBJECTIVE 3: late delivery
d = fe["fact_deliveries"].dropna(subset=["late_delivery"]).copy()
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
sp = pd.cut(d["promised_speed_kmh"], [0, 20, 25, 30, 60], labels=["under 20", "20-25", "25-30", "over 30"])
r3 = rate_bar(ax[0], sp, d["late_delivery"], "Tight promises are late more often", "promised speed (km/h)")
wx = rate_bar(ax[1], d["weather"], d["late_delivery"], "Late rate by weather", "weather", rot=15)
hr = d.groupby("dispatch_hour")["late_delivery"].mean() * 100
ax[2].plot(hr.index, hr.values, marker="o", color=C1); ax[2].axhline(d["late_delivery"].mean() * 100, color=C2, ls="--")
ax[2].set(title="Late rate by dispatch hour", xlabel="hour of day", ylabel="% late")
fig.suptitle("Objective 3: on-time delivery", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout(); plt.savefig("eda_3_delivery.png", dpi=140, bbox_inches="tight"); plt.show()
FINDINGS.append(f"Obj 3: {d['late_delivery'].mean():.1%} of trips are late; {r3.iloc[-1]*100:.0f}% when the promise needs over 30 km/h vs {r3.iloc[0]*100:.0f}% under 20; "
                f"worst weather is {wx.idxmax()} ({wx.max()*100:.0f}%).")

# ============================================================ OBJECTIVE 4: off-spec batches
q = fe["fact_quality_tests"].copy()
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
wb = pd.cut(q["water_content_ppm"], [-1, 25, 50, 80, 120, 2000], labels=["0-25", "25-50", "50-80", "80-120", "over 120"])
r4 = rate_bar(ax[0], wb, q["off_spec"], "Water content drives off-spec batches", "water content (ppm)")
db = pd.cut(q["dose_deviation_ppm"], [-1, 10, 20, 30, 300], labels=["0-10", "10-20", "20-30", "over 30"])
rate_bar(ax[1], db, q["off_spec"], "Additive dose away from normal", "dose deviation (ppm)")
cr = q.groupby("crude_grade")["off_spec"].mean().sort_values() * 100
ax[2].barh(cr.index, cr.values, color=C1); ax[2].axvline(q["off_spec"].mean() * 100, color=C2, ls="--")
ax[2].set(title="Off-spec rate by crude grade", xlabel="% of batches")
fig.suptitle("Objective 4: off-spec product", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout(); plt.savefig("eda_4_quality.png", dpi=140, bbox_inches="tight"); plt.show()
cost = q.loc[q["off_spec"] == 1, "rework_cost_usd"].mean()
FINDINGS.append(f"Obj 4: {q['off_spec'].mean():.1%} of batches fail; {r4.iloc[-1]*100:.0f}% fail when water exceeds 120 ppm vs {r4.iloc[0]*100:.0f}% under 25; "
                f"average rework cost of a failed batch is about {cost:,.0f} USD.")

# ============================================================ OBJECTIVE 5: inventory anomalies
v = fe["fact_inventory"].dropna(subset=["gap_pct"]).copy()
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].hist(v["gap_pct"].clip(-8, 8), bins=80, color=C1)
ax[0].set_yscale("log"); ax[0].set(title="Measured vs book volume gap", xlabel="gap (% of book, clipped at +/-8)", ylabel="tank-days (log scale)")
worst = v.groupby("tank_id")["abs_gap_pct"].apply(lambda x: (x > 1).mean()).idxmax()
tk = v[v["tank_id"] == worst].sort_values("record_date")
ax[1].plot(tk["record_date"], tk["gap_pct"].clip(-8, 8), lw=.8, color=C1)
big = tk[tk["abs_gap_pct"] > 1]
ax[1].scatter(big["record_date"], big["gap_pct"].clip(-8, 8), color=C2, s=25, zorder=3, label="gap over 1%")
ax[1].set(title=f"Daily gap for tank {worst}", xlabel="", ylabel="gap (% of book)"); ax[1].legend(frameon=False); ax[1].tick_params(axis="x", rotation=20)
ax[2].hist(v["gap_zscore"].dropna().clip(-8, 8), bins=70, color=C3); ax[2].set_yscale("log")
ax[2].set(title="Gap vs the tank's own normal (z-score)", xlabel="z-score (clipped at +/-8)", ylabel="tank-days (log scale)")
fig.suptitle("Objective 5: inventory anomalies (unlabelled)", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout(); plt.savefig("eda_5_inventory.png", dpi=140, bbox_inches="tight"); plt.show()
FINDINGS.append(f"Obj 5: {(v['abs_gap_pct'] > 1).mean():.1%} of tank-days have a gap above 1% of book volume and {(v['gap_zscore'].abs() > 4).mean():.1%} sit more than 4 standard deviations "
                f"from the tank's own normal; the most affected tank is {worst}.")

# ============================================================ data-health overview
m = pd.DataFrame({t: [fe[t].isna().sum().sum() / fe[t].size * 100] for t in fe if t != "dim_date"}).T
m.columns = ["missing_%"]
fig, ax = plt.subplots(figsize=(9, 3.8))
m["missing_%"].sort_values().plot.barh(ax=ax, color=C1)
ax.set(title="Share of cells still missing, by table (filled after the train/test split)", xlabel="% missing")
plt.tight_layout(); plt.savefig("eda_0_missing.png", dpi=140, bbox_inches="tight"); plt.show()

print("KEY EDA FINDINGS")
for f in FINDINGS:
    print("-", f)
print("\nSaved: eda_0_missing.png, eda_1_maintenance.png ... eda_5_inventory.png")
```

---

## 9. Analytical Base Table

One modelling table per objective, each at the grain of its decision: asset-month, refinery-day, trip, batch, tank-day. Rows with no target are dropped.

| Table | Task | Rows | Numeric / categorical features | Target (mean) | Period |
|---|---|---|---|---|---|
| `abt_1_maintenance` | Classification | 8,477 | 22 / 5 | `breakdown_next_30d` (4.6%) | 2023-01 to 2024-12 |
| `abt_2_margin` | Regression | 4,339 | 20 / 2 | `gross_margin_usd_per_bbl` (11.51) | 2023-01 to 2024-12 |
| `abt_3_delivery` | Classification | 5,712 | 17 / 5 | `late_delivery` (10.5%) | 2023-01 to 2025-01 |
| `abt_4_quality` | Classification | 4,152 | 12 / 5 | `off_spec` (7.2%) | 2023-01 to 2024-12 |
| `abt_5_inventory` | Anomaly detection | 10,182 | 19 / 3 | none | 2024 |

**Columns left out on purpose (leakage)**

| Table | Excluded |
|---|---|
| `abt_1_maintenance` | downtime_hours, cost_usd, closed_date, failure_mode, technician_hours |
| `abt_2_margin` | gasoline, diesel, jet and fuel oil yield percentages |
| `abt_3_delivery` | actual_arrival, freight_cost_usd, trip_status, promised_arrival |
| `abt_4_quality` | sulphur_ppm, density, octane, flash point, spec_result, rework_cost_usd, lab_name |
| `abt_5_inventory` | anomaly_type, is_anomaly |

### Code: analytical base tables

```python
# STEP 6: ANALYTICAL BASE TABLE (ABT)
# Joins the star schema into ONE modelling table per business objective: one row per thing we predict,
# identifier columns, the target, and only leakage-safe features. Nothing is encoded, split, filled or scaled yet.
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    fe
except NameError:
    fe = pd.read_pickle("meridian_features.pkl")

sites = fe["dim_sites"][["site_id", "country", "region", "site_type"]]
assets = fe["dim_assets"][["asset_id", "site_id", "asset_type", "manufacturer", "criticality"]]

# name -> (source table, joins, id columns, date column, target, feature columns, columns that must NOT be features)
SPEC = {
    "abt_1_maintenance": dict(
        src="fact_sensor_readings", join="assets+sites", ids=["reading_id", "asset_id", "site_id"], date="reading_month",
        target="breakdown_next_30d", task="classification",
        features=["asset_type", "manufacturer", "criticality", "country", "site_type", "avg_vibration_mm_s", "avg_temperature_c",
                  "avg_pressure_bar", "operating_hours", "lubricant_condition_score", "asset_age_years", "life_used_ratio",
                  "utilisation_pct", "month_of_year", "vib_avg_3m", "vib_change_1m", "temp_avg_3m", "temp_change_1m",
                  "press_avg_3m", "press_change_1m", "lube_avg_3m", "lube_change_1m", "vib_alert", "lube_poor",
                  "breakdowns_last_6m", "preventive_jobs_last_6m", "months_since_last_breakdown"],
        forbidden=["downtime_hours", "cost_usd", "closed_date", "failure_mode", "technician_hours"]),
    "abt_2_margin": dict(
        src="fact_production", join="sites", ids=["run_id", "site_id"], date="run_date",
        target="gross_margin_usd_per_bbl", task="regression",
        features=["country", "crude_grade", "crude_api_gravity", "crude_sulphur_pct", "throughput_bbl", "crude_cost_usd_per_bbl",
                  "furnace_outlet_temp_c", "column_pressure_bar", "energy_use_gj_per_bbl", "unplanned_downtime_hours",
                  "crack_spread_usd_per_bbl", "capacity_utilisation", "month", "quarter", "day_of_week", "sour_crude",
                  "heavy_crude", "had_downtime", "energy_x_sulphur", "crack_spread_7d_avg", "crude_cost_change_7d",
                  "margin_prev_7d_avg"],
        forbidden=["gasoline_yield_pct", "diesel_yield_pct", "jet_yield_pct", "fuel_oil_yield_pct"]),
    "abt_3_delivery": dict(
        src="fact_deliveries", join="sites_origin", ids=["trip_id", "origin_site_id"], date="dispatch_time",
        target="late_delivery", task="classification",
        features=["country", "site_type", "destination_region", "product", "volume_litres", "distance_km", "traffic_index",
                  "weather", "driver_experience_years", "truck_age_years", "promised_hours", "promised_speed_kmh",
                  "dispatch_hour", "dispatch_day", "dispatch_month", "is_weekend", "peak_hour", "long_haul", "old_truck",
                  "novice_driver", "bad_weather", "traffic_x_distance"],
        forbidden=["actual_arrival", "freight_cost_usd", "trip_status", "promised_arrival"]),
    "abt_4_quality": dict(
        src="fact_quality_tests", join="sites", ids=["test_id", "batch_id", "site_id"], date="test_date",
        target="off_spec", task="classification",
        features=["country", "site_type", "product", "product_group", "crude_grade", "feed_crude_sulphur_pct", "additive_dose_ppm",
                  "blend_temp_c", "storage_days", "water_content_ppm", "dose_deviation_ppm", "blend_temp_deviation",
                  "high_water", "long_storage", "sour_feed", "test_month", "test_quarter"],
        forbidden=["sulphur_ppm", "density_kg_m3", "octane_rating", "flash_point_c", "spec_result", "rework_cost_usd", "lab_name"]),
    "abt_5_inventory": dict(
        src="fact_inventory", join="sites", ids=["record_id", "site_id", "tank_id"], date="record_date",
        target=None, task="anomaly detection (no target)",
        features=["country", "site_type", "product", "opening_stock_kl", "receipts_kl", "dispatches_kl", "book_closing_kl",
                  "measured_closing_kl", "tank_temp_c", "days_since_meter_calibration", "gap_kl", "gap_pct", "abs_gap_pct",
                  "measured_to_book_ratio", "book_arithmetic_gap", "dispatch_rate_pct", "received_today", "gap_pct_mean_14d",
                  "gap_pct_std_14d", "gap_zscore", "record_month", "record_day"],
        forbidden=["anomaly_type", "is_anomaly"]),
}

abt, meta = {}, {}
for name, sp in SPEC.items():
    d = fe[sp["src"]].copy()
    if sp["join"] == "assets+sites":
        d = d.merge(assets, on="asset_id", how="left").merge(sites, on="site_id", how="left")
    elif sp["join"] == "sites":
        d = d.merge(sites, on="site_id", how="left")
    elif sp["join"] == "sites_origin":
        d = d.merge(sites.rename(columns={"site_id": "origin_site_id"}), on="origin_site_id", how="left")
    leaked = set(sp["features"]) & set(sp["forbidden"])
    assert not leaked, f"{name}: leakage columns in features {leaked}"
    cols = sp["ids"] + [sp["date"]] + ([sp["target"]] if sp["target"] else []) + sp["features"]
    d = d[cols]
    if sp["target"]:                                          # rows without a label cannot train or test a model
        n0 = len(d)
        d = d[d[sp["target"]].notna()]
        print(f"{name}: dropped {n0 - len(d)} rows with no target")
    d = d.sort_values(sp["date"]).reset_index(drop=True)      # chronological order, needed for the time-based split
    abt[name] = d
    cat = [c for c in sp["features"] if not pd.api.types.is_numeric_dtype(d[c])]
    meta[name] = {"ids": sp["ids"], "date": sp["date"], "target": sp["target"], "task": sp["task"],
                  "categorical": cat, "numeric": [c for c in sp["features"] if c not in cat]}

print("\nANALYTICAL BASE TABLES")
rows = []
for name, d in abt.items():
    m = meta[name]
    cut = d[m["date"]].quantile(0.8)
    rows.append({"table": name, "task": m["task"], "rows": len(d), "numeric_features": len(m["numeric"]),
                 "categorical_features": len(m["categorical"]), "target": m["target"] or "none",
                 "target_mean": round(d[m["target"]].mean(), 3) if m["target"] else "-",
                 "from": d[m["date"]].min().date(), "to": d[m["date"]].max().date(),
                 "train_rows_80%": int((d[m["date"]] <= cut).sum())})
pd.set_option("display.width", 220, "display.max_columns", 20)
print(pd.DataFrame(rows).to_string(index=False))

print("\nCOLUMNS LEFT OUT ON PURPOSE (leakage)")
for name, sp in SPEC.items():
    print(f"  {name:<18} {', '.join(sp['forbidden'])}")

print("\nMISSING VALUES LEFT IN FEATURES (top 5 per table; filled after the split)")
for name, d in abt.items():
    f = meta[name]["numeric"] + meta[name]["categorical"]
    miss = d[f].isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False).head(5)
    print(f"  {name:<18}", ", ".join(f"{k} {v}" for k, v in miss.items()) or "none")

print("\nPREVIEW: abt_3_delivery")
print(abt["abt_3_delivery"].head(3).T.head(14).to_string())

pd.to_pickle({"abt": abt, "meta": meta}, "meridian_abt.pkl")
for name, d in abt.items():
    d.to_csv(f"{name}.csv", index=False)
print("\nSaved: meridian_abt.pkl and 5 CSV files (abt_1_maintenance.csv ... abt_5_inventory.csv)")
```

---

## 10. Encode Categorical Columns

Text columns become 0/1 columns (one-hot encoding), so every model gets numbers only.

| Table | Categorical columns | Dummy columns created | Features before | Features after |
|---|---|---|---|---|
| `abt_1_maintenance` | 5 | 29 | 27 | 51 |
| `abt_2_margin` | 2 | 13 | 22 | 33 |
| `abt_3_delivery` | 5 | 30 | 22 | 47 |
| `abt_4_quality` | 5 | 25 | 17 | 37 |
| `abt_5_inventory` | 3 | 16 | 22 | 35 |

### Code: encode categorical columns

```python
# STEP 7: ENCODE CATEGORICAL COLUMNS (one-hot encoding)
# Each text category becomes its own 0/1 column, e.g. product -> product_PMS, product_AGO ...
# Identifier, date and target columns are left alone. Missing numbers are still untouched.
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    abt, meta
except NameError:
    _p = pd.read_pickle("meridian_abt.pkl")
    abt, meta = _p["abt"], _p["meta"]

enc, meta_enc, levels, rows = {}, {}, {}, []
for name, d in abt.items():
    m = meta[name]
    cat = m["categorical"]
    d = d.copy()
    d[cat] = d[cat].fillna("Unknown").astype(str)                  # safety: no blank category
    levels[name] = {c: sorted(d[c].unique()) for c in cat}         # keep the levels, the app needs them later
    e, groups = d.copy(), {}
    for c in cat:                                                   # one column at a time, so we know which dummies belong to it
        dm = pd.get_dummies(e[c], prefix=c, dtype=int)
        dm.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", x).strip("_") for x in dm.columns]     # clean column names
        groups[c] = list(dm.columns)
        e = pd.concat([e.drop(columns=c), dm], axis=1)
    dummies = [x for g in groups.values() for x in g]
    # checks: same rows, exactly one dummy switched on per category, nothing left as text except identifiers
    assert len(e) == len(d)
    for c, group in groups.items():
        assert (e[group].sum(axis=1) == 1).all(), f"{name}.{c}: dummy columns do not sum to 1"
    leftover = [c for c in e.columns if c not in m["ids"] and not pd.api.types.is_numeric_dtype(e[c]) and c != m["date"]]
    assert not leftover, f"text columns left: {leftover}"
    enc[name] = e
    meta_enc[name] = {**m, "dummies": dummies, "features": m["numeric"] + dummies}
    rows.append({"table": name, "categorical_columns": len(cat), "dummy_columns_created": len(dummies),
                 "feature_columns_before": len(m["numeric"]) + len(cat), "feature_columns_after": len(m["numeric"]) + len(dummies)})

pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 120)
print("ENCODING SUMMARY")
print(pd.DataFrame(rows).to_string(index=False))

print("\nCATEGORIES ENCODED")
for name, lv in levels.items():
    print(f"\n{name}")
    for c, v in lv.items():
        print(f"  {c:<20} {len(v):>2} levels: {', '.join(v)}")

print("\nPREVIEW: abt_4_quality after encoding (first 3 rows, first 14 columns)")
print(enc["abt_4_quality"].iloc[:3, :14].T.to_string())

pd.to_pickle({"enc": enc, "meta": meta_enc, "levels": levels}, "meridian_encoded.pkl")
print("\nSaved: meridian_encoded.pkl  (enc = 5 encoded tables, meta = column roles, levels = category lists)")
```

---

## 11. Separate X and Y

Features go to `X`, the target to `y`. ID and date columns are kept aside, not used as predictors, so they can be joined back to results later.

| Table | Rows | X columns | y | Positive rate |
|---|---|---|---|---|
| `abt_1_maintenance` | 8,477 | 51 | `breakdown_next_30d` | 4.6% |
| `abt_2_margin` | 4,339 | 33 | `gross_margin_usd_per_bbl` | mean 11.51, std 3.43 |
| `abt_3_delivery` | 5,712 | 47 | `late_delivery` | 10.5% |
| `abt_4_quality` | 4,152 | 37 | `off_spec` | 7.2% |
| `abt_5_inventory` | 10,182 | 35 | none | n/a |

### Code: separate X and y

```python
# STEP 8: SEPARATE X (features) AND y (target)
# X = the model inputs. y = what we predict. Identifiers and dates are set aside (not features) but kept
# in the same row order so Step 9 can split chronologically. Objective 5 has no y (unsupervised).
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
try:
    enc, meta_enc
except NameError:
    _p = pd.read_pickle("meridian_encoded.pkl")
    enc, meta_enc, levels = _p["enc"], _p["meta"], _p["levels"]

XY, rows = {}, []
for name, d in enc.items():
    m = meta_enc[name]
    X = d[m["features"]].copy()
    y = d[m["target"]].copy() if m["target"] else None
    side = d[m["ids"] + [m["date"]]].copy()                        # identifiers and date, kept aside
    # checks
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X), "X must be numeric"
    assert m["target"] not in X.columns and m["date"] not in X.columns and not set(m["ids"]) & set(X.columns)
    assert len(X) == len(side) and (y is None or len(y) == len(X))
    if y is not None:
        assert y.notna().all()
        y = y.astype(int) if m["task"] == "classification" else y.astype(float)
    XY[name] = {"X": X, "y": y, "side": side, "task": m["task"], "date": m["date"], "target": m["target"]}
    rows.append({"table": name, "task": m["task"], "rows": len(X), "X_columns": X.shape[1],
                 "y": m["target"] or "none", "kept_aside": ", ".join(m["ids"] + [m["date"]])})

pd.set_option("display.width", 220, "display.max_colwidth", 80)
print("X AND y")
print(pd.DataFrame(rows).to_string(index=False))

print("\nTARGET CHECK")
for name, v in XY.items():
    y = v["y"]
    if y is None:
        print(f"  {name:<18} no y (anomaly detection)")
    elif v["task"] == "classification":
        print(f"  {name:<18} class counts {y.value_counts().sort_index().to_dict()}  positive rate {y.mean():.1%}")
    else:
        print(f"  {name:<18} mean {y.mean():.2f}, min {y.min():.1f}, max {y.max():.1f}, std {y.std():.2f}")

print("\nFIRST 3 ROWS OF X AND y: abt_3_delivery")
print(XY["abt_3_delivery"]["X"].iloc[:3, :10].T.to_string())
print("y:", XY["abt_3_delivery"]["y"].iloc[:3].tolist())

print("\nMISSING VALUES STILL IN X (filled after the split)")
for name, v in XY.items():
    n = int(v["X"].isna().any(axis=1).sum())
    print(f"  {name:<18} {n:,} of {len(v['X']):,} rows have at least one blank ({n / len(v['X']):.0%})")

pd.to_pickle(XY, "meridian_xy.pkl")
print("\nSaved: meridian_xy.pkl  (XY['abt_1_maintenance']['X'], ['y'], ['side'] and so on)")
```

---

## 12. Train Test Split

**By date, not at random.** Models learn from the earlier 80% and are graded on the later 20%, the way they would be used in real life.

| Table | Train rows | Test rows | Train period | Test period |
|---|---|---|---|---|
| `abt_1_maintenance` | 6,711 | 1,412 | 2023-01 to 2024-07 | 2024-09 to 2024-12 |
| `abt_2_margin` | 3,475 | 864 | 2023-01-01 to 2024-08-08 | 2024-08-09 to 2024-12-31 |
| `abt_3_delivery` | 4,569 | 1,143 | 2023-01-01 to 2024-08-10 | 2024-08-10 to 2025-01-01 |
| `abt_4_quality` | 3,325 | 827 | 2023-01-01 to 2024-08-02 | 2024-08-03 to 2024-12-31 |
| `abt_5_inventory` | all 10,182 | none | 2024 | unsupervised, graded against an answer key |

Rare events are the limit here: the breakdown test set holds only **73** real breakdowns, the delivery set 124 late trips and the quality set 54 off-spec batches. That is why results carry 95% ranges later.

### Code: train test split

```python
# STEP 9: TRAIN / TEST SPLIT (chronological, never random)
# Models learn from the past (first ~80% of the timeline) and are tested on the future (last ~20%).
# A random split would let the model peek at the future, which is not how it will be used.
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
try:
    XY
except NameError:
    XY = pd.read_pickle("meridian_xy.pkl")

TRAIN_SHARE = 0.80
GAP_DAYS = {"abt_1_maintenance": 31}     # the label looks one month ahead, so the last training month is dropped
                                         # (its label would otherwise come from the test period)
split, rows = {}, []
for name, v in XY.items():
    X, y, side, dcol = v["X"], v["y"], v["side"], v["date"]
    dates = side[dcol]
    if y is None:                         # anomaly detection: no labels, so fit on all tank-days and judge later
        split[name] = {"X_all": X, "side_all": side, "task": v["task"]}
        rows.append({"table": name, "train_rows": "all", "test_rows": "-", "train_period": f"{dates.min().date()} to {dates.max().date()}",
                     "test_period": "-", "train_target": "-", "test_target": "-"})
        continue
    cut = dates.sort_values().iloc[int(len(dates) * TRAIN_SHARE) - 1]          # last date that belongs to training
    tr = dates <= cut - pd.Timedelta(days=GAP_DAYS.get(name, 0))
    te = dates > cut
    s = {"X_train": X[tr].copy(), "X_test": X[te].copy(), "y_train": y[tr].copy(), "y_test": y[te].copy(),
         "side_train": side[tr].copy(), "side_test": side[te].copy(), "task": v["task"], "cut_date": cut}
    assert dates[tr].max() < dates[te].min(), "train and test overlap in time"
    assert list(s["X_train"].columns) == list(s["X_test"].columns)
    split[name] = s
    fmt = (lambda a: f"{a.mean():.1%}") if v["task"] == "classification" else (lambda a: f"{a.mean():.2f}")
    rows.append({"table": name, "train_rows": f"{tr.sum():,}", "test_rows": f"{te.sum():,}",
                 "train_period": f"{dates[tr].min().date()} to {dates[tr].max().date()}",
                 "test_period": f"{dates[te].min().date()} to {dates[te].max().date()}",
                 "train_target": fmt(y[tr]), "test_target": fmt(y[te])})

pd.set_option("display.width", 230, "display.max_columns", 20)
print("CHRONOLOGICAL SPLIT")
print(pd.DataFrame(rows).to_string(index=False))

print("\nPOSITIVES AVAILABLE FOR LEARNING / TESTING (rare-event objectives)")
for name, s in split.items():
    if "y_train" in s and s["task"] == "classification":
        print(f"  {name:<18} train {int(s['y_train'].sum()):>4} positives of {len(s['y_train']):,}   |   test {int(s['y_test'].sum()):>4} positives of {len(s['y_test']):,}")

print("\nSHAPES")
for name, s in split.items():
    if "X_train" in s:
        print(f"  {name:<18} X_train {s['X_train'].shape}  X_test {s['X_test'].shape}")
    else:
        print(f"  {name:<18} X_all {s['X_all'].shape}  (no split, unsupervised)")

pd.to_pickle(split, "meridian_split.pkl")
print("\nSaved: meridian_split.pkl  (split['abt_3_delivery']['X_train'], ['y_test'] and so on)")
```

---

## 13. Feature Scaling of Numerical Columns

Missing values are filled with the **training** median, then numeric columns are standardised (mean 0, spread 1) with a `StandardScaler` fitted on training rows only. The 0/1 dummy columns are left as they are.

| Table | Features | Scaled | Left as 0/1 | Gaps filled (train / test) |
|---|---|---|---|---|
| `abt_1_maintenance` | 51 | 20 | 31 | 4,368 / 698 |
| `abt_2_margin` | 33 | 17 | 16 | 834 / 146 |
| `abt_3_delivery` | 47 | 11 | 36 | 962 / 226 |
| `abt_4_quality` | 37 | 9 | 28 | 1,146 / 242 |
| `abt_5_inventory` | 35 | 18 | 17 | 2,661 / n/a |

Sanity check: scaled training columns average 0.000 with spread 1.000. Test columns drift slightly (for example +0.098 for maintenance), which is expected, because they come from a later period.

### Code: feature scaling

```python
# STEP 10: FILL GAPS AND SCALE NUMERICAL COLUMNS
# Everything is learned from the TRAINING rows only, then applied unchanged to the test rows:
#   1) blanks in numeric columns -> training median; blanks in 0/1 columns -> training most-common value
#   2) numeric columns -> StandardScaler (mean 0, standard deviation 1)
#   3) 0/1 columns (dummies and flags) are left as they are
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    split
except NameError:
    split = pd.read_pickle("meridian_split.pkl")

scaled, summary = {}, []
for name, s in split.items():
    train = s["X_train"] if "X_train" in s else s["X_all"]            # objective 5 has no split: fit on all rows
    test = s.get("X_test")
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan) if test is not None else None

    binary = [c for c in train.columns if set(train[c].dropna().unique()) <= {0, 1, 0.0, 1.0}]
    numeric = [c for c in train.columns if c not in binary]

    fill = pd.concat([train[numeric].median(), train[binary].mode().iloc[0]])       # learned from training rows only
    gaps_train, gaps_test = int(train.isna().sum().sum()), int(test.isna().sum().sum()) if test is not None else 0
    train_f = train.fillna(fill)
    test_f = test.fillna(fill) if test is not None else None
    assert not train_f.isna().any().any() and (test_f is None or not test_f.isna().any().any())

    scaler = StandardScaler().fit(train_f[numeric])                    # fitted on training rows only
    train_s, test_s = train_f.copy(), (test_f.copy() if test_f is not None else None)
    train_s[numeric] = scaler.transform(train_f[numeric])
    if test_s is not None:
        test_s[numeric] = scaler.transform(test_f[numeric])

    out = {"X_train": train_s, "scaler": scaler, "fill_values": fill, "scaled_columns": numeric, "binary_columns": binary,
           "feature_names": list(train.columns)}
    if test_s is not None:
        out.update({"X_test": test_s, "y_train": s["y_train"], "y_test": s["y_test"],
                    "side_train": s["side_train"], "side_test": s["side_test"]})
    else:
        out["side_all"] = s["side_all"]
    scaled[name] = out
    summary.append({"Objective": name, "Features": len(train.columns), "Scaled numeric": len(numeric),
                    "Left as 0 or 1": len(binary), "Gaps filled (train / test)": f"{gaps_train:,} / {gaps_test:,}" if test is not None else f"{gaps_train:,} / -"})

pd.set_option("display.width", 200, "display.max_columns", 20)
print("FEATURE SCALING SUMMARY")
print(pd.DataFrame(summary).to_string(index=False))

print("\nSANITY CHECK: scaled numeric columns, training rows")
for name, o in scaled.items():
    z = o["X_train"][o["scaled_columns"]]
    print(f"  {name:<18} mean of means {z.mean().mean():+.4f}   mean of std {z.std().mean():.3f}", end="")
    if "X_test" in o:
        zt = o["X_test"][o["scaled_columns"]]
        print(f"   |  test: mean of means {zt.mean().mean():+.3f}, largest value {zt.abs().max().max():.1f}")
    else:
        print()

print("\nEXAMPLE: abt_3_delivery, 4 numeric columns before and after (first 3 training rows)")
before = split["abt_3_delivery"]["X_train"][["distance_km", "traffic_index", "promised_speed_kmh", "truck_age_years"]].head(3)
after = scaled["abt_3_delivery"]["X_train"][["distance_km", "traffic_index", "promised_speed_kmh", "truck_age_years"]].head(3)
print(pd.concat([before.add_suffix(" (raw)"), after.add_suffix(" (scaled)").round(2)], axis=1).T.to_string())

print("\nWHICH COLUMNS WERE SCALED: abt_1_maintenance")
print(", ".join(scaled["abt_1_maintenance"]["scaled_columns"]))

pd.to_pickle(scaled, "meridian_scaled.pkl")
print("\nSaved: meridian_scaled.pkl  (scaled['abt_1_maintenance']['X_train'], ['scaler'], ['fill_values'] and so on)")
```

---

## 14. Train Model

Four candidate models per supervised objective. The positive class is weighted so rare events are not ignored. The inventory objective trains three anomaly detectors on a 3% contamination setting.

| Objective | Candidates |
|---|---|
| 1, 3, 4 (classification) | Logistic Regression, Random Forest, Extra Trees, Gradient Boosting |
| 2 (regression) | Ridge Regression, Random Forest, Extra Trees, Gradient Boosting |
| 5 (anomaly) | Isolation Forest, Local Outlier Factor, One-Class SVM |

| Objective | Training rows | Features | Models trained |
|---|---|---|---|
| 1. Maintenance | 6,711 (299 breakdowns, 4.5%) | 51 | 4 classifiers |
| 2. Margin | 3,475 | 33 | 4 regressors |
| 3. Delivery | 4,569 (477 late, 10.4%) | 47 | 4 classifiers |
| 4. Off-spec | 3,325 (246 off-spec, 7.4%) | 37 | 4 classifiers |
| 5. Inventory | 10,182 | 16 numeric | 3 detectors |

Training scores only prove the models fitted. Tree models score 0.92 to 0.95 ROC AUC on training data and much lower on the test period, so training scores are never reported as results.

### Code: train model

#### Objective 1: Maintenance risk

```python
# STEP 11: TRAIN MODELS, OBJECTIVE 1 (predictive maintenance: will this asset break down next month?)
# Four candidate models are trained on the scaled TRAINING data only. The test set is not touched here.
# Breakdowns are rare (about 4.5%), so every model is told to give extra weight to the rare class.
import time
import warnings
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")

o = scaled["abt_1_maintenance"]
X_train, y_train = o["X_train"], o["y_train"]
print(f"Training data: {X_train.shape[0]:,} rows x {X_train.shape[1]} features, "
      f"{int(y_train.sum())} breakdowns ({y_train.mean():.1%})\n")

MODELS = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                            class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                        class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                    subsample=0.8, random_state=42),
}

obj1_models, rows = {}, []
for name, model in MODELS.items():
    t0 = time.time()
    if name == "Gradient Boosting":                    # this model has no class_weight option, so weight each row instead
        model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    else:
        model.fit(X_train, y_train)
    obj1_models[name] = model
    rows.append({"model": name, "fit_seconds": round(time.time() - t0, 1),
                 "training_ROC_AUC (sanity check only)": round(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]), 3)})

print("TRAINED MODELS")
print(pd.DataFrame(rows).to_string(index=False))
print("\nThese training scores only prove the models fitted. They are NOT the real result:")
print("the real result comes from the unseen test months in Step 12 and Step 14.")

joblib.dump({"models": obj1_models, "feature_names": o["feature_names"]}, "obj1_models.joblib", compress=3)
print("\nSaved: obj1_models.joblib  (the 4 trained models and the feature order)")
```

#### Objective 2: Margin forecast

```python
# STEP 11: TRAIN MODELS, OBJECTIVE 2 (refinery margin: predict gross margin in USD per barrel)
# Four candidate regression models are trained on the scaled TRAINING data only. The test set is not touched here.
import time
import warnings
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")

o = scaled["abt_2_margin"]
X_train, y_train = o["X_train"], o["y_train"]
print(f"Training data: {X_train.shape[0]:,} refinery-days x {X_train.shape[1]} features; "
      f"margin averages {y_train.mean():.2f} USD/bbl (std {y_train.std():.2f})\n")

MODELS = {
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42),
}

obj2_models, rows = {}, []
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    obj2_models[name] = model
    p = model.predict(X_train)
    rows.append({"model": name, "fit_seconds": round(time.time() - t0, 1),
                 "training_R2 (sanity check only)": round(r2_score(y_train, p), 3),
                 "training_MAE_USD_per_bbl": round(mean_absolute_error(y_train, p), 2)})

print("TRAINED MODELS")
print(pd.DataFrame(rows).to_string(index=False))
print("\nThese training scores only prove the models fitted. They are NOT the real result:")
print("the real result comes from the unseen test period in Step 12 and Step 14.")

joblib.dump({"models": obj2_models, "feature_names": o["feature_names"]}, "obj2_models.joblib", compress=3)
print("\nSaved: obj2_models.joblib  (the 4 trained models and the feature order)")
```

#### Objective 3: Delivery risk

```python
# STEP 11: TRAIN MODELS, OBJECTIVE 3 (late delivery: will this fuel trip arrive more than 30 minutes late?)
# Four candidate models are trained on the scaled TRAINING data only. The test set is not touched here.
# Late trips are rare (about 10%), so every model is told to give extra weight to the rare class.
import time
import warnings
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")

o = scaled["abt_3_delivery"]
X_train, y_train = o["X_train"], o["y_train"]
print(f"Training data: {X_train.shape[0]:,} trips x {X_train.shape[1]} features, "
      f"{int(y_train.sum())} late trips ({y_train.mean():.1%})\n")

MODELS = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                            class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                        class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                    subsample=0.8, random_state=42),
}

obj3_models, rows = {}, []
for name, model in MODELS.items():
    t0 = time.time()
    if name == "Gradient Boosting":                    # no class_weight option, so weight each row instead
        model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    else:
        model.fit(X_train, y_train)
    obj3_models[name] = model
    rows.append({"model": name, "fit_seconds": round(time.time() - t0, 1),
                 "training_ROC_AUC (sanity check only)": round(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]), 3)})

print("TRAINED MODELS")
print(pd.DataFrame(rows).to_string(index=False))
print("\nThese training scores only prove the models fitted. They are NOT the real result:")
print("the real result comes from the unseen test period in Step 12 and Step 14.")

joblib.dump({"models": obj3_models, "feature_names": o["feature_names"]}, "obj3_models.joblib", compress=3)
print("\nSaved: obj3_models.joblib  (the 4 trained models and the feature order)")
```

#### Objective 4: Off-spec risk

```python
# STEP 11: TRAIN MODELS, OBJECTIVE 4 (off-spec product: will this batch fail the lab specification?)
# Four candidate models are trained on the scaled TRAINING data only. The test set is not touched here.
# Off-spec batches are rare (about 7%), so every model is told to give extra weight to the rare class.
import time
import warnings
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")

o = scaled["abt_4_quality"]
X_train, y_train = o["X_train"], o["y_train"]
print(f"Training data: {X_train.shape[0]:,} batches x {X_train.shape[1]} features, "
      f"{int(y_train.sum())} off-spec batches ({y_train.mean():.1%})\n")

MODELS = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                            class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                        class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                    subsample=0.8, random_state=42),
}

obj4_models, rows = {}, []
for name, model in MODELS.items():
    t0 = time.time()
    if name == "Gradient Boosting":                    # no class_weight option, so weight each row instead
        model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    else:
        model.fit(X_train, y_train)
    obj4_models[name] = model
    rows.append({"model": name, "fit_seconds": round(time.time() - t0, 1),
                 "training_ROC_AUC (sanity check only)": round(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]), 3)})

print("TRAINED MODELS")
print(pd.DataFrame(rows).to_string(index=False))
print("\nThese training scores only prove the models fitted. They are NOT the real result:")
print("the real result comes from the unseen test period in Step 12 and Step 14.")

joblib.dump({"models": obj4_models, "feature_names": o["feature_names"]}, "obj4_models.joblib", compress=3)
print("\nSaved: obj4_models.joblib  (the 4 trained models and the feature order)")
```

#### Objective 5: Inventory check

```python
# STEP 11: TRAIN MODELS, OBJECTIVE 5 (inventory anomalies: find tank-days with unusual measured-vs-book volume)
# There is no target, so this is unsupervised: each model learns what a NORMAL tank-day looks like and
# scores how far every tank-day is from normal. The anomaly answer key is NOT used here (only after, in Step 14).
# Inputs: the numeric, scaled inventory features only (the 0/1 product, country and site-type columns describe
# where the tank is, not whether its stock is behaving strangely).
import time
import warnings
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")

o = scaled["abt_5_inventory"]
cols = [c for c in o["scaled_columns"] if c not in ("record_month", "record_day")]     # numeric, filled and standardised; calendar columns say nothing about stock behaviour
X = o["X_train"][cols]                                  # no split here: fit on all tank-days
EXPECTED = 0.03                                         # assumption: about 3% of tank-days are abnormal (EDA: 3.3% have a gap above 1%)
print(f"Training data: {X.shape[0]:,} tank-days x {X.shape[1]} numeric features")
print("Features:", ", ".join(cols), "\n")

MODELS = {
    "Isolation Forest": IsolationForest(n_estimators=300, contamination=EXPECTED, random_state=42, n_jobs=-1),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=35, contamination=EXPECTED, novelty=True, n_jobs=-1),
    "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=EXPECTED),
}

obj5_models, rows = {}, []
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X)
    obj5_models[name] = model
    rows.append({"model": name, "fit_seconds": round(time.time() - t0, 1),
                 "tank_days_flagged_on_training_data": int((model.predict(X) == -1).sum()),
                 "flagged_%": f"{(model.predict(X) == -1).mean():.1%}"})

print("TRAINED MODELS")
print(pd.DataFrame(rows).to_string(index=False))
print("\nThese counts only show the models fitted and roughly respect the 3% setting.")
print("Whether the flagged tank-days are REAL problems is checked in Step 14 against the answer key.")

joblib.dump({"models": obj5_models, "feature_names": list(cols)}, "obj5_models.joblib", compress=3)
print("\nSaved: obj5_models.joblib  (the 3 trained models and the feature list)")
```

---

## 15. Predict X_test

Each model scores the unseen test period. For classifiers I keep the **probability**, which is what the business ranks on.

| Objective | Test rows | Test period | What the predictions show |
|---|---|---|---|
| 1. Maintenance | 1,412 asset-months (73 breakdowns) | 2024-09 to 2024-12 | At a 0.5 cut-off Logistic Regression flags 39.6% of rows, Random Forest 17.8% |
| 2. Margin | 864 refinery-days | 2024-08-09 to 2024-12-31 | Actual average 9.70 USD/bbl, Ridge predicts 9.62 |
| 3. Delivery | 1,143 trips (124 late) | 2024-08-10 to 2025-01-01 | Probabilities per trip for all four models |
| 4. Off-spec | 827 batches (54 off-spec) | 2024-08-03 to 2024-12-31 | Probabilities per batch for all four models |
| 5. Inventory | 10,182 tank-days | 2024 | Each detector flags about 3%. 65 tank-days are flagged by all three models |

### Code: predict X_test

#### Objective 1: Maintenance risk

```python
# STEP 12: PREDICT X_TEST, OBJECTIVE 1 (predictive maintenance)
# Each trained model scores the unseen test months. We keep, for every asset-month: the probability of a
# breakdown next month, and the yes/no prediction at the default 0.5 cut-off. No scoring or judging yet (Step 14).
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")
try:
    obj1_models
except NameError:
    _b = joblib.load("obj1_models.joblib")
    obj1_models = _b["models"]

o = scaled["abt_1_maintenance"]
X_test, y_test, side_test = o["X_test"], o["y_test"], o["side_test"]
print(f"Test data: {len(X_test):,} asset-months, {side_test['reading_month'].min().date()} to {side_test['reading_month'].max().date()}\n")

pred1 = side_test[["reading_id", "asset_id", "reading_month"]].copy()
pred1["actual_breakdown_next_30d"] = y_test.values
for name, model in obj1_models.items():
    assert list(X_test.columns) == list(model.feature_names_in_), "feature order differs from training"
    tag = name.lower().replace(" ", "_")
    pred1[f"{tag}_probability"] = model.predict_proba(X_test)[:, 1].round(4)     # chance of a breakdown next month
    pred1[f"{tag}_prediction"] = model.predict(X_test)                           # 1 = breakdown expected, 0 = no

print("HOW MANY ASSET-MONTHS EACH MODEL FLAGS (actual breakdowns in the test period: "
      f"{int(y_test.sum())} of {len(y_test):,})")
rows = []
for name in obj1_models:
    tag = name.lower().replace(" ", "_")
    rows.append({"model": name, "flagged_as_breakdown": int(pred1[f"{tag}_prediction"].sum()),
                 "flagged_%": f"{pred1[f'{tag}_prediction'].mean():.1%}",
                 "average_probability": round(pred1[f"{tag}_probability"].mean(), 3),
                 "highest_probability": round(pred1[f"{tag}_probability"].max(), 3)})
pd.set_option("display.width", 220, "display.max_columns", 20)
print(pd.DataFrame(rows).to_string(index=False))

print("\nFIRST 8 TEST PREDICTIONS")
show = ["asset_id", "reading_month", "actual_breakdown_next_30d"] + [f"{n.lower().replace(' ', '_')}_probability" for n in obj1_models]
print(pred1[show].head(8).to_string(index=False))

print("\nTHE 5 ASSET-MONTHS WITH THE HIGHEST RANDOM FOREST RISK")
print(pred1.sort_values("random_forest_probability", ascending=False)[["asset_id", "reading_month", "random_forest_probability",
      "actual_breakdown_next_30d"]].head(5).to_string(index=False))

pred1.to_csv("obj1_test_predictions.csv", index=False)
pd.to_pickle(pred1, "obj1_test_predictions.pkl")
print("\nSaved: obj1_test_predictions.csv  (pred1 holds the probabilities and predictions of all 4 models)")
```

#### Objective 2: Margin forecast

```python
# STEP 12: PREDICT X_TEST, OBJECTIVE 2 (refinery margin)
# Each trained model predicts the gross margin (USD per barrel) for every unseen refinery-day in the test period.
# We keep the predictions next to the actual margin. No scoring or judging yet (Step 14).
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")
try:
    obj2_models
except NameError:
    obj2_models = joblib.load("obj2_models.joblib")["models"]

o = scaled["abt_2_margin"]
X_test, y_test, side_test = o["X_test"], o["y_test"], o["side_test"]
print(f"Test data: {len(X_test):,} refinery-days, {side_test['run_date'].min().date()} to {side_test['run_date'].max().date()}\n")

pred2 = side_test[["run_id", "site_id", "run_date"]].copy()
pred2["actual_margin"] = y_test.values
for name, model in obj2_models.items():
    assert list(X_test.columns) == list(model.feature_names_in_), "feature order differs from training"
    pred2[f"{name.lower().replace(' ', '_')}_prediction"] = model.predict(X_test).round(3)

cols = [c for c in pred2.columns if c.endswith("_prediction")]
print("AVERAGE MARGIN: ACTUAL VS PREDICTED (USD per barrel)")
rows = [{"": "Actual", "average": round(pred2["actual_margin"].mean(), 2), "lowest": round(pred2["actual_margin"].min(), 1),
         "highest": round(pred2["actual_margin"].max(), 1)}]
for c in cols:
    rows.append({"": c.replace("_prediction", "").replace("_", " ").title(), "average": round(pred2[c].mean(), 2),
                 "lowest": round(pred2[c].min(), 1), "highest": round(pred2[c].max(), 1)})
pd.set_option("display.width", 220, "display.max_columns", 20)
print(pd.DataFrame(rows).to_string(index=False))

print("\nFIRST 8 TEST PREDICTIONS")
print(pred2[["site_id", "run_date", "actual_margin"] + cols].head(8).to_string(index=False))

print("\nPREDICTED vs ACTUAL AVERAGE BY MONTH (Random Forest)")
m = pred2.assign(month=pred2["run_date"].dt.to_period("M").astype(str)).groupby("month")[["actual_margin", "random_forest_prediction"]].mean().round(2)
print(m.to_string())

pred2.to_csv("obj2_test_predictions.csv", index=False)
pd.to_pickle(pred2, "obj2_test_predictions.pkl")
print("\nSaved: obj2_test_predictions.csv  (pred2 holds the predictions of all 4 models next to the actual margin)")
```

#### Objective 3: Delivery risk

```python
# STEP 12: PREDICT X_TEST, OBJECTIVE 3 (late delivery)
# Each trained model scores every unseen trip in the test period: the probability of arriving more than
# 30 minutes late, and the yes/no prediction at the default 0.5 cut-off. No scoring or judging yet (Step 14).
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")
try:
    obj3_models
except NameError:
    obj3_models = joblib.load("obj3_models.joblib")["models"]

o = scaled["abt_3_delivery"]
X_test, y_test, side_test = o["X_test"], o["y_test"], o["side_test"]
print(f"Test data: {len(X_test):,} trips, {side_test['dispatch_time'].min().date()} to {side_test['dispatch_time'].max().date()}\n")

pred3 = side_test[["trip_id", "origin_site_id", "dispatch_time"]].copy()
pred3["actual_late"] = y_test.values
for name, model in obj3_models.items():
    assert list(X_test.columns) == list(model.feature_names_in_), "feature order differs from training"
    tag = name.lower().replace(" ", "_")
    pred3[f"{tag}_probability"] = model.predict_proba(X_test)[:, 1].round(4)
    pred3[f"{tag}_prediction"] = model.predict(X_test)

print(f"HOW MANY TRIPS EACH MODEL FLAGS AS LATE (actual late trips in the test period: {int(y_test.sum())} of {len(y_test):,})")
rows = []
for name in obj3_models:
    tag = name.lower().replace(" ", "_")
    rows.append({"model": name, "flagged_as_late": int(pred3[f"{tag}_prediction"].sum()),
                 "flagged_%": f"{pred3[f'{tag}_prediction'].mean():.1%}",
                 "average_probability": round(pred3[f"{tag}_probability"].mean(), 3),
                 "highest_probability": round(pred3[f"{tag}_probability"].max(), 3)})
pd.set_option("display.width", 220, "display.max_columns", 20)
print(pd.DataFrame(rows).to_string(index=False))

print("\nFIRST 8 TEST PREDICTIONS")
show = ["trip_id", "dispatch_time", "actual_late"] + [f"{n.lower().replace(' ', '_')}_probability" for n in obj3_models]
print(pred3[show].head(8).to_string(index=False))

print("\nTHE 5 TRIPS WITH THE HIGHEST LOGISTIC REGRESSION RISK")
print(pred3.sort_values("logistic_regression_probability", ascending=False)[
    ["trip_id", "dispatch_time", "logistic_regression_probability", "actual_late"]].head(5).to_string(index=False))

pred3.to_csv("obj3_test_predictions.csv", index=False)
pd.to_pickle(pred3, "obj3_test_predictions.pkl")
print("\nSaved: obj3_test_predictions.csv  (pred3 holds the probabilities and predictions of all 4 models)")
```

#### Objective 4: Off-spec risk

```python
# STEP 12: PREDICT X_TEST, OBJECTIVE 4 (off-spec product)
# Each trained model scores every unseen batch in the test period: the probability of failing the lab
# specification (off-spec), and the yes/no prediction at the default 0.5 cut-off. No scoring or judging yet (Step 14).
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")
try:
    obj4_models
except NameError:
    obj4_models = joblib.load("obj4_models.joblib")["models"]

o = scaled["abt_4_quality"]
X_test, y_test, side_test = o["X_test"], o["y_test"], o["side_test"]
print(f"Test data: {len(X_test):,} batches, {side_test['test_date'].min().date()} to {side_test['test_date'].max().date()}\n")

pred4 = side_test[["test_id", "batch_id", "site_id", "test_date"]].copy()
pred4["actual_off_spec"] = y_test.values
for name, model in obj4_models.items():
    assert list(X_test.columns) == list(model.feature_names_in_), "feature order differs from training"
    tag = name.lower().replace(" ", "_")
    pred4[f"{tag}_probability"] = model.predict_proba(X_test)[:, 1].round(4)
    pred4[f"{tag}_prediction"] = model.predict(X_test)

print(f"HOW MANY BATCHES EACH MODEL FLAGS AS OFF-SPEC (actual off-spec batches in the test period: {int(y_test.sum())} of {len(y_test):,})")
rows = []
for name in obj4_models:
    tag = name.lower().replace(" ", "_")
    rows.append({"model": name, "flagged_as_off_spec": int(pred4[f"{tag}_prediction"].sum()),
                 "flagged_%": f"{pred4[f'{tag}_prediction'].mean():.1%}",
                 "average_probability": round(pred4[f"{tag}_probability"].mean(), 3),
                 "highest_probability": round(pred4[f"{tag}_probability"].max(), 3)})
pd.set_option("display.width", 220, "display.max_columns", 20)
print(pd.DataFrame(rows).to_string(index=False))

print("\nFIRST 8 TEST PREDICTIONS")
show = ["batch_id", "test_date", "actual_off_spec"] + [f"{n.lower().replace(' ', '_')}_probability" for n in obj4_models]
print(pred4[show].head(8).to_string(index=False))

print("\nTHE 5 BATCHES WITH THE HIGHEST LOGISTIC REGRESSION RISK")
print(pred4.sort_values("logistic_regression_probability", ascending=False)[
    ["batch_id", "test_date", "logistic_regression_probability", "actual_off_spec"]].head(5).to_string(index=False))

pred4.to_csv("obj4_test_predictions.csv", index=False)
pd.to_pickle(pred4, "obj4_test_predictions.pkl")
print("\nSaved: obj4_test_predictions.csv  (pred4 holds the probabilities and predictions of all 4 models)")
```

#### Objective 5: Inventory check

```python
# STEP 12: SCORE THE TANK-DAYS, OBJECTIVE 5 (inventory anomalies)
# There is no separate X_test here: the detectors have no labels, so they were fitted on every tank-day
# and now score those same tank-days. For each one we keep an anomaly score (higher = stranger) and a flag.
# The answer key is still not used (Step 14).
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")
try:
    obj5_models
except NameError:
    _b = joblib.load("obj5_models.joblib")
    obj5_models, _cols = _b["models"], _b["feature_names"]

o = scaled["abt_5_inventory"]
cols = [c for c in o["scaled_columns"] if c not in ("record_month", "record_day")]
X, side = o["X_train"][cols], o["side_all"]
print(f"Scoring {len(X):,} tank-days, {side['record_date'].min().date()} to {side['record_date'].max().date()}\n")

pred5 = side[["record_id", "site_id", "tank_id", "record_date"]].copy()
for name, model in obj5_models.items():
    assert list(X.columns) == list(model.feature_names_in_), "feature order differs from training"
    tag = name.lower().replace(" ", "_").replace("-", "_")
    pred5[f"{tag}_score"] = (-model.decision_function(X)).round(4)         # higher = more unusual
    pred5[f"{tag}_flag"] = (model.predict(X) == -1).astype(int)             # 1 = flagged as anomaly
flags = [c for c in pred5.columns if c.endswith("_flag")]
pred5["models_agreeing"] = pred5[flags].sum(axis=1)                         # how many of the 3 models flag it

# add the raw gap for readability (from the unscaled table of Step 6)
try:
    raw = pd.read_pickle("meridian_encoded.pkl")["enc"]["abt_5_inventory"][["record_id", "gap_pct"]]
    pred5 = pred5.merge(raw, on="record_id", how="left")
except Exception:
    pass

print("TANK-DAYS FLAGGED BY EACH MODEL")
LABEL = {"isolation_forest": "Isolation Forest", "local_outlier_factor": "Local Outlier Factor", "one_class_svm": "One-Class SVM"}
rows = [{"model": LABEL[f.replace("_flag", "")], "flagged": int(pred5[f].sum()), "flagged_%": f"{pred5[f].mean():.1%}"} for f in flags]
pd.set_option("display.width", 220, "display.max_columns", 20)
print(pd.DataFrame(rows).to_string(index=False))
print("\nAGREEMENT BETWEEN MODELS")
print(pred5["models_agreeing"].value_counts().sort_index().rename({0: "0 models (normal)", 1: "1 model", 2: "2 models", 3: "all 3 models"}).to_frame("tank_days").to_string())

print("\nTHE 10 MOST UNUSUAL TANK-DAYS (average rank across the 3 models)")
rk = pd.concat([pred5[c.replace("_flag", "_score")].rank(ascending=False) for c in flags], axis=1).mean(axis=1)
cols_show = ["record_id", "tank_id", "record_date", "models_agreeing"] + (["gap_pct"] if "gap_pct" in pred5 else [])
print(pred5.loc[rk.sort_values().index[:10], cols_show].round(2).to_string(index=False))

pred5.to_csv("obj5_scores.csv", index=False)
pd.to_pickle(pred5, "obj5_scores.pkl")
print("\nSaved: obj5_scores.csv  (pred5 holds each model's score and flag for every tank-day)")
```

---

## 16. Compare X and Y

Which inputs move with the outcome? For each objective I compare Random Forest feature importance with the direction of each feature's relationship to the target.

| Objective | Top drivers | Direction |
|---|---|---|
| 1. Maintenance | `lube_avg_3m`, `vib_avg_3m`, `avg_vibration_mm_s`, `temp_avg_3m` | Hotter, shakier, worse-lubricated assets break down more. Average predicted risk 0.47 for assets that failed vs 0.30 for those that did not |
| 2. Margin | `crack_spread_usd_per_bbl` (importance 0.69), `sour_crude`, `crude_sulphur_pct`, `margin_prev_7d_avg` | Higher crack spread, higher margin (correlation 0.83). Sour crude, lower margin. Predicted vs actual correlation on test days: 0.92 |
| 3. Delivery | `traffic_x_distance`, `traffic_index`, `distance_km`, `promised_speed_kmh` | Heavy traffic over long distance and storms raise late risk. Average risk 0.64 for late trips vs 0.37 for on-time |
| 4. Off-spec | `water_content_ppm` (0.24), `high_water`, `feed_crude_sulphur_pct`, `storage_days` | Wetter, higher-sulphur, longer-stored batches fail more. Average risk 0.56 vs 0.36 |
| 5. Inventory | Gap features | Models call 189 tank-days (1.9%) unusual. Only 24% of those have a big gap, against 2.9% for normal tank-days |

### Charts

<p align="center">
  <img src="images/obj1_features.png" alt="Objective 1: maintenance risk drivers" width="100%">
</p>
<p align="center"><em>Objective 1: maintenance risk drivers</em></p>

<p align="center">
  <img src="images/obj2_features.png" alt="Objective 2: margin drivers and predicted vs actual" width="100%">
</p>
<p align="center"><em>Objective 2: margin drivers and predicted vs actual</em></p>

<p align="center">
  <img src="images/obj3_features.png" alt="Objective 3: delivery risk drivers" width="100%">
</p>
<p align="center"><em>Objective 3: delivery risk drivers</em></p>

<p align="center">
  <img src="images/obj4_features.png" alt="Objective 4: off-spec risk drivers" width="100%">
</p>
<p align="center"><em>Objective 4: off-spec risk drivers</em></p>

<p align="center">
  <img src="images/obj5_features.png" alt="Objective 5: unusual tank-days" width="100%">
</p>
<p align="center"><em>Objective 5: unusual tank-days</em></p>

### Code: compare X and y

#### Objective 1: Maintenance risk

```python
# STEP 13: COMPARE FEATURES (X) AND TARGET (y), OBJECTIVE 1
# Which inputs move the breakdown risk, in which direction, and do the predicted risks separate real breakdowns?
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")
try:
    obj1_models
except NameError:
    obj1_models = joblib.load("obj1_models.joblib")["models"]
try:
    pred1
except NameError:
    pred1 = pd.read_pickle("obj1_test_predictions.pkl")

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
C1, C2, C3 = "#264653", "#E76F51", "#2A9D8F"
o = scaled["abt_1_maintenance"]
Xtr, ytr = o["X_train"], o["y_train"]
rf, lr = obj1_models["Random Forest"], obj1_models["Logistic Regression"]

imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
coef = pd.Series(lr.coef_[0], index=Xtr.columns)
corr = Xtr.corrwith(ytr).dropna()                       # correlation of each feature with the 0/1 target (training rows)
top_numeric = [c for c in imp.index if c in o["scaled_columns"]][:3]
nice = lambda s: s.replace("_", " ")

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
t = imp.head(12)[::-1]
ax[0, 0].barh([nice(c) for c in t.index], t.values, color=C1)
ax[0, 0].set(title="What the Random Forest relies on most", xlabel="importance")

t = corr.reindex(corr.abs().sort_values(ascending=False).head(12).index)[::-1]
ax[0, 1].barh([nice(c) for c in t.index], t.values, color=[C2 if v > 0 else C3 for v in t.values])
ax[0, 1].axvline(0, color="#333", lw=.8)
ax[0, 1].set(title="Direction of the link with breakdowns", xlabel="correlation with the target: right (orange) = higher risk, left (green) = lower risk")

for c in top_numeric:                                    # breakdown rate across the low-to-high range of each key feature
    q = pd.qcut(Xtr[c], 5, duplicates="drop", labels=False)
    r = ytr.groupby(q).mean() * 100
    ax[1, 0].plot(range(1, len(r) + 1), r.values, marker="o", lw=2, label=nice(c))
ax[1, 0].axhline(ytr.mean() * 100, color="#888", ls="--", label="overall rate")
ax[1, 0].set(title="Breakdown rate from lowest to highest fifth of each key feature", xlabel="fifth of the feature (1 = lowest, 5 = highest)",
             ylabel="% of asset-months with a breakdown next month"); ax[1, 0].legend(frameon=False)

for lab, col, txt in [(0, C1, "no breakdown next month"), (1, C2, "breakdown next month")]:
    ax[1, 1].hist(pred1.loc[pred1["actual_breakdown_next_30d"] == lab, "random_forest_probability"], bins=30, density=True,
                  alpha=.65, color=col, label=txt)
ax[1, 1].set(title="Predicted risk on the test months, by what really happened", xlabel="Random Forest breakdown probability", ylabel="density")
ax[1, 1].legend(frameon=False)
fig.suptitle("Objective 1: comparing features and target", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("compare_x_y_obj1.png", dpi=140, bbox_inches="tight"); plt.show()

# plain-language summary table
tab = pd.DataFrame({"RF_importance": imp, "correlation_with_target": corr.reindex(imp.index)}).head(10).round(3)
tab["direction"] = np.where(tab["correlation_with_target"] > 0, "higher value, higher risk", "higher value, lower risk")
pd.set_option("display.width", 200)
print("TOP 10 FEATURES")
print(tab.to_string())
sep = pred1.groupby("actual_breakdown_next_30d")["random_forest_probability"].mean()
print(f"\nAverage Random Forest risk: {sep[1]:.2f} for asset-months that broke down vs {sep[0]:.2f} for those that did not.")
print("Saved: compare_x_y_obj1.png")
```

#### Objective 2: Margin forecast

```python
# STEP 13: COMPARE FEATURES (X) AND TARGET (y), OBJECTIVE 2 (refinery margin)
# Which inputs move the margin, in which direction, and how close are the predictions to the real margin?
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")
try:
    obj2_models
except NameError:
    obj2_models = joblib.load("obj2_models.joblib")["models"]
try:
    pred2
except NameError:
    pred2 = pd.read_pickle("obj2_test_predictions.pkl")

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
C1, C2, C3 = "#264653", "#E76F51", "#2A9D8F"
o = scaled["abt_2_margin"]
Xtr, ytr = o["X_train"], o["y_train"]
rf = obj2_models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
corr = Xtr.corrwith(ytr).dropna()
top_numeric = [c for c in imp.index if c in o["scaled_columns"]][:3]
nice = lambda s: s.replace("_", " ")

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
t = imp.head(12)[::-1]
ax[0, 0].barh([nice(c) for c in t.index], t.values, color=C1)
ax[0, 0].set(title="What the Random Forest relies on most", xlabel="importance")

t = corr.reindex(corr.abs().sort_values(ascending=False).head(12).index)[::-1]
ax[0, 1].barh([nice(c) for c in t.index], t.values, color=[C3 if v > 0 else C2 for v in t.values])
ax[0, 1].axvline(0, color="#333", lw=.8)
ax[0, 1].set(title="Direction of the link with margin", xlabel="correlation with margin: right (green) = higher margin, left (orange) = lower margin")

for c in top_numeric:
    q = pd.qcut(Xtr[c], 5, duplicates="drop", labels=False)
    r = ytr.groupby(q).mean()
    ax[1, 0].plot(range(1, len(r) + 1), r.values, marker="o", lw=2, label=nice(c))
ax[1, 0].axhline(ytr.mean(), color="#888", ls="--", label="overall average")
ax[1, 0].set(title="Average margin from lowest to highest fifth of each key feature", xlabel="fifth of the feature (1 = lowest, 5 = highest)",
             ylabel="average margin (USD per barrel)"); ax[1, 0].legend(frameon=False)

ax[1, 1].scatter(pred2["actual_margin"], pred2["random_forest_prediction"], s=10, alpha=.4, color=C1)
lo, hi = pred2["actual_margin"].min(), pred2["actual_margin"].max()
ax[1, 1].plot([lo, hi], [lo, hi], "--", color=C2, lw=2, label="perfect prediction")
ax[1, 1].set(title="Test period: predicted vs actual margin (Random Forest)", xlabel="actual margin (USD per barrel)",
             ylabel="predicted margin (USD per barrel)"); ax[1, 1].legend(frameon=False)
fig.suptitle("Objective 2: comparing features and target", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("compare_x_y_obj2.png", dpi=140, bbox_inches="tight"); plt.show()

tab = pd.DataFrame({"RF_importance": imp, "correlation_with_margin": corr.reindex(imp.index)}).head(10).round(3)
tab["direction"] = np.where(tab["correlation_with_margin"] > 0, "higher value, higher margin", "higher value, lower margin")
pd.set_option("display.width", 200)
print("TOP 10 FEATURES")
print(tab.to_string())
print(f"\nCorrelation between predicted and actual margin on the test period: {np.corrcoef(pred2['actual_margin'], pred2['random_forest_prediction'])[0, 1]:.2f}")
print("Saved: compare_x_y_obj2.png")
```

#### Objective 3: Delivery risk

```python
# STEP 13: COMPARE FEATURES (X) AND TARGET (y), OBJECTIVE 3 (late delivery)
# Which trip characteristics move the chance of arriving late, in which direction, and do the predicted risks
# separate the trips that really were late?
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")
try:
    obj3_models
except NameError:
    obj3_models = joblib.load("obj3_models.joblib")["models"]
try:
    pred3
except NameError:
    pred3 = pd.read_pickle("obj3_test_predictions.pkl")

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
C1, C2, C3 = "#264653", "#E76F51", "#2A9D8F"
o = scaled["abt_3_delivery"]
Xtr, ytr = o["X_train"], o["y_train"]
rf = obj3_models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
corr = Xtr.corrwith(ytr).dropna()
top_numeric = [c for c in imp.index if c in o["scaled_columns"]][:3]
nice = lambda s: s.replace("_", " ")

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
t = imp.head(12)[::-1]
ax[0, 0].barh([nice(c) for c in t.index], t.values, color=C1)
ax[0, 0].set(title="What the Random Forest relies on most", xlabel="importance")

t = corr.reindex(corr.abs().sort_values(ascending=False).head(12).index)[::-1]
ax[0, 1].barh([nice(c) for c in t.index], t.values, color=[C2 if v > 0 else C3 for v in t.values])
ax[0, 1].axvline(0, color="#333", lw=.8)
ax[0, 1].set(title="Direction of the link with late delivery", xlabel="correlation with the target: right (orange) = more likely late, left (green) = less likely")

for c in top_numeric:
    q = pd.qcut(Xtr[c], 5, duplicates="drop", labels=False)
    r = ytr.groupby(q).mean() * 100
    ax[1, 0].plot(range(1, len(r) + 1), r.values, marker="o", lw=2, label=nice(c))
ax[1, 0].axhline(ytr.mean() * 100, color="#888", ls="--", label="overall rate")
ax[1, 0].set(title="Late rate from lowest to highest fifth of each key feature", xlabel="fifth of the feature (1 = lowest, 5 = highest)",
             ylabel="% of trips that were late"); ax[1, 0].legend(frameon=False)

for lab, col, txt in [(0, C1, "on time"), (1, C2, "late")]:
    ax[1, 1].hist(pred3.loc[pred3["actual_late"] == lab, "logistic_regression_probability"], bins=30, density=True, alpha=.65, color=col, label=txt)
ax[1, 1].set(title="Predicted risk on the test period, by what really happened", xlabel="Logistic Regression probability of being late", ylabel="density")
ax[1, 1].legend(frameon=False)
fig.suptitle("Objective 3: comparing features and target", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("compare_x_y_obj3.png", dpi=140, bbox_inches="tight"); plt.show()

tab = pd.DataFrame({"RF_importance": imp, "correlation_with_target": corr.reindex(imp.index)}).head(10).round(3)
tab["direction"] = np.where(tab["correlation_with_target"] > 0, "higher value, more likely late", "higher value, less likely late")
pd.set_option("display.width", 200)
print("TOP 10 FEATURES")
print(tab.to_string())
sep = pred3.groupby("actual_late")["logistic_regression_probability"].mean()
print(f"\nAverage Logistic Regression risk: {sep[1]:.2f} for trips that were late vs {sep[0]:.2f} for trips on time.")
print("Saved: compare_x_y_obj3.png")
```

#### Objective 4: Off-spec risk

```python
# STEP 13: COMPARE FEATURES (X) AND TARGET (y), OBJECTIVE 4 (off-spec product)
# Which batch conditions move the chance of failing the lab specification, in which direction, and do the
# predicted risks separate the batches that really failed?
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")
try:
    obj4_models
except NameError:
    obj4_models = joblib.load("obj4_models.joblib")["models"]
try:
    pred4
except NameError:
    pred4 = pd.read_pickle("obj4_test_predictions.pkl")

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
C1, C2, C3 = "#264653", "#E76F51", "#2A9D8F"
o = scaled["abt_4_quality"]
Xtr, ytr = o["X_train"], o["y_train"]
rf = obj4_models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
corr = Xtr.corrwith(ytr).dropna()
top_numeric = [c for c in imp.index if c in o["scaled_columns"]][:3]
nice = lambda s: s.replace("_", " ")

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
t = imp.head(12)[::-1]
ax[0, 0].barh([nice(c) for c in t.index], t.values, color=C1)
ax[0, 0].set(title="What the Random Forest relies on most", xlabel="importance")

t = corr.reindex(corr.abs().sort_values(ascending=False).head(12).index)[::-1]
ax[0, 1].barh([nice(c) for c in t.index], t.values, color=[C2 if v > 0 else C3 for v in t.values])
ax[0, 1].axvline(0, color="#333", lw=.8)
ax[0, 1].set(title="Direction of the link with off-spec product", xlabel="correlation with the target: right (orange) = more likely off-spec, left (green) = less likely")

for c in top_numeric:
    q = pd.qcut(Xtr[c], 5, duplicates="drop", labels=False)
    r = ytr.groupby(q).mean() * 100
    ax[1, 0].plot(range(1, len(r) + 1), r.values, marker="o", lw=2, label=nice(c))
ax[1, 0].axhline(ytr.mean() * 100, color="#888", ls="--", label="overall rate")
ax[1, 0].set(title="Off-spec rate from lowest to highest fifth of each key feature", xlabel="fifth of the feature (1 = lowest, 5 = highest)",
             ylabel="% of batches that were off-spec"); ax[1, 0].legend(frameon=False)

for lab, col, txt in [(0, C1, "passed"), (1, C2, "off-spec")]:
    ax[1, 1].hist(pred4.loc[pred4["actual_off_spec"] == lab, "logistic_regression_probability"], bins=30, density=True, alpha=.65, color=col, label=txt)
ax[1, 1].set(title="Predicted risk on the test period, by what really happened", xlabel="Logistic Regression probability of being off-spec", ylabel="density")
ax[1, 1].legend(frameon=False)
fig.suptitle("Objective 4: comparing features and target", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("compare_x_y_obj4.png", dpi=140, bbox_inches="tight"); plt.show()

tab = pd.DataFrame({"RF_importance": imp, "correlation_with_target": corr.reindex(imp.index)}).head(10).round(3)
tab["direction"] = np.where(tab["correlation_with_target"] > 0, "higher value, more likely off-spec", "higher value, less likely off-spec")
pd.set_option("display.width", 200)
print("TOP 10 FEATURES")
print(tab.to_string())
sep = pred4.groupby("actual_off_spec")["logistic_regression_probability"].mean()
print(f"\nAverage Logistic Regression risk: {sep[1]:.2f} for batches that were off-spec vs {sep[0]:.2f} for batches that passed.")
print("Saved: compare_x_y_obj4.png")
```

#### Objective 5: Inventory check

```python
# STEP 13: COMPARE FEATURES (X) AND THE MODEL OUTPUT, OBJECTIVE 5 (inventory anomalies)
# There is no target (y) to compare with, so we compare tank-days the detectors call unusual against normal ones.
# "Unusual" here = flagged by at least 2 of the 3 models. The answer key is still not used.
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
try:
    scaled
except NameError:
    scaled = pd.read_pickle("meridian_scaled.pkl")
try:
    pred5
except NameError:
    pred5 = pd.read_pickle("obj5_scores.pkl")

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
C1, C2, C3 = "#264653", "#E76F51", "#2A9D8F"
o = scaled["abt_5_inventory"]
cols = [c for c in o["scaled_columns"] if c not in ("record_month", "record_day")]
X = o["X_train"][cols].reset_index(drop=True)
p = pred5.reset_index(drop=True)
unusual = (p["models_agreeing"] >= 2).values
nice = lambda s: s.replace("_", " ")

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
# A. how far unusual features sit from normal (in standard deviations)
diff = (X[unusual].mean() - X[~unusual].mean()).sort_values()
ax[0, 0].barh([nice(c) for c in diff.index], diff.values, color=[C2 if v > 0 else C3 for v in diff.values])
ax[0, 0].axvline(0, color="#333", lw=.8)
ax[0, 0].set(title="How unusual tank-days differ from normal ones", xlabel="difference in average (standard deviations): right = higher than normal, left = lower")

# B. gap distribution
gap = p["gap_pct"].clip(-12, 12)
ax[0, 1].hist(gap[~unusual].dropna(), bins=60, color=C1, alpha=.75, density=True, label="normal")
ax[0, 1].hist(gap[unusual].dropna(), bins=60, color=C2, alpha=.7, density=True, label="unusual (2+ models agree)")
ax[0, 1].set(title="Measured vs book gap", xlabel="gap as % of book volume (clipped at +/-12)", ylabel="density"); ax[0, 1].legend(frameon=False)

# C. score against the gap
sc = p["isolation_forest_score"]
colr = np.where(p["models_agreeing"] >= 2, C2, np.where(p["models_agreeing"] == 1, "#E9C46A", "#B0B0B0"))
ax[1, 0].scatter(gap, sc, c=colr, s=9, alpha=.6)
ax[1, 0].set(title="Isolation Forest score vs gap", xlabel="gap as % of book volume (clipped at +/-12)", ylabel="anomaly score (higher = stranger)")
ax[1, 0].text(0.02, 0.95, "orange: 2-3 models flag   yellow: 1 model   grey: none", transform=ax[1, 0].transAxes, fontsize=8.5, va="top")

# D. which tanks are flagged most
rate = pd.Series(unusual, index=p["tank_id"]).groupby(level=0).mean().sort_values(ascending=False).head(10)[::-1] * 100
ax[1, 1].barh(rate.index, rate.values, color=C1)
ax[1, 1].axvline(unusual.mean() * 100, color=C2, ls="--", label=f"all tanks ({unusual.mean():.1%})")
ax[1, 1].set(title="Tanks with the most unusual days", xlabel="% of the tank's days flagged by 2+ models"); ax[1, 1].legend(frameon=False)
fig.suptitle("Objective 5: unusual vs normal tank-days", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("compare_x_y_obj5.png", dpi=140, bbox_inches="tight"); plt.show()

rawX = pd.read_pickle("meridian_encoded.pkl")["enc"]["abt_5_inventory"].reset_index(drop=True)
big = rawX["abs_gap_pct"] > 1
print(f"Unusual tank-days (2+ models): {unusual.sum()} of {len(p):,} ({unusual.mean():.1%})")
print("\nHOW MANY HAVE A BIG GAP (over 1% of book volume)?")
print(f"  among unusual tank-days: {big[unusual].mean():.0%}     among normal tank-days: {big[~unusual].mean():.1%}")
print(f"  share of all big-gap tank-days that the models call unusual: {unusual[big.values].mean():.0%}")
print("\nTOP 3 TANKS BY SHARE OF UNUSUAL DAYS")
print(rate[::-1].head(3).round(1).to_string())
print("\nSaved: compare_x_y_obj5.png")
```

---

## 17. Evaluate Model

For rare events, accuracy is misleading, so I use **PR AUC** (compared with the no-skill line), ROC AUC with 95% bootstrap ranges, and a **top-k action plan**: if the team can only act on the riskiest 10% or 20%, how many real problems are inside that list?

### Objective 1: Maintenance (73 breakdowns in 1,412 asset-months)

| Model | ROC AUC | PR AUC (no-skill 0.052) |
|---|---|---|
| **Logistic Regression** | 0.753 (0.69 to 0.81) | **0.155** (0.11 to 0.23) |
| Random Forest | 0.759 | 0.132 |
| Extra Trees | 0.740 | 0.122 |
| Gradient Boosting | 0.697 | 0.104 |

| Service the riskiest | Assets | Breakdowns caught | Share caught | Lift vs random |
|---|---|---|---|---|
| 5% | 71 | 17 | 23% | 4.6x |
| **10%** | 141 | 28 | **38%** | 3.8x |
| 20% | 282 | 39 | 53% | 2.7x |
| 30% | 424 | 46 | 63% | 2.1x |

### Objective 2: Margin (864 refinery-days, actual average 9.70 USD/bbl)

| Model | MAE (USD/bbl) | RMSE | R² |
|---|---|---|---|
| Baseline: training average | 2.948 | 3.641 | -0.624 |
| Baseline: last week's margin | 1.773 | 2.225 | 0.393 |
| **Ridge Regression** | **0.856** | 1.096 | **0.853** |
| Random Forest | 0.900 | 1.151 | 0.838 |
| Extra Trees | 0.902 | 1.158 | 0.836 |
| Gradient Boosting | 0.891 | 1.128 | 0.844 |

The best model cuts the typical error by 52% against last week's margin. Predictions land within 1 USD/bbl 65% of the time and within 2 USD/bbl 92% of the time. The simple linear model won, which tells me margin is driven by straightforward, additive factors.

### Objective 3: Delivery (124 late trips in 1,143)

| Model | ROC AUC | PR AUC (no-skill 0.108) |
|---|---|---|
| Logistic Regression | 0.782 | 0.465 |
| **Random Forest** | **0.789** (0.74 to 0.84) | **0.480** (0.39 to 0.57) |
| Extra Trees | 0.774 | 0.435 |
| Gradient Boosting | 0.787 | 0.467 |

| Flag the riskiest | Trips | Late trips caught | Share caught | Lift |
|---|---|---|---|---|
| 5% | 57 | 40 | 32% | 6.5x |
| 10% | 114 | 55 | 44% | 4.4x |
| **20%** | 229 | 77 | **62%** | 3.1x |
| 30% | 343 | 87 | 70% | 2.3x |

### Objective 4: Off-spec (54 off-spec batches in 827)

| Model | ROC AUC | PR AUC (no-skill 0.065) |
|---|---|---|
| **Logistic Regression** | **0.733** (0.67 to 0.80) | **0.162** (0.11 to 0.25) |
| Random Forest | 0.691 | 0.140 |
| Extra Trees | 0.676 | 0.137 |
| Gradient Boosting | 0.674 | 0.125 |

| Check the riskiest | Batches | Off-spec caught | Share caught | Lift |
|---|---|---|---|---|
| 5% | 41 | 11 | 20% | 4.1x |
| 10% | 83 | 18 | 33% | 3.3x |
| **20%** | 165 | 26 | **48%** | 2.4x |
| 30% | 248 | 33 | 61% | 2.0x |

### Objective 5: Inventory (367 real anomalies in 10,182 tank-days, 3.6%)

| Detector | ROC AUC | PR AUC (no-skill 0.036) | Precision | Recall |
|---|---|---|---|---|
| **Isolation Forest** | **0.927** | **0.401** | 0.435 | 0.362 |
| Local Outlier Factor | 0.578 | 0.138 | 0.144 | 0.112 |
| One-Class SVM | 0.534 | 0.119 | 0.135 | 0.114 |
| **Rule: gap over 1% of book** | 0.986 | **0.987** | 1.000 | 0.907 |

Share of each anomaly type caught:

| Anomaly type | Isolation Forest | LOF | One-Class SVM | Gap rule |
|---|---|---|---|---|
| Data entry error (34) | 97% | 94% | 94% | 97% |
| Theft (62) | 60% | 3% | 3% | 98% |
| Unrecorded receipt (40) | 50% | 5% | 2% | 92% |
| Leak (49) | 12% | 4% | 6% | 96% |
| Meter drift (182) | 20% | 2% | 2% | 85% |

**The honest finding:** the best machine learning detector is Isolation Forest, but a one-line rule beats it. So the rule becomes the daily control and the model stays as a second opinion.

### Did each model meet its objective?

I set success bars before looking at results (`ML_Objectives` sheet).

| # | Objective | Success bar | Achieved | Verdict |
|---|---|---|---|---|
| 1 | Maintenance | Lift in top 10% of at least 3, share caught of at least 0.3 | 3.8 and 0.38 | Met |
| 2 | Margin | R² of at least 0.8, MAE cut vs last week of at least 0.3 | 0.85 and 0.52 | Met |
| 3 | Delivery | ROC AUC of at least 0.75, share caught in top 20% of at least 0.5 | 0.79 and 0.62 | Met |
| 4 | Off-spec | ROC AUC of at least 0.70, share caught in top 20% of at least 0.4 | 0.73 and 0.48 | Met |
| 5 | Inventory | PR AUC at least 5x chance, and at least match the simple rule | 11.1x chance, but only 0.41 of the rule | Partly met |

**4 of 5 met, 1 partly met.**

### Charts

<p align="center">
  <img src="images/obj1_evaluation.png" alt="Objective 1: evaluation" width="100%">
</p>
<p align="center"><em>Objective 1: evaluation</em></p>

<p align="center">
  <img src="images/obj2_evaluation.png" alt="Objective 2: evaluation" width="100%">
</p>
<p align="center"><em>Objective 2: evaluation</em></p>

<p align="center">
  <img src="images/obj3_evaluation.png" alt="Objective 3: evaluation" width="100%">
</p>
<p align="center"><em>Objective 3: evaluation</em></p>

<p align="center">
  <img src="images/obj4_evaluation.png" alt="Objective 4: evaluation" width="100%">
</p>
<p align="center"><em>Objective 4: evaluation</em></p>

<p align="center">
  <img src="images/obj5_evaluation.png" alt="Objective 5: evaluation" width="100%">
</p>
<p align="center"><em>Objective 5: evaluation</em></p>

<p align="center">
  <img src="images/ml_objectives.png" alt="Success bars vs achieved results" width="100%">
</p>
<p align="center"><em>Success bars vs achieved results</em></p>

### Code: evaluate model

#### Objective 1: Maintenance risk

```python
# STEP 14: EVALUATE MODELS, OBJECTIVE 1 (predictive maintenance) on the unseen test months
# Breakdowns are rare, so accuracy is useless (predicting "no breakdown" always is 95% accurate).
# We judge on ROC AUC, PR AUC (compared with the 5% chance baseline) and on how many real breakdowns
# are caught when the maintenance team can only inspect a fixed share of assets.
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
try:
    pred1
except NameError:
    pred1 = pd.read_pickle("obj1_test_predictions.pkl")

y = pred1["actual_breakdown_next_30d"].values
MODELS = ["Logistic Regression", "Random Forest", "Extra Trees", "Gradient Boosting"]
tag = lambda m: m.lower().replace(" ", "_")
prob = {m: pred1[f"{tag(m)}_probability"].values for m in MODELS}
base = y.mean()
rng = np.random.default_rng(42)
boot = [rng.integers(0, len(y), len(y)) for _ in range(500)]
boot = [b for b in boot if y[b].sum() > 0]

rows = []
for m in MODELS:
    p, hard = prob[m], pred1[f"{tag(m)}_prediction"].values
    ci_roc = np.percentile([roc_auc_score(y[b], p[b]) for b in boot], [2.5, 97.5])
    ci_pr = np.percentile([average_precision_score(y[b], p[b]) for b in boot], [2.5, 97.5])
    rows.append({"model": m, "ROC_AUC": round(roc_auc_score(y, p), 3), "ROC_AUC_95%_range": f"{ci_roc[0]:.2f}-{ci_roc[1]:.2f}",
                 "PR_AUC": round(average_precision_score(y, p), 3), "PR_AUC_95%_range": f"{ci_pr[0]:.2f}-{ci_pr[1]:.2f}",
                 "precision_at_0.5": round(precision_score(y, hard), 3), "recall_at_0.5": round(recall_score(y, hard), 3),
                 "F1_at_0.5": round(f1_score(y, hard), 3)})
res = pd.DataFrame(rows)
best = res.sort_values("PR_AUC", ascending=False).iloc[0]["model"]
pd.set_option("display.width", 230, "display.max_columns", 20)
print(f"TEST RESULTS ({len(y):,} asset-months, {int(y.sum())} real breakdowns; a model with no skill would score PR AUC {base:.3f} and ROC AUC 0.500)")
print(res.to_string(index=False))
print(f"\nBest model by PR AUC (the right yardstick for a rare event): {best}")

# What the maintenance team gets if it inspects only the riskiest share of assets
def top_share(p, share):
    n = int(round(len(p) * share)); idx = np.argsort(-p)[:n]
    caught = y[idx].sum()
    return {"inspect_top_%": f"{share:.0%}", "assets_inspected": n, "breakdowns_caught": int(caught),
            "recall_%": round(caught / y.sum() * 100, 1), "precision_%": round(caught / n * 100, 1),
            "lift_vs_random": round((caught / n) / base, 1)}
tab = pd.DataFrame([top_share(prob[best], s) for s in [.05, .10, .15, .20, .30]])
print(f"\nINSPECTION PLAN WITH {best.upper()}")
print(tab.to_string(index=False))

# Figure
COL = {"Logistic Regression": "#264653", "Random Forest": "#E76F51", "Extra Trees": "#2A9D8F", "Gradient Boosting": "#E9C46A"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))
for m in MODELS:
    f, t, _ = roc_curve(y, prob[m]); ax[0, 0].plot(f, t, lw=2, color=COL[m], label=f"{m} ({roc_auc_score(y, prob[m]):.2f})")
ax[0, 0].plot([0, 1], [0, 1], "--", color="#888", label="no skill")
ax[0, 0].set(title="ROC curve", xlabel="false alarm rate", ylabel="breakdowns caught (recall)"); ax[0, 0].legend(frameon=False)
for m in MODELS:
    pr, rc, _ = precision_recall_curve(y, prob[m]); ax[0, 1].plot(rc, pr, lw=2, color=COL[m], label=f"{m} ({average_precision_score(y, prob[m]):.2f})")
ax[0, 1].axhline(base, ls="--", color="#888", label=f"no skill ({base:.2f})")
ax[0, 1].set(title="Precision-recall curve", xlabel="breakdowns caught (recall)", ylabel="precision"); ax[0, 1].legend(frameon=False)
for m in MODELS:
    order = np.argsort(-prob[m]); gain = np.cumsum(y[order]) / y.sum()
    ax[1, 0].plot(np.arange(1, len(y) + 1) / len(y) * 100, gain * 100, lw=2, color=COL[m], label=m)
ax[1, 0].plot([0, 100], [0, 100], "--", color="#888", label="random inspection")
ax[1, 0].set(title="Breakdowns caught vs share of assets inspected", xlabel="% of asset-months inspected (riskiest first)",
             ylabel="% of breakdowns caught"); ax[1, 0].legend(frameon=False)
n10 = int(len(y) * 0.10); cut = np.sort(prob[best])[::-1][n10 - 1]
ConfusionMatrixDisplay(confusion_matrix(y, prob[best] >= cut), display_labels=["no breakdown", "breakdown"]).plot(ax=ax[1, 1], cmap="Blues", colorbar=False)
ax[1, 1].grid(False); ax[1, 1].set(title=f"{best}: inspecting the riskiest 10%", xlabel="predicted", ylabel="actual")
fig.suptitle("Objective 1: model evaluation on the test months", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("evaluate_obj1.png", dpi=140, bbox_inches="tight"); plt.show()

res.to_csv("obj1_evaluation.csv", index=False)
pd.to_pickle({"results": res, "best_model": best, "inspection_plan": tab, "cutoff_top10": float(cut)}, "obj1_evaluation.pkl")
print(f"\nSaved: evaluate_obj1.png, obj1_evaluation.csv   |   chosen model: {best}, risk cut-off for the top 10%: {cut:.2f}")
```

#### Objective 2: Margin forecast

```python
# STEP 14: EVALUATE MODELS, OBJECTIVE 2 (refinery margin) on the unseen test period
# MAE = typical error in USD per barrel. RMSE punishes big misses. R2 = share of the ups and downs explained.
# Two simple "no model" baselines are included, so we can see whether the models add real value.
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
try:
    pred2
except NameError:
    pred2 = pd.read_pickle("obj2_test_predictions.pkl")
try:
    split
except NameError:
    split = pd.read_pickle("meridian_split.pkl")

y = pred2["actual_margin"].values
MODELS = ["Ridge Regression", "Random Forest", "Extra Trees", "Gradient Boosting"]
tag = lambda m: m.lower().replace(" ", "_")
pred = {m: pred2[f"{tag(m)}_prediction"].values for m in MODELS}

s = split["abt_2_margin"]
train_mean = s["y_train"].mean()
last_week = s["X_test"]["margin_prev_7d_avg"].fillna(train_mean).values       # simple baseline: "margin will equal last week's average"
BASE = {"Baseline: training average": np.full(len(y), train_mean), "Baseline: last week's margin": last_week}


def scores(p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "R2": r2_score(y, p), "average_error": (p - y).mean()}


rows = [{"model": m, **{k: round(v, 3) for k, v in scores(p).items()}} for m, p in {**BASE, **pred}.items()]
res = pd.DataFrame(rows)
best = res[res["model"].isin(MODELS)].sort_values("MAE").iloc[0]["model"]
pd.set_option("display.width", 200, "display.max_columns", 20)
print(f"TEST RESULTS ({len(y):,} refinery-days; actual margin averages {y.mean():.2f} USD/bbl, std {y.std():.2f})")
print(res.to_string(index=False))
print(f"\nBest model by MAE: {best}")
b = pred[best]
gain = 1 - mean_absolute_error(y, b) / mean_absolute_error(y, last_week)
print(f"It cuts the typical error by {gain:.0%} compared with simply using last week's margin.")
print(f"Typical error {mean_absolute_error(y, b):.2f} USD/bbl = {mean_absolute_error(y, b) / y.mean():.0%} of the average margin.")
within = lambda tol: (np.abs(b - y) <= tol).mean()
print(f"Predictions within 1 USD/bbl: {within(1):.0%}   within 2 USD/bbl: {within(2):.0%}")

# Figure
COL = {"Ridge Regression": "#264653", "Random Forest": "#E76F51", "Extra Trees": "#2A9D8F", "Gradient Boosting": "#E9C46A"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
order = res.sort_values("MAE", ascending=False)
ax[0, 0].barh(order["model"], order["MAE"], color=[COL.get(m, "#B0B0B0") for m in order["model"]])
for i, v in enumerate(order["MAE"]):
    ax[0, 0].text(v + 0.02, i, f"{v:.2f}", va="center")
ax[0, 0].set(title="Typical error (MAE): lower is better", xlabel="USD per barrel")

ax[0, 1].scatter(y, b, s=10, alpha=.4, color=COL[best])
lo, hi = y.min(), y.max(); ax[0, 1].plot([lo, hi], [lo, hi], "--", color="#333", label="perfect prediction")
ax[0, 1].set(title=f"{best}: predicted vs actual (R2 = {r2_score(y, b):.2f})", xlabel="actual margin (USD per barrel)",
             ylabel="predicted margin (USD per barrel)"); ax[0, 1].legend(frameon=False)

err = b - y
ax[1, 0].hist(err, bins=40, color=COL[best]); ax[1, 0].axvline(0, color="#333", lw=1)
ax[1, 0].set(title="Prediction errors", xlabel="predicted minus actual (USD per barrel): left = too low, right = too high", ylabel="refinery-days")

month = pred2["run_date"].dt.to_period("M").astype(str)
mm = pd.DataFrame({"month": month, "abs": np.abs(err), "act": y, "pred": b}).groupby("month").agg(MAE=("abs", "mean"), actual=("act", "mean"), predicted=("pred", "mean"))
ax[1, 1].bar(mm.index, mm["MAE"], color="#B0B0B0", label="typical error (MAE)")
ax[1, 1].set(title="Accuracy by month", ylabel="MAE (USD per barrel)")
ax2 = ax[1, 1].twinx(); ax2.plot(mm.index, mm["actual"], marker="o", color="#264653", label="actual average margin")
ax2.plot(mm.index, mm["predicted"], marker="s", ls="--", color="#E76F51", label="predicted average margin"); ax2.set_ylabel("average margin (USD per barrel)")
ax2.grid(False); ax2.spines["right"].set_visible(True)
h1, l1 = ax[1, 1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[1, 1].legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", fontsize=8)
fig.suptitle("Objective 2: model evaluation on the test period", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("evaluate_obj2.png", dpi=140, bbox_inches="tight"); plt.show()

res.to_csv("obj2_evaluation.csv", index=False)
pd.to_pickle({"results": res, "best_model": best}, "obj2_evaluation.pkl")
print(f"\nSaved: evaluate_obj2.png, obj2_evaluation.csv   |   chosen model: {best}")
```

#### Objective 3: Delivery risk

```python
# STEP 14: EVALUATE MODELS, OBJECTIVE 3 (late delivery) on the unseen test period
# Late trips are rare (about 11%), so accuracy is useless (predicting "on time" always is 89% accurate).
# We judge on ROC AUC, PR AUC (compared with the chance baseline) and on how many late trips are caught
# when dispatchers can only add a safety buffer to, or re-plan, a fixed share of trips.
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
try:
    pred3
except NameError:
    pred3 = pd.read_pickle("obj3_test_predictions.pkl")

y = pred3["actual_late"].values
MODELS = ["Logistic Regression", "Random Forest", "Extra Trees", "Gradient Boosting"]
tag = lambda m: m.lower().replace(" ", "_")
prob = {m: pred3[f"{tag(m)}_probability"].values for m in MODELS}
base = y.mean()
rng = np.random.default_rng(42)
boot = [rng.integers(0, len(y), len(y)) for _ in range(500)]
boot = [b for b in boot if y[b].sum() > 0]

rows = []
for m in MODELS:
    p, hard = prob[m], pred3[f"{tag(m)}_prediction"].values
    ci_roc = np.percentile([roc_auc_score(y[b], p[b]) for b in boot], [2.5, 97.5])
    ci_pr = np.percentile([average_precision_score(y[b], p[b]) for b in boot], [2.5, 97.5])
    rows.append({"model": m, "ROC_AUC": round(roc_auc_score(y, p), 3), "ROC_AUC_95%_range": f"{ci_roc[0]:.2f}-{ci_roc[1]:.2f}",
                 "PR_AUC": round(average_precision_score(y, p), 3), "PR_AUC_95%_range": f"{ci_pr[0]:.2f}-{ci_pr[1]:.2f}",
                 "precision_at_0.5": round(precision_score(y, hard), 3), "recall_at_0.5": round(recall_score(y, hard), 3),
                 "F1_at_0.5": round(f1_score(y, hard), 3)})
res = pd.DataFrame(rows)
best = res.sort_values("PR_AUC", ascending=False).iloc[0]["model"]
pd.set_option("display.width", 230, "display.max_columns", 20)
print(f"TEST RESULTS ({len(y):,} trips, {int(y.sum())} real late trips; a model with no skill would score PR AUC {base:.3f} and ROC AUC 0.500)")
print(res.to_string(index=False))
print(f"\nBest model by PR AUC (the right yardstick for a rare event): {best}")

# What dispatch gets if it only acts on the riskiest share of trips
def top_share(p, share):
    n = int(round(len(p) * share)); idx = np.argsort(-p)[:n]
    caught = y[idx].sum()
    return {"flag_riskiest_%": f"{share:.0%}", "trips_flagged": n, "late_trips_caught": int(caught),
            "recall_%": round(caught / y.sum() * 100, 1), "precision_%": round(caught / n * 100, 1),
            "lift_vs_random": round((caught / n) / base, 1)}
tab = pd.DataFrame([top_share(prob[best], s) for s in [.05, .10, .20, .30, .40]])
print(f"\nDISPATCH PLAN WITH {best.upper()}")
print(tab.to_string(index=False))

# Figure
COL = {"Logistic Regression": "#264653", "Random Forest": "#E76F51", "Extra Trees": "#2A9D8F", "Gradient Boosting": "#E9C46A"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))
for m in MODELS:
    f, t, _ = roc_curve(y, prob[m]); ax[0, 0].plot(f, t, lw=2, color=COL[m], label=f"{m} ({roc_auc_score(y, prob[m]):.2f})")
ax[0, 0].plot([0, 1], [0, 1], "--", color="#888", label="no skill")
ax[0, 0].set(title="ROC curve", xlabel="false alarm rate", ylabel="late trips caught (recall)"); ax[0, 0].legend(frameon=False)
for m in MODELS:
    pr, rc, _ = precision_recall_curve(y, prob[m]); ax[0, 1].plot(rc, pr, lw=2, color=COL[m], label=f"{m} ({average_precision_score(y, prob[m]):.2f})")
ax[0, 1].axhline(base, ls="--", color="#888", label=f"no skill ({base:.2f})")
ax[0, 1].set(title="Precision-recall curve", xlabel="late trips caught (recall)", ylabel="precision"); ax[0, 1].legend(frameon=False)
for m in MODELS:
    order = np.argsort(-prob[m]); gain = np.cumsum(y[order]) / y.sum()
    ax[1, 0].plot(np.arange(1, len(y) + 1) / len(y) * 100, gain * 100, lw=2, color=COL[m], label=m)
ax[1, 0].plot([0, 100], [0, 100], "--", color="#888", label="random flagging")
ax[1, 0].set(title="Late trips caught vs share of trips flagged", xlabel="% of trips flagged (riskiest first)",
             ylabel="% of late trips caught"); ax[1, 0].legend(frameon=False)
n20 = int(len(y) * 0.20); cut = np.sort(prob[best])[::-1][n20 - 1]
ConfusionMatrixDisplay(confusion_matrix(y, prob[best] >= cut), display_labels=["on time", "late"]).plot(ax=ax[1, 1], cmap="Blues", colorbar=False)
ax[1, 1].grid(False); ax[1, 1].set(title=f"{best}: flagging the riskiest 20%", xlabel="predicted", ylabel="actual")
fig.suptitle("Objective 3: model evaluation on the test period", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("evaluate_obj3.png", dpi=140, bbox_inches="tight"); plt.show()

res.to_csv("obj3_evaluation.csv", index=False)
pd.to_pickle({"results": res, "best_model": best, "inspection_plan": tab, "cutoff_top20": float(cut)}, "obj3_evaluation.pkl")
print(f"\nSaved: evaluate_obj3.png, obj3_evaluation.csv   |   chosen model: {best}, risk cut-off for the riskiest 20%: {cut:.2f}")
```

#### Objective 4: Off-spec risk

```python
# STEP 14: EVALUATE MODELS, OBJECTIVE 4 (off-spec product) on the unseen test period
# Off-spec batches are rare (about 7%), so accuracy is useless (predicting "pass" always is 93% accurate).
# We judge on ROC AUC, PR AUC (compared with the chance baseline) and on how many failing batches are caught
# when the quality team can only run extra checks on a fixed share of batches.
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
try:
    pred4
except NameError:
    pred4 = pd.read_pickle("obj4_test_predictions.pkl")

y = pred4["actual_off_spec"].values
MODELS = ["Logistic Regression", "Random Forest", "Extra Trees", "Gradient Boosting"]
tag = lambda m: m.lower().replace(" ", "_")
prob = {m: pred4[f"{tag(m)}_probability"].values for m in MODELS}
base = y.mean()
rng = np.random.default_rng(42)
boot = [rng.integers(0, len(y), len(y)) for _ in range(500)]
boot = [b for b in boot if y[b].sum() > 0]

rows = []
for m in MODELS:
    p, hard = prob[m], pred4[f"{tag(m)}_prediction"].values
    ci_roc = np.percentile([roc_auc_score(y[b], p[b]) for b in boot], [2.5, 97.5])
    ci_pr = np.percentile([average_precision_score(y[b], p[b]) for b in boot], [2.5, 97.5])
    rows.append({"model": m, "ROC_AUC": round(roc_auc_score(y, p), 3), "ROC_AUC_95%_range": f"{ci_roc[0]:.2f}-{ci_roc[1]:.2f}",
                 "PR_AUC": round(average_precision_score(y, p), 3), "PR_AUC_95%_range": f"{ci_pr[0]:.2f}-{ci_pr[1]:.2f}",
                 "precision_at_0.5": round(precision_score(y, hard), 3), "recall_at_0.5": round(recall_score(y, hard), 3),
                 "F1_at_0.5": round(f1_score(y, hard), 3)})
res = pd.DataFrame(rows)
best = res.sort_values("PR_AUC", ascending=False).iloc[0]["model"]
pd.set_option("display.width", 230, "display.max_columns", 20)
print(f"TEST RESULTS ({len(y):,} batches, {int(y.sum())} real off-spec batches; a model with no skill would score PR AUC {base:.3f} and ROC AUC 0.500)")
print(res.to_string(index=False))
print(f"\nBest model by PR AUC (the right yardstick for a rare event): {best}")

# What the quality team gets if it only checks the riskiest share of batches
def top_share(p, share):
    n = int(round(len(p) * share)); idx = np.argsort(-p)[:n]
    caught = y[idx].sum()
    return {"check_riskiest_%": f"{share:.0%}", "batches_checked": n, "off_spec_caught": int(caught),
            "recall_%": round(caught / y.sum() * 100, 1), "precision_%": round(caught / n * 100, 1),
            "lift_vs_random": round((caught / n) / base, 1)}
tab = pd.DataFrame([top_share(prob[best], s) for s in [.05, .10, .20, .30, .40]])
print(f"\nQUALITY-CHECK PLAN WITH {best.upper()}")
print(tab.to_string(index=False))

# Figure
COL = {"Logistic Regression": "#264653", "Random Forest": "#E76F51", "Extra Trees": "#2A9D8F", "Gradient Boosting": "#E9C46A"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))
for m in MODELS:
    f, t, _ = roc_curve(y, prob[m]); ax[0, 0].plot(f, t, lw=2, color=COL[m], label=f"{m} ({roc_auc_score(y, prob[m]):.2f})")
ax[0, 0].plot([0, 1], [0, 1], "--", color="#888", label="no skill")
ax[0, 0].set(title="ROC curve", xlabel="false alarm rate", ylabel="off-spec batches caught (recall)"); ax[0, 0].legend(frameon=False)
for m in MODELS:
    pr, rc, _ = precision_recall_curve(y, prob[m]); ax[0, 1].plot(rc, pr, lw=2, color=COL[m], label=f"{m} ({average_precision_score(y, prob[m]):.2f})")
ax[0, 1].axhline(base, ls="--", color="#888", label=f"no skill ({base:.2f})")
ax[0, 1].set(title="Precision-recall curve", xlabel="off-spec batches caught (recall)", ylabel="precision"); ax[0, 1].legend(frameon=False)
for m in MODELS:
    order = np.argsort(-prob[m]); gain = np.cumsum(y[order]) / y.sum()
    ax[1, 0].plot(np.arange(1, len(y) + 1) / len(y) * 100, gain * 100, lw=2, color=COL[m], label=m)
ax[1, 0].plot([0, 100], [0, 100], "--", color="#888", label="random checking")
ax[1, 0].set(title="Off-spec batches caught vs share of batches checked", xlabel="% of batches checked (riskiest first)",
             ylabel="% of off-spec batches caught"); ax[1, 0].legend(frameon=False)
n20 = int(len(y) * 0.20); cut = np.sort(prob[best])[::-1][n20 - 1]
ConfusionMatrixDisplay(confusion_matrix(y, prob[best] >= cut), display_labels=["pass", "off-spec"]).plot(ax=ax[1, 1], cmap="Blues", colorbar=False)
ax[1, 1].grid(False); ax[1, 1].set(title=f"{best}: checking the riskiest 20%", xlabel="predicted", ylabel="actual")
fig.suptitle("Objective 4: model evaluation on the test period", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("evaluate_obj4.png", dpi=140, bbox_inches="tight"); plt.show()

res.to_csv("obj4_evaluation.csv", index=False)
pd.to_pickle({"results": res, "best_model": best, "inspection_plan": tab, "cutoff_top20": float(cut)}, "obj4_evaluation.pkl")
print(f"\nSaved: evaluate_obj4.png, obj4_evaluation.csv   |   chosen model: {best}, risk cut-off for the riskiest 20%: {cut:.2f}")
```

#### Objective 5: Inventory check

```python
# STEP 14: EVALUATE MODELS, OBJECTIVE 5 (inventory anomalies)
# The detectors never saw labels. Only NOW do we open the answer key (anomaly_answer_key.csv) to see whether
# the tank-days they flagged are the real problems: theft, leaks, entry errors, unrecorded receipts, meter drift.
# A simple rule ("gap above 1% of book volume") is included as the baseline to beat.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")
try:
    pred5
except NameError:
    if not os.path.exists("obj5_scores.pkl"):
        raise FileNotFoundError("obj5_scores.pkl is missing. Run Step 12 (predict) for Objective 5 first, then re-run this cell.")
    pred5 = pd.read_pickle("obj5_scores.pkl")

import os, glob
KEY = None
hits = glob.glob("meridian_inventory_truth_labels*.csv")
if hits:
    KEY = max(hits, key=os.path.getmtime)
else:                                                     # upload the small CSV (NOT the xlsx workbook)
    print("Upload meridian_inventory_truth_labels.csv now (the small CSV file, not the Excel workbook).")
    from google.colab import files
    up = files.upload()                                   # whatever name Colab gives it is used as is
    csvs = [k for k in up if k.lower().endswith(".csv")]
    if not csvs:
        raise FileNotFoundError("No CSV was uploaded. Re-run this cell and choose meridian_inventory_truth_labels.csv")
    KEY = csvs[0]
print("Using answer key:", KEY)
key = pd.read_csv(KEY)

m = pred5.merge(key, on="record_id", how="left")
assert m["is_anomaly"].notna().all(), "some tank-days are missing from the answer key"
y = m["is_anomaly"].astype(int).values
N = int(y.sum())
base = y.mean()
MODELS = {"Isolation Forest": "isolation_forest", "Local Outlier Factor": "local_outlier_factor", "One-Class SVM": "one_class_svm"}
score = {n: m[f"{t}_score"].values for n, t in MODELS.items()}
flag = {n: m[f"{t}_flag"].values for n, t in MODELS.items()}
score["Rule: gap over 1%"] = m["gap_pct"].abs().fillna(0).values                  # baseline rule
flag["Rule: gap over 1%"] = (m["gap_pct"].abs() > 1).astype(int).values
ALL = list(score)

rows = []
for n in ALL:
    top = np.argsort(-score[n])[:N]
    f = flag[n] == 1
    rows.append({"model": n, "ROC_AUC": round(roc_auc_score(y, score[n]), 3), "PR_AUC": round(average_precision_score(y, score[n]), 3),
                 "flagged": int(f.sum()), "precision": round(y[f].mean(), 3), "recall": round(y[f].sum() / N, 3),
                 f"precision_in_top_{N}": round(y[top].mean(), 3)})
res = pd.DataFrame(rows)
pd.set_option("display.width", 220, "display.max_columns", 20)
print(f"TEST RESULTS ({len(y):,} tank-days, {N} real anomalies = {base:.1%}; a model with no skill would score PR AUC {base:.3f})")
print(res.to_string(index=False))
best = res[res["model"].isin(MODELS)].sort_values("PR_AUC", ascending=False).iloc[0]["model"]
print(f"\nBest detector by PR AUC: {best}")

print("\nSHARE OF EACH ANOMALY TYPE CAUGHT (recall of the flags)")
types = ["data_entry_error", "theft", "unrecorded_receipt", "leak", "meter_drift"]
by_type = pd.DataFrame({n: [(m.loc[m["anomaly_type"] == t, "gap_pct"].notna().sum() and flag[n][m["anomaly_type"].values == t].mean()) for t in types] for n in ALL},
                       index=[f"{t} (n={int((m['anomaly_type'] == t).sum())})" for t in types])
print((by_type * 100).round(0).astype(int).astype(str).add("%").to_string())

print(f"\nREVIEW PLAN WITH {best.upper()}: investigators check the highest-scoring tank-days first")
order = np.argsort(-score[best]); plan = []
for k in [100, 200, N, 500, 800]:
    c = y[order[:k]].sum()
    plan.append({"tank_days_reviewed": k, "real_anomalies_found": int(c), "recall_%": round(c / N * 100, 1), "precision_%": round(c / k * 100, 1),
                 "lift_vs_random": round(c / k / base, 1)})
print(pd.DataFrame(plan).to_string(index=False))

# Figure
COL = {"Isolation Forest": "#264653", "Local Outlier Factor": "#E76F51", "One-Class SVM": "#2A9D8F", "Rule: gap over 1%": "#999999"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))
for n in ALL:
    pr, rc, _ = precision_recall_curve(y, score[n]); ax[0, 0].plot(rc, pr, lw=2, color=COL[n], label=f"{n} ({average_precision_score(y, score[n]):.2f})")
ax[0, 0].axhline(base, ls="--", color="#555", label=f"no skill ({base:.2f})")
ax[0, 0].set(title="Precision-recall curve", xlabel="real anomalies found (recall)", ylabel="precision"); ax[0, 0].legend(frameon=False)

w = 0.2
for i, n in enumerate(ALL):
    ax[0, 1].bar(np.arange(len(types)) + (i - 1.5) * w, by_type[n].values * 100, w, color=COL[n], label=n)
ax[0, 1].set_xticks(np.arange(len(types))); ax[0, 1].set_xticklabels([t.replace("_", "\n") for t in types])
ax[0, 1].set(title="Which kinds of anomaly each method catches", ylabel="% caught"); ax[0, 1].legend(frameon=False, fontsize=8)

ks = np.arange(50, 1501, 25)
ax[1, 0].plot(ks, [y[order[:k]].sum() / N * 100 for k in ks], color=COL[best], lw=2, label="real anomalies found (%)")
ax[1, 0].plot(ks, [y[order[:k]].mean() * 100 for k in ks], color="#E76F51", lw=2, label="share of reviewed days that are real (%)")
ax[1, 0].axvline(N, ls=":", color="#555")
ax[1, 0].set(title=f"{best}: how far down the ranked list to review", xlabel="tank-days reviewed (highest score first)", ylabel="%"); ax[1, 0].legend(frameon=False)

ConfusionMatrixDisplay(confusion_matrix(y, flag[best]), display_labels=["normal", "anomaly"]).plot(ax=ax[1, 1], cmap="Blues", colorbar=False)
ax[1, 1].grid(False); ax[1, 1].set(title=f"{best}: flags vs the answer key", xlabel="flagged by the model", ylabel="actual")
fig.suptitle("Objective 5: anomaly detection checked against the answer key", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("evaluate_obj5.png", dpi=140, bbox_inches="tight"); plt.show()

res.to_csv("obj5_evaluation.csv", index=False)
pd.to_pickle({"results": res, "best_model": best, "recall_by_type": by_type}, "obj5_evaluation.pkl")
print(f"\nSaved: evaluate_obj5.png, obj5_evaluation.csv   |   chosen detector: {best}")
```


#### Success bars vs results (scorecard)

```python
# STEP 16: MACHINE LEARNING OBJECTIVES
# For each of the five business objectives: what ML task was set, which model was chosen, and did it reach the success bar?
# The success bars below were set for this project (editable). Every result comes from the Step 14 evaluation files,
# so nothing is re-trained here.
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ SUCCESS BARS (edit freely)
BAR = {
    1: {"Lift over random in the top 10%": 3.0, "Share of breakdowns caught in the top 10%": 0.30},
    2: {"R2": 0.80, "MAE cut vs last week's margin": 0.30},
    3: {"ROC AUC": 0.75, "Share of late trips caught in the top 20%": 0.50},
    4: {"ROC AUC": 0.70, "Share of off-spec batches caught in the top 20%": 0.40},
    5: {"PR AUC as a multiple of chance": 5.0, "Best model PR AUC / simple gap rule PR AUC": 1.0},
}

def load(n):
    p = f"obj{n}_evaluation.pkl"
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} is missing. Run Step 14 for Objective {n} first, then re-run this cell.")
    return pd.read_pickle(p)

ev = {n: load(n) for n in range(1, 6)}
best = {n: ev[n]["best_model"] for n in ev}

def row(n):
    r = ev[n]["results"]; return r[r["model"] == best[n]].iloc[0]
def plan_row(n, label):
    pl = ev[n]["inspection_plan"]; return pl[pl.iloc[:, 0].astype(str).str.strip() == label].iloc[0]

ACH = {}   # achieved values
r1, r2, r3, r4, r5 = row(1), row(2), row(3), row(4), row(5)
pl1, pl3, pl4 = plan_row(1, "10%"), plan_row(3, "20%"), plan_row(4, "20%")
ACH[1] = {"Lift over random in the top 10%": float(pl1["lift_vs_random"]), "Share of breakdowns caught in the top 10%": float(pl1["recall_%"]) / 100}
rr = ev[2]["results"]; base2 = rr[rr["model"].str.contains("last week")].iloc[0]
ACH[2] = {"R2": float(r2["R2"]), "MAE cut vs last week's margin": 1 - float(r2["MAE"]) / float(base2["MAE"])}
ACH[3] = {"ROC AUC": float(r3["ROC_AUC"]), "Share of late trips caught in the top 20%": float(pl3["recall_%"]) / 100}
ACH[4] = {"ROC AUC": float(r4["ROC_AUC"]), "Share of off-spec batches caught in the top 20%": float(pl4["recall_%"]) / 100}
res5 = ev[5]["results"]; rule5 = res5[res5["model"].str.startswith("Rule")].iloc[0]
n_anom = sum(int(re.search(r"n=(\d+)", i).group(1)) for i in ev[5]["recall_by_type"].index)
n_days = len(globals()["pred5"]) if "pred5" in globals() else (len(pd.read_pickle("obj5_scores.pkl")) if os.path.exists("obj5_scores.pkl") else 10182)
chance5 = n_anom / n_days
ACH[5] = {"PR AUC as a multiple of chance": float(r5["PR_AUC"]) / chance5, "Best model PR AUC / simple gap rule PR AUC": float(r5["PR_AUC"]) / float(rule5["PR_AUC"])}

TASKS = {
    1: ("Predictive maintenance", "Classification", "breakdown_next_30d", "asset-month", "Asset Integrity and Maintenance", "Cut unplanned downtime and maintenance cost"),
    2: ("Margin prediction", "Regression", "gross_margin_usd_per_bbl", "refinery-day", "Refining and Planning", "Maximise refinery yield and gross margin"),
    3: ("Late delivery prediction", "Classification", "late_delivery", "trip", "Supply and Distribution", "Raise on-time delivery and cut distribution cost"),
    4: ("Off-spec batch prediction", "Classification", "off_spec", "batch", "Quality and Blending", "Reduce off-spec product and rework cost"),
    5: ("Inventory anomaly detection", "Unsupervised anomaly detection", "none (checked against answer key afterwards)", "tank-day", "Terminal Operations and Security", "Detect product loss, theft and metering errors"),
}
NOTE = {
    1: f"Wide uncertainty: PR AUC 95% range {r1['PR_AUC_95%_range']}. Use as a ranking aid, not a yes/no verdict.",
    2: "Linear Ridge beat the tree models, so the margin relationship is largely linear. Forecast only; planners still decide the crude.",
    3: f"PR AUC 95% range {r3['PR_AUC_95%_range']}. Best of the four, but the other models are close.",
    4: f"Weakest classifier: PR AUC 95% range {r4['PR_AUC_95%_range']}, so the gain over random is real but modest.",
    5: "Models find recording errors and some theft, but a plain 'gap over 1%' rule beats all three. Keep the rule as the control, the model as a second opinion.",
}

rows, flat = [], []
for n in range(1, 6):
    ok = {k: ACH[n][k] >= v for k, v in BAR[n].items()}
    verdict = "Met" if all(ok.values()) else ("Partly met" if any(ok.values()) else "Not met")
    t = TASKS[n]
    headline = {1: f"PR AUC {r1['PR_AUC']}, ROC AUC {r1['ROC_AUC']}", 2: f"MAE {r2['MAE']} USD/bbl, RMSE {r2['RMSE']}, R2 {r2['R2']}",
                3: f"PR AUC {r3['PR_AUC']}, ROC AUC {r3['ROC_AUC']}", 4: f"PR AUC {r4['PR_AUC']}, ROC AUC {r4['ROC_AUC']}",
                5: f"PR AUC {r5['PR_AUC']} (chance {chance5:.3f}), ROC AUC {r5['ROC_AUC']}"}[n]
    rows.append({"objective": n, "business_objective": t[5], "owner": t[4], "ml_task": t[0], "ml_type": t[1], "target": t[2], "grain": t[3],
                 "chosen_model": best[n], "headline_result": headline,
                 "success_bars": "; ".join(f"{k} >= {v:g}" for k, v in BAR[n].items()),
                 "achieved": "; ".join(f"{k} = {ACH[n][k]:.2f}" for k in BAR[n]), "verdict": verdict, "note": NOTE[n]})
    for k, v in BAR[n].items():
        flat.append({"objective": n, "criterion": k, "bar": v, "achieved": ACH[n][k], "ratio": ACH[n][k] / v})
S = pd.DataFrame(rows); F = pd.DataFrame(flat)

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 120)
print("MACHINE LEARNING OBJECTIVES: WHAT WAS SET, WHAT WAS BUILT, WHAT WAS ACHIEVED\n")
for _, r in S.iterrows():
    print(f"OBJECTIVE {r['objective']}: {r['business_objective']}  (owner: {r['owner']})")
    print(f"  ML task      : {r['ml_task']} ({r['ml_type']}), one row per {r['grain']}")
    print(f"  Target       : {r['target']}")
    print(f"  Chosen model : {r['chosen_model']}   ->  {r['headline_result']}")
    print(f"  Success bars : {r['success_bars']}")
    print(f"  Achieved     : {r['achieved']}")
    print(f"  Verdict      : {r['verdict'].upper()}.  {r['note']}\n")
print(F.assign(ratio=F["ratio"].round(2), achieved=F["achieved"].round(3)).to_string(index=False))
print(f"\nObjectives met: {(S['verdict'] == 'Met').sum()} of 5, partly met: {(S['verdict'] == 'Partly met').sum()}, not met: {(S['verdict'] == 'Not met').sum()}")

# Figure: achieved / bar for every criterion (1.0 = exactly on the bar)
COL = {1: "#264653", 2: "#2A9D8F", 3: "#E9C46A", 4: "#F4A261", 5: "#E76F51"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "font.size": 10, "axes.titleweight": "bold"})
fig, ax = plt.subplots(figsize=(13, 6.8))
F = F.iloc[::-1].reset_index(drop=True)
ax.barh(range(len(F)), F["ratio"], color=[COL[o] for o in F["objective"]], height=.65)
ax.axvline(1, color="#222", lw=1.6, ls="--"); ax.text(1.01, len(F) - .45, "success bar", fontsize=9)
ax.set_yticks(range(len(F))); ax.set_yticklabels([f"{o}  {c}" for o, c in zip(F["objective"], F["criterion"])])
for i, r in F.iterrows():
    ax.text(r["ratio"] + .02, i, f"{r['achieved']:.2f} vs bar {r['bar']:g}", va="center", fontsize=9)
ax.set(xlabel="achieved divided by the success bar (right of the dashed line = bar reached)", xlim=(0, max(F["ratio"]) * 1.35))
ax.set_title("Machine learning objectives: result against success bar", fontsize=13)
plt.tight_layout(); plt.savefig("ml_objectives.png", dpi=140, bbox_inches="tight"); plt.show()

S.to_csv("ml_objectives_scorecard.csv", index=False)
pd.to_pickle({"scorecard": S, "criteria": F, "bars": BAR}, "ml_objectives.pkl")
print("\nSaved: ml_objectives.png, ml_objectives_scorecard.csv, ml_objectives.pkl")
```

---

## 18. Business Analysis

What are the models worth in dollars? Each plan uses facts from the data (a breakdown costs about $29,842 to repair and loses 25 hours; a preventive job costs $1,941; a failed batch costs $11,699 in rework), plus **assumptions** I state openly: downtime at $2,500 per hour, 60% of flagged breakdowns avoided, 40% of flagged late trips saved, 70% of flagged off-spec batches corrected.

| Objective | Action | Units acted on | Problems inside the list | Gross value | Cost | Net (test window) | Net per year |
|---|---|---|---|---|---|---|---|
| 1. Maintenance | Preventive job on riskiest 10% | 141 | 28 of 73 | $1,544,453 | $273,678 | $1,270,774 | **$3,801,906** |
| 3. Delivery | Buffer or re-route riskiest 20% | 229 | 77 of 124 | $18,480 | $9,160 | $9,320 | **$23,624** |
| 4. Quality | Extra check on riskiest 20% | 165 | 26 of 54 | $212,930 | $41,250 | $171,680 | **$414,989** |
| **Total** | | | | | | | **$4,240,519** |

**Random picking with the same effort** would net $128,983 for maintenance, **negative $3,208** for delivery and $47,198 for quality, so the models are what makes the plans pay.

**Objectives 2 and 5 get no dollar saving** on purpose. The margin forecast is decision support. For inventory, the 1% gap rule puts about 462,533 kl in front of investigators over the year (2.8% of dispatched product), but much of that is metering drift and timing, so I will not call it recovered money. Another 32 tank-days are recording errors (a reading 10 times too high), counted separately.

**Margin by crude grade, test period (USD/bbl)**

| Crude | Actual | Predicted | Run-days |
|---|---|---|---|
| WTI | 11.42 | 11.23 | 97 |
| Brent | 10.90 | 11.02 | 106 |
| Forcados | 10.88 | 10.80 | 101 |
| Azeri | 10.78 | 10.53 | 196 |
| Bonny Light | 9.98 | 9.94 | 103 |
| Arab Light | 8.27 | 8.30 | 121 |
| Basrah Medium | 6.26 | 6.22 | 140 |

The model's top-quarter days really earn 6.84 USD/bbl more than its bottom-quarter days, so its ranking can be trusted.

### Charts

<p align="center">
  <img src="images/business_analysis.png" alt="Business analysis: value by objective" width="100%">
</p>
<p align="center"><em>Business analysis: value by objective</em></p>

### Code: business analysis

```python
# STEP 15: BUSINESS ANALYSIS
# Turns the model results from Steps 12-14 into what the business cares about: money, volumes and workload.
# Facts such as breakdown cost, rework cost, throughput and freight cost come from the cleaned data.
# Items the data does not hold (cost of an inspection visit, penalty per late trip, price of fuel) are ASSUMPTIONS
# in the box below. They are illustrative for this fictional company: change them and re-run to see how the answer moves.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ ASSUMPTIONS (edit freely)
A = {
    "downtime_cost_per_hour_usd": 2500,      # Obj 1: lost margin for each hour a unit is down
    "breakdown_avoided_share": 0.60,         # Obj 1: share of flagged real breakdowns a timely preventive job avoids
    "late_penalty_per_trip_usd": 600,        # Obj 3: contract penalty, demurrage and customer handling per late trip
    "late_action_cost_usd": 40,               # Obj 3: cost to add a buffer / re-route one flagged trip
    "late_avoided_share": 0.40,              # Obj 3: share of flagged real late trips the action saves
    "quality_check_cost_usd": 250,           # Obj 4: extra lab test / blend adjustment per flagged batch
    "rework_avoided_share": 0.70,            # Obj 4: share of flagged real off-spec batches corrected before release
    "fuel_value_usd_per_kl": 750,            # Obj 5: value of one kilolitre of product
    "investigation_cost_usd": 300,           # Obj 5: cost of investigating one flagged tank-day
}
PLAN = {1: 0.10, 3: 0.20, 4: 0.20}           # share of assets / trips / batches acted on (same plans as Step 14)

# ------------------------------------------------------------------ load (Colab variables first, saved files second)
def get(name, path, reader=pd.read_pickle):
    g = globals()
    if name in g:
        return g[name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing. Run the earlier step that creates it, then re-run this cell.")
    return reader(path)

clean = get("clean", "meridian_clean.pkl")
pred = {1: get("pred1", "obj1_test_predictions.pkl"), 2: get("pred2", "obj2_test_predictions.pkl"),
        3: get("pred3", "obj3_test_predictions.pkl"), 4: get("pred4", "obj4_test_predictions.pkl"),
        5: get("pred5", "obj5_scores.pkl")}
DEFAULT_BEST = {1: "Logistic Regression", 2: "Ridge Regression", 3: "Random Forest", 4: "Logistic Regression", 5: "Isolation Forest"}
best = {}
for n in range(1, 6):
    p = f"obj{n}_evaluation.pkl"
    best[n] = pd.read_pickle(p)["best_model"] if os.path.exists(p) else DEFAULT_BEST[n]
slug = lambda s: s.lower().replace(" ", "_").replace("-", "_")
usd = lambda v: f"${v:,.0f}"
def days(s):
    s = pd.to_datetime(s); return max((s.max() - s.min()).days + 1, 1)

summary = []
pd.set_option("display.width", 220, "display.max_columns", 30)

# ================================================================== OBJECTIVE 1: maintenance
mt = clean["fact_maintenance"]
bd = mt[mt["work_type"] == "Breakdown"]; pv = mt[mt["work_type"] == "Preventive"]
bd_cost, bd_dt, pv_cost = bd["cost_usd"].mean(), bd["downtime_hours"].mean(), pv["cost_usd"].mean()
bd_total = bd_cost + bd_dt * A["downtime_cost_per_hour_usd"]
p1 = pred[1].dropna(subset=["actual_breakdown_next_30d"]).sort_values(f"{slug(best[1])}_probability", ascending=False)
k = int(round(len(p1) * PLAN[1])); caught = int(p1["actual_breakdown_next_30d"].head(k).sum()); total1 = int(p1["actual_breakdown_next_30d"].sum())
saving1 = caught * A["breakdown_avoided_share"] * bd_total
cost1 = k * pv_cost
net1 = saving1 - cost1
rand_caught = total1 * PLAN[1]
net1_rand = rand_caught * A["breakdown_avoided_share"] * bd_total - cost1
d1 = days(p1["reading_month"]) + 30
print("=" * 100); print("OBJECTIVE 1: CUT UNPLANNED DOWNTIME AND MAINTENANCE COST")
print(f"Facts from the data: a breakdown costs {usd(bd_cost)} to repair and loses {bd_dt:.0f} hours; a preventive job costs {usd(pv_cost)}.")
print(f"With downtime valued at {usd(A['downtime_cost_per_hour_usd'])}/h, one breakdown is worth about {usd(bd_total)}.")
print(f"Plan: send preventive jobs to the riskiest {PLAN[1]:.0%} of asset-months ({k} jobs) over the {d1}-day test window.")
print(f"  real breakdowns inside that list: {caught} of {total1} ({caught/total1:.0%}); random picking would find about {rand_caught:.0f}")
print(f"  value avoided {usd(saving1)} - cost of jobs {usd(cost1)} = net {usd(net1)}   (random picking with the same jobs: {usd(net1_rand)})")
summary.append({"objective": "1 Maintenance", "action": f"preventive job on top {PLAN[1]:.0%} risk asset-months", "units_acted_on": k,
                "problems_in_group": caught, "gross_value_usd": saving1, "action_cost_usd": cost1, "net_value_usd": net1, "test_days": d1})
ks1 = np.arange(1, 41) / 100
curve1 = [((p1["actual_breakdown_next_30d"].head(int(round(len(p1) * f))).sum() * A["breakdown_avoided_share"] * bd_total) - int(round(len(p1) * f)) * pv_cost) / 1e6 for f in ks1]

# ================================================================== OBJECTIVE 2: margin
pr = clean["fact_production"][["run_id", "throughput_bbl", "crude_grade"]]
p2 = pred[2].merge(pr, on="run_id", how="left").dropna(subset=["actual_margin", "throughput_bbl"])
col2 = f"{slug(best[2])}_prediction"
err = (p2[col2] - p2["actual_margin"]).abs()
mae = err.mean(); thr = p2["throughput_bbl"].mean()
ev2 = pd.read_pickle("obj2_evaluation.pkl")["results"] if os.path.exists("obj2_evaluation.pkl") else None
mae_naive = float(ev2.loc[ev2["model"].str.contains("last week"), "MAE"].iloc[0]) if ev2 is not None else 1.773
p2["q"] = pd.qcut(p2[col2], 4, labels=["Lowest 25%", "Low-mid", "High-mid", "Highest 25%"])
byq = p2.groupby("q", observed=True).agg(predicted=(col2, "mean"), actual=("actual_margin", "mean"), days=("run_id", "size"))
spread = byq["actual"].iloc[-1] - byq["actual"].iloc[0]
d2 = days(p2["run_date"])
dollar_err = mae * thr
print("=" * 100); print("OBJECTIVE 2: MAXIMISE REFINERY YIELD AND GROSS MARGIN")
print(f"{best[2]} is off by {mae:.2f} USD/bbl on average. On a typical run of {thr:,.0f} bbl that is about {usd(dollar_err)} of margin per run-day.")
print(f"Planning with last week's margin would be off by {mae_naive:.2f} USD/bbl, so the model cuts forecast error by {1 - mae/mae_naive:.0%}.")
print("Do predictions rank days correctly? Actual margin by predicted quartile:")
print(byq.round(2).to_string())
print(f"  The model's top-quarter days really earn {spread:.2f} USD/bbl more than its bottom-quarter days, so its ranking can be trusted.")
print("  Much of that spread is crude choice (see the table below), which is the lever planners control.")
by_crude = p2.groupby("crude_grade").agg(actual_margin=("actual_margin", "mean"), predicted_margin=(col2, "mean"), run_days=("run_id", "size")).round(2).sort_values("actual_margin", ascending=False)
print("\nMargin by crude grade (test period):"); print(by_crude.to_string())
summary.append({"objective": "2 Margin", "action": "plan crude and settings with the margin forecast (decision support, not valued)", "units_acted_on": len(p2),
                "problems_in_group": np.nan, "gross_value_usd": np.nan, "action_cost_usd": np.nan, "net_value_usd": np.nan, "test_days": d2})

# ================================================================== OBJECTIVE 3: late delivery
dl = clean["fact_deliveries"][["trip_id", "freight_cost_usd"]]
p3 = pred[3].dropna(subset=["actual_late"]).sort_values(f"{slug(best[3])}_probability", ascending=False)
k3 = int(round(len(p3) * PLAN[3])); c3 = int(p3["actual_late"].head(k3).sum()); t3 = int(p3["actual_late"].sum())
sav3 = c3 * A["late_avoided_share"] * A["late_penalty_per_trip_usd"]; cost3 = k3 * A["late_action_cost_usd"]; net3 = sav3 - cost3
rand3 = t3 * PLAN[3] * A["late_avoided_share"] * A["late_penalty_per_trip_usd"] - cost3
d3 = days(p3["dispatch_time"])
print("=" * 100); print("OBJECTIVE 3: RAISE ON-TIME DELIVERY AND CUT DISTRIBUTION COST")
print(f"Test window {d3} days, {len(p3):,} trips, {t3} late ({t3/len(p3):.0%}). Average freight cost per trip in the data: {usd(dl['freight_cost_usd'].mean())}.")
print(f"Plan: add a buffer or change the route on the riskiest {PLAN[3]:.0%} of trips ({k3} trips).")
print(f"  real late trips inside that list: {c3} of {t3} ({c3/t3:.0%}); on-time rate if {A['late_avoided_share']:.0%} of them are saved: "
      f"{1 - (t3 - c3 * A['late_avoided_share']) / len(p3):.1%} (now {1 - t3/len(p3):.1%})")
print(f"  penalties avoided {usd(sav3)} - cost of actions {usd(cost3)} = net {usd(net3)}   (random picking: {usd(rand3)})")
summary.append({"objective": "3 Delivery", "action": f"buffer/re-route top {PLAN[3]:.0%} risk trips", "units_acted_on": k3,
                "problems_in_group": c3, "gross_value_usd": sav3, "action_cost_usd": cost3, "net_value_usd": net3, "test_days": d3})
by_region = (p3.merge(clean["fact_deliveries"][["trip_id", "destination_region"]], on="trip_id", how="left").groupby("destination_region")["actual_late"]
             .agg(["mean", "size"]).rename(columns={"mean": "late_rate"}).sort_values("late_rate", ascending=False))
print("\nLate rate by destination region (test period):"); print((by_region.assign(late_rate=(by_region["late_rate"] * 100).round(1))).to_string())

# ================================================================== OBJECTIVE 4: off-spec
q = clean["fact_quality_tests"]; rw = q.loc[q["spec_result"] == "Off-spec", "rework_cost_usd"].mean()
p4 = pred[4].dropna(subset=["actual_off_spec"]).sort_values(f"{slug(best[4])}_probability", ascending=False)
k4 = int(round(len(p4) * PLAN[4])); c4 = int(p4["actual_off_spec"].head(k4).sum()); t4 = int(p4["actual_off_spec"].sum())
sav4 = c4 * A["rework_avoided_share"] * rw; cost4 = k4 * A["quality_check_cost_usd"]; net4 = sav4 - cost4
rand4 = t4 * PLAN[4] * A["rework_avoided_share"] * rw - cost4
d4 = days(p4["test_date"])
print("=" * 100); print("OBJECTIVE 4: REDUCE OFF-SPEC PRODUCT AND REWORK COST")
print(f"Each off-spec batch costs {usd(rw)} in rework on average (from the data). Test window {d4} days, {len(p4)} batches, {t4} off-spec ({t4/len(p4):.1%}).")
print(f"Plan: extra check and blend correction on the riskiest {PLAN[4]:.0%} of batches ({k4} batches).")
print(f"  real off-spec batches inside that list: {c4} of {t4} ({c4/t4:.0%}); rework avoided {usd(sav4)} - cost of checks {usd(cost4)} = net {usd(net4)}   (random picking: {usd(rand4)})")
summary.append({"objective": "4 Quality", "action": f"extra check on top {PLAN[4]:.0%} risk batches", "units_acted_on": k4,
                "problems_in_group": c4, "gross_value_usd": sav4, "action_cost_usd": cost4, "net_value_usd": net4, "test_days": d4})
breakeven4 = A["quality_check_cost_usd"] / (A["rework_avoided_share"] * rw)
print(f"  A check only pays for itself when the flagged group holds more than {breakeven4:.1%} off-spec batches (the plain off-spec rate is {t4/len(p4):.1%}).")

# ================================================================== OBJECTIVE 5: inventory
inv = clean["fact_inventory"][["record_id", "book_closing_kl", "measured_closing_kl", "dispatches_kl"]].copy()
inv["gap_kl"] = inv["measured_closing_kl"] - inv["book_closing_kl"]
p5 = pred[5].merge(inv, on="record_id", how="left")
p5["abs_gap_kl"] = p5["gap_kl"].abs()
p5["abs_gap_pct_book"] = p5["abs_gap_kl"] / p5["book_closing_kl"] * 100
rec_err = p5["abs_gap_pct_book"] > 50                          # a tank cannot lose 50%+ of its stock overnight: a bad reading or entry
real = ~rec_err
methods = {"Isolation Forest": p5["isolation_forest_flag"] == 1, "Local Outlier Factor": p5["local_outlier_factor_flag"] == 1,
           "One-Class SVM": p5["one_class_svm_flag"] == 1, "Rule: gap over 1%": p5["gap_pct"].abs() > 1}
big_real = (p5["gap_pct"].abs() > 1) & real
tot_big = p5.loc[big_real, "abs_gap_kl"].sum()
rows5 = []
for name, f in methods.items():
    vol = p5.loc[f & real, "abs_gap_kl"].sum()
    rows5.append({"method": name, "tank_days_flagged": int(f.sum()), "recording_errors_caught": int((f & rec_err).sum()),
                  "real_gap_kl_in_flags": round(vol), "share_of_real_big_gap_volume_%": round(p5.loc[f & big_real, "abs_gap_kl"].sum() / tot_big * 100, 1),
                  "value_usd": vol * A["fuel_value_usd_per_kl"], "investigation_cost_usd": f.sum() * A["investigation_cost_usd"]})
t5 = pd.DataFrame(rows5)
d5 = days(p5["record_date"])
print("=" * 100); print("OBJECTIVE 5: DETECT PRODUCT LOSS, THEFT AND METERING ERRORS")
print(f"{int(rec_err.sum())} tank-days have a measured stock more than 50% away from book (e.g. a reading 10 times too high). These are recording errors,")
print("not product moving, so they are counted separately and kept out of the volume value. They still need fixing, because they corrupt stock reports.")
print(f"Remaining tank-days over {d5} days: absolute gap {p5.loc[real, 'abs_gap_kl'].sum():,.0f} kl; net shortage (measured below book) {-p5.loc[real & (p5['gap_kl'] < 0), 'gap_kl'].sum():,.0f} kl.")
print(f"Volume in real tank-days with a gap above 1%: {tot_big:,.0f} kl, worth about {usd(tot_big * A['fuel_value_usd_per_kl'])} at {usd(A['fuel_value_usd_per_kl'])}/kl.")
print(f"For scale: that is {tot_big / p5['dispatches_kl'].sum() * 100:.1f}% of all product dispatched from these tanks in the year.")
print("This is volume put in front of investigators, not volume recovered; much of it is metering drift and timing rather than theft.")
print(t5.assign(value_usd=t5["value_usd"].map(usd), investigation_cost_usd=t5["investigation_cost_usd"].map(usd)).to_string(index=False))
ml_best = t5.loc[t5["method"] == best[5]].iloc[0]; rule = t5.loc[t5["method"] == "Rule: gap over 1%"].iloc[0]
print(f"\nAnalyst's read: the gap rule puts {rule['share_of_real_big_gap_volume_%']:.0f}% of the large real-gap volume in front of investigators; {best[5]} puts {ml_best['share_of_real_big_gap_volume_%']:.0f}%.")
print("Recommendation: run the 1% gap rule as the daily control, and keep the model as a second opinion for patterns that sit under the rule's threshold.")
summary.append({"objective": "5 Inventory", "action": "investigate tank-days flagged by the gap rule (volume reviewed, not recovered)", "units_acted_on": int(rule["tank_days_flagged"]),
                "problems_in_group": np.nan, "gross_value_usd": np.nan, "action_cost_usd": rule["investigation_cost_usd"],
                "net_value_usd": np.nan, "test_days": d5})

# ================================================================== roll-up
S = pd.DataFrame(summary)
S["net_annualised_usd"] = S["net_value_usd"] * 365 / S["test_days"]    # NaN for objectives 2 and 5 (not valued in dollars)
print("=" * 100); print("ROLL-UP (values inside the test window; annualised = scaled to 365 days, indicative only)")
show = S.copy()
for c in ["gross_value_usd", "action_cost_usd", "net_value_usd", "net_annualised_usd"]:
    show[c] = show[c].map(lambda v: usd(v) if pd.notna(v) else "not valued")
print(show.to_string(index=False))
print(f"\nNet value, annualised, objectives 1, 3 and 4: {usd(S['net_annualised_usd'].sum())}")
print("Objective 2 (forecasting) and Objective 5 (investigation volume) are decision support and get no dollar saving, because the data cannot show what would actually be recovered.")

# ================================================================== figure
C = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "font.size": 10,
                     "axes.titlesize": 11.5, "axes.titleweight": "bold"})
fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
a = ax[0, 0]; a.plot(ks1 * 100, curve1, color=C[0], lw=2); a.axvline(PLAN[1] * 100, ls=":", color="#555")
a.axhline(0, color="#999", lw=1); a.set(title="1 Maintenance: net value by share of asset-months serviced", xlabel="% of asset-months serviced (riskiest first)", ylabel="net value, USD millions")
a = ax[0, 1]; a.bar(byq.index.astype(str), byq["actual"], color=C[1]); a.set_ylim(byq["actual"].min() - 1.2, byq["actual"].max() + .8)
for i, v in enumerate(byq["actual"]): a.text(i, v + .08, f"{v:.1f}", ha="center")
a.set(title="2 Margin: actual margin by predicted quartile", ylabel="USD per barrel")
a = ax[0, 2]; a.bar(["on time\nnow", "on time with\nthe plan"], [(1 - t3 / len(p3)) * 100, (1 - (t3 - c3 * A["late_avoided_share"]) / len(p3)) * 100], color=[C[3], C[2]])
a.set_ylim(60, 100); [a.text(i, v + .5, f"{v:.1f}%", ha="center") for i, v in enumerate([(1 - t3 / len(p3)) * 100, (1 - (t3 - c3 * A["late_avoided_share"]) / len(p3)) * 100])]
a.set(title="3 Delivery: on-time rate", ylabel="%")
a = ax[1, 0]; vals = [sav4, cost4, net4]; a.bar(["rework\navoided", "cost of\nchecks", "net"], vals, color=[C[1], C[4], C[0]])
[a.text(i, v, usd(v), ha="center", va="bottom") for i, v in enumerate(vals)]; a.set(title=f"4 Quality: checking the riskiest {PLAN[4]:.0%} of batches", ylabel="USD")
a = ax[1, 1]; a.barh(t5["method"], t5["share_of_real_big_gap_volume_%"], color=[C[0], C[4], C[1], "#999999"]); a.invert_yaxis()
[a.text(v + 1, i, f"{v:.0f}%", va="center") for i, v in enumerate(t5["share_of_real_big_gap_volume_%"])]
a.set(title="5 Inventory: share of real large-gap volume flagged", xlabel="%", xlim=(0, 110))
V = S[S["net_annualised_usd"].notna()]
a = ax[1, 2]; a.bar(V["objective"].str.slice(0, 1), V["net_annualised_usd"] / 1e6, color=[C[0], C[3], C[4]])
[a.text(i, v / 1e6, f"{v/1e6:.2f}", ha="center", va="bottom") for i, v in enumerate(V["net_annualised_usd"])]
a.set(title="Net value, annualised, objectives 1, 3, 4 (USD m, uses assumptions)", xlabel="objective")
fig.suptitle("Meridian Downstream: what the models are worth to the business", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("business_analysis.png", dpi=140, bbox_inches="tight"); plt.show()

S.to_csv("business_analysis_summary.csv", index=False)
pd.to_pickle({"summary": S, "assumptions": A, "plans": PLAN, "best_models": best, "inventory_methods": t5, "margin_by_quartile": byq}, "business_analysis.pkl")
print("\nSaved: business_analysis.png, business_analysis_summary.csv, business_analysis.pkl")
```

---

## 19. Save Model, Scaler and Features

Everything is saved with `joblib`: for each objective the chosen model, the fitted scaler (with its imputer) and the exact feature list, plus a `model_registry.joblib` and a small `meridian_predict.py` helper.

| Objective | Task | Saved model | Inputs | Scaled columns | Size |
|---|---|---|---|---|---|
| 1 | Classification | Logistic Regression | 51 | 20 | 0.01 MB |
| 2 | Regression | Ridge Regression | 33 | 17 | 0.01 MB |
| 3 | Classification | Random Forest | 47 | 11 | 1.94 MB |
| 4 | Classification | Logistic Regression | 37 | 9 | 0.01 MB |
| 5 | Anomaly | Isolation Forest | 16 | 18 | 0.75 MB |

**Verified, not assumed.** The script reloads every file from disk, rebuilds features from raw rows and reproduces the Step 15 predictions for all five objectives. All checks passed.

### Code: save model, scaler and features

```python
# STEP 17: SAVE MODEL, SCALER AND FEATURES
# For each of the 5 objectives this saves the three things needed to score new data later:
#   objN_model.joblib      the chosen trained model (the winner from Step 14)
#   objN_scaler.joblib     the StandardScaler fitted on the training rows
#   objN_features.joblib    feature names in the exact order, which columns are scaled, the fill values, and the model input columns
# It also writes meridian_predict.py (a small helper that applies them to new data), model_registry.joblib (what each model is,
# its test results and its decision cut-off), and then reloads everything from disk to prove it reproduces the saved predictions.
import os
import sys
import warnings
import datetime
import importlib
import joblib
import numpy as np
import pandas as pd
import sklearn

warnings.filterwarnings("ignore")
PKG = "meridian_model_package"
os.makedirs(PKG, exist_ok=True)

def get(name, path):
    g = globals()
    if name in g:
        return g[name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing. Run the earlier step that creates it, then re-run this cell.")
    return pd.read_pickle(path)

scaled = get("scaled", "meridian_scaled.pkl")
split = get("split", "meridian_split.pkl")
ABT = {1: "abt_1_maintenance", 2: "abt_2_margin", 3: "abt_3_delivery", 4: "abt_4_quality", 5: "abt_5_inventory"}
TASK = {1: "classification", 2: "regression", 3: "classification", 4: "classification", 5: "anomaly"}
TARGET = {1: "breakdown_next_30d (next month)", 2: "gross_margin_usd_per_bbl", 3: "late_delivery", 4: "off_spec", 5: "none (unsupervised)"}
CUT_KEY = {1: "cutoff_top10", 3: "cutoff_top20", 4: "cutoff_top20"}
CUT_RULE = {1: "riskiest 10% of asset-months", 3: "riskiest 20% of trips", 4: "riskiest 20% of batches"}

registry = {"created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__, "objectives": {}}
rows = []
for n in range(1, 6):
    for p in (f"obj{n}_models.joblib", f"obj{n}_evaluation.pkl"):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} is missing. Run Step 11 / Step 14 for Objective {n} first, then re-run this cell.")
    bundle = joblib.load(f"obj{n}_models.joblib")
    ev = pd.read_pickle(f"obj{n}_evaluation.pkl")
    best = ev["best_model"]
    model = bundle["models"][best]
    o = scaled[ABT[n]]
    scaler = o["scaler"]
    model_cols = list(bundle["feature_names"]) if n < 5 else list(bundle["feature_names"])       # columns the model was trained on, in order

    joblib.dump(model, f"{PKG}/obj{n}_model.joblib", compress=3)
    joblib.dump(scaler, f"{PKG}/obj{n}_scaler.joblib")
    feats = {"objective": n, "feature_names": list(o["feature_names"]), "scaled_columns": list(o["scaled_columns"]),
             "binary_columns": list(o["binary_columns"]), "model_input_columns": model_cols,
             "fill_values": o["fill_values"].copy()}
    joblib.dump(feats, f"{PKG}/obj{n}_features.joblib")

    r = ev["results"]; r = r[r["model"] == best].iloc[0].to_dict()
    r = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
    entry = {"task": TASK[n], "target": TARGET[n], "chosen_model": best, "n_model_inputs": len(model_cols), "test_results": r}
    if n in CUT_KEY:
        entry["decision_cutoff"] = float(ev[CUT_KEY[n]]); entry["cutoff_meaning"] = f"flag when probability is at or above the cut-off = {CUT_RULE[n]} on the test set"
    if n == 5:
        entry["note"] = "Flag = model.predict(x) == -1. A plain rule (abs gap_pct above 1) scored higher in Step 14; run both."
        entry["simple_rule"] = "abs(gap_pct) > 1"
    registry["objectives"][str(n)] = entry
    size = sum(os.path.getsize(f"{PKG}/obj{n}_{x}") for x in ("model.joblib", "scaler.joblib", "features.joblib")) / 1e6
    rows.append({"objective": n, "task": TASK[n], "chosen_model": best, "model_inputs": len(model_cols), "scaled_cols": len(o["scaled_columns"]), "size_MB": round(size, 2)})

joblib.dump(registry, f"{PKG}/model_registry.joblib")

HELPER = '''"""Score new Meridian data with the saved models.
Input: a DataFrame that already went through cleaning, feature engineering and one-hot encoding (the same columns as the training
data). Missing columns are added as 0/NaN, extra columns are ignored.
    import meridian_predict as mp
    out = mp.predict(df, objective=3)      # 1 maintenance, 2 margin, 3 late delivery, 4 off-spec, 5 inventory anomalies
"""
import os
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

def load(n):
    model = joblib.load(os.path.join(HERE, f"obj{n}_model.joblib"))
    scaler = joblib.load(os.path.join(HERE, f"obj{n}_scaler.joblib"))
    feats = joblib.load(os.path.join(HERE, f"obj{n}_features.joblib"))
    return model, scaler, feats

def prepare(df, n):
    model, scaler, feats = load(n)
    x = df.reindex(columns=feats["feature_names"]).replace([np.inf, -np.inf], np.nan)
    fill = feats["fill_values"]
    x = x.fillna(fill)
    x[feats["scaled_columns"]] = scaler.transform(x[feats["scaled_columns"]])
    return x[feats["model_input_columns"]]

def predict(df, objective):
    n = objective
    model, scaler, feats = load(n)
    reg = joblib.load(os.path.join(HERE, "model_registry.joblib"))["objectives"][str(n)]
    x = prepare(df, n)
    out = pd.DataFrame(index=df.index)
    if reg["task"] == "classification":
        out["probability"] = model.predict_proba(x)[:, 1]
        out["flag"] = (out["probability"] >= reg["decision_cutoff"]).astype(int)
    elif reg["task"] == "regression":
        out["predicted_margin_usd_per_bbl"] = model.predict(x)
    else:
        out["anomaly_score"] = -model.decision_function(x)      # higher = more unusual
        out["flag"] = (model.predict(x) == -1).astype(int)
    return out
'''
open(f"{PKG}/meridian_predict.py", "w").write(HELPER)

print("SAVED PACKAGE")
print(pd.DataFrame(rows).to_string(index=False))

# ---- prove it works: reload from disk and rebuild predictions from the UNSCALED data
sys.path.insert(0, os.path.abspath(PKG))
import meridian_predict as mp
importlib.reload(mp)
print("\nCHECK: reload from disk and reproduce the saved predictions (Step 12 files are rounded to 3-4 decimals, so the match is within that rounding)")
ok_all = True
for n in range(1, 6):
    s = split[ABT[n]]
    raw = s["X_test"] if n < 5 else s["X_all"]
    x = mp.prepare(raw, n)
    ref = scaled[ABT[n]]["X_test"] if n < 5 else scaled[ABT[n]]["X_train"]
    same_x = np.allclose(x.values, ref[x.columns].values, atol=1e-9)
    out = mp.predict(raw, n)
    best = registry["objectives"][str(n)]["chosen_model"]; slug = best.lower().replace(" ", "_").replace("-", "_")
    if n == 1: pf = pd.read_pickle("obj1_test_predictions.pkl"); same_y = np.allclose(out["probability"], pf[f"{slug}_probability"], atol=1e-4)
    elif n == 2: pf = pd.read_pickle("obj2_test_predictions.pkl"); same_y = np.allclose(out.iloc[:, 0], pf[f"{slug}_prediction"], atol=1e-3)
    elif n in (3, 4): pf = pd.read_pickle(f"obj{n}_test_predictions.pkl"); same_y = np.allclose(out["probability"], pf[f"{slug}_probability"], atol=1e-4)
    else: pf = pd.read_pickle("obj5_scores.pkl"); same_y = bool((out["flag"].values == pf["isolation_forest_flag"].values).all())
    ok_all &= bool(same_x and same_y)
    print(f"  Objective {n}: features rebuilt from raw = {same_x} | predictions identical to Step 12 = {same_y}")
print("\nAll checks passed." if ok_all else "\nA CHECK FAILED: do not use this package until it is fixed.")
print(f"Files in {PKG}/:", ", ".join(sorted(f for f in os.listdir(PKG) if f != "__pycache__")))
```

---

## 20. Download Model, Scaler and Features

The 17 files are zipped into `meridian_model_package.zip` (about 2.7 MB, so no GitHub size problem) for download. Unzip it as the `model` folder next to `app.py`.

### Code: download model, scaler and features

```python
# STEP 18: DOWNLOAD MODEL, SCALER AND FEATURES
# Zips the folder made in Step 17 (models, scalers, features, registry and helper) into one file and downloads it
# to your computer. Check the file list printed below before downloading.
import os
import zipfile

PKG = "meridian_model_package"
ZIP = "meridian_model_package.zip"

if not os.path.isdir(PKG):
    raise FileNotFoundError(f"The folder {PKG} was not found. Run Step 17 first, then re-run this cell.")
expected = [f"obj{n}_{x}.joblib" for n in range(1, 6) for x in ("model", "scaler", "features")] + ["model_registry.joblib", "meridian_predict.py"]
missing = [f for f in expected if not os.path.exists(os.path.join(PKG, f))]
if missing:
    raise FileNotFoundError("These files are missing from the package, re-run Step 17: " + ", ".join(missing))

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in expected:
        z.write(os.path.join(PKG, f), arcname=f"{PKG}/{f}")

print(f"{ZIP}  ({os.path.getsize(ZIP) / 1e6:.2f} MB) contains:")
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None, "the zip file is damaged"
    for i in z.infolist():
        print(f"  {i.filename:<48} {i.file_size / 1e3:>9,.1f} KB")

try:
    from google.colab import files
    files.download(ZIP)                       # your browser may ask permission to download
    print("\nDownload started. If nothing happens, allow downloads for this site, or use the Files panel on the left: right-click the zip and choose Download.")
except ImportError:
    print(f"\nNot running in Colab. The zip is saved here: {os.path.abspath(ZIP)}")
```


### The Streamlit app

The saved files power [`app.py`](app.py), a single file with seven pages:

- **Home:** the five objectives, the model behind each and how far to trust it.
- **1 Maintenance risk:** probability an asset breaks down next month, a risk level and a suggested action.
- **2 Margin forecast:** predicted gross margin in USD per barrel from the crude, plant conditions and market prices.
- **3 Delivery risk:** probability a trip arrives late and a suggested action.
- **4 Off-spec risk:** probability a batch fails specification and a suggested action.
- **5 Inventory check:** compares measured and book stock, shows the gap rule (over 1%) beside the model, and warns on readings more than 50% from book.
- **Prediction log:** every score from the session, downloadable as CSV.

The app finds the model folder on its own, whatever it is called. Risk levels use cut-offs from the test period, so "High" means "in the riskiest group", not "certain".

**Run it yourself**

```bash
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` pins `scikit-learn==1.6.1`, the version that trained the saved files, so loading raises no version warning.

Live app: [Open the Streamlit app](https://meridiandownstreamenergyoperationsandriskpredictivesupportsys.streamlit.app/)

### Full code: app.py

```python
"""
Meridian Downstream Energy: Operations and Risk Predictive Support System
==========================================================================
A simple Streamlit app with one page per business objective:

    1  Maintenance risk   -> will this asset break down next month?
    2  Margin forecast    -> what gross margin will this refinery run earn?
    3  Delivery risk      -> will this fuel trip arrive late?
    4  Off-spec risk      -> will this blend batch fail specification?
    5  Inventory check    -> is this tank-day a possible loss, theft or metering error?

HOW TO RUN
    1. Put this file (app.py) in a folder, and put the files from your Step 18 download in a folder next to it.
       The folder can have any name (model, models, meridian_model_package...); the app finds it by itself:

           meridian_app/
               app.py
               model/                       <- the .joblib files from your Step 18 download
                   obj1_model.joblib, obj1_scaler.joblib, obj1_features.joblib ... (and so on)
                   model_registry.joblib

    2. In the VS Code terminal:
           pip install streamlit pandas numpy joblib scikit-learn==1.6.1
           streamlit run app.py

How it works (short version): you type in the things a person would know (readings, distances, volumes).
The app turns them into the exact columns the saved model was trained on, fills anything you did not
enter with the typical value from the training data, scales the numbers with the saved scaler, and asks
the saved model for its answer. The scores are decision support: they rank risk, they do not guarantee outcomes.
"""
import os
import calendar
import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# ----------------------------------------------------------------------------------------------
# 1. SETTINGS you may want to change
# ----------------------------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def find_model_folder():
    """Finds the folder that holds obj1_model.joblib, wherever you put it (named model, models, meridian_model_package...).
    Looks next to app.py and up to two folders deep."""
    for root, folders, files in os.walk(APP_DIR):
        if root[len(APP_DIR):].count(os.sep) > 2:
            folders[:] = []
            continue
        if "obj1_model.joblib" in files and "model_registry.joblib" in files:
            return root
    return None


MODEL_DIR = find_model_folder()

# Typical values used in the explanations. They come from the project data (Steps 15 and 19).
# Dollar figures marked "assumed" were assumptions in Step 15: replace them with your real costs.
COST = {
    "breakdown_repair_usd": 29842,   # average repair cost of a breakdown (from the data)
    "breakdown_downtime_h": 25,      # average downtime hours of a breakdown (from the data)
    "preventive_job_usd": 1941,      # average preventive job cost (from the data)
    "late_penalty_usd": 600,         # assumed penalty per late trip
    "rework_usd": 11699,             # average rework cost of an off-spec batch (from the data)
    "quality_check_usd": 250,        # assumed cost of an extra check
}
# Risk scores at which a case moves up a level. "Act now" = the action level saved with each model (the cases the project plan acts on).
# "Watch" = the next tier below it (roughly the top 20-30% of cases in the test data). Taken from the test results in Step 12.
WATCH_LEVEL = {1: 0.657, 3: 0.356, 4: 0.480}
AVERAGE_MARGIN = 11.5                # average gross margin in the data, USD per barrel
TYPICAL_DOSE_PPM, TYPICAL_BLEND_TEMP_C = 199.7, 38.0   # typical settings in the training period (Step 3)

st.set_page_config(page_title="Meridian Operations and Risk Support", page_icon="⛽", layout="wide")


# ----------------------------------------------------------------------------------------------
# 2. LOAD THE SAVED MODELS (once, then kept in memory)
# ----------------------------------------------------------------------------------------------
@st.cache_resource
def load_package():
    """Reads model, scaler and feature list for each objective, plus the registry."""
    package = {}
    for n in range(1, 6):
        package[n] = (joblib.load(os.path.join(MODEL_DIR, f"obj{n}_model.joblib")),
                      joblib.load(os.path.join(MODEL_DIR, f"obj{n}_scaler.joblib")),
                      joblib.load(os.path.join(MODEL_DIR, f"obj{n}_features.joblib")))
    registry = joblib.load(os.path.join(MODEL_DIR, "model_registry.joblib"))
    return package, registry


if MODEL_DIR is None:
    st.error(f"The saved models were not found. I looked in {APP_DIR} and the folders inside it for obj1_model.joblib and "
             "model_registry.joblib. Put the files from your Step 18 download (the .joblib files) in a folder next to app.py.")
    st.stop()
PACKAGE, REGISTRY = load_package()


# ----------------------------------------------------------------------------------------------
# 3. HELPERS shared by all pages
# ----------------------------------------------------------------------------------------------
def category(feats, label, prefix, key, exclude=(), default=None):
    """Drop-down for a category the model knows (made from its one-hot columns, e.g. country_Nigeria).
    Returns the chosen name and a dict of 0/1 values for those columns. 'Other' sets them all to 0."""
    cols = [c for c in feats["feature_names"] if c.startswith(prefix + "_") and not c.startswith(tuple(exclude))]
    names = [c[len(prefix) + 1:].replace("_", " ") for c in cols]
    index = names.index(default) if default in names else 0
    choice = st.selectbox(label, names + ["Other / not listed"], index=index, key=key)
    return choice, {c: float(n == choice) for c, n in zip(cols, names)}


def run_model(n, values):
    """Builds the one-row table the model expects. Anything not entered gets the typical training value."""
    model, scaler, feats = PACKAGE[n]
    row = {**feats["fill_values"], **values}
    X = pd.DataFrame([row])[feats["feature_names"]]
    X[feats["scaled_columns"]] = scaler.transform(X[feats["scaled_columns"]])    # same scaling as training
    return model, X[feats["model_input_columns"]]


def risk_score(n, values):
    """Risk score from 0 to 1 for the classification models (objectives 1, 3, 4)."""
    model, X = run_model(n, values)
    return float(model.predict_proba(X)[0, 1])


def cutoff(n):
    return float(REGISTRY["objectives"][str(n)]["decision_cutoff"])


def show_risk(n, score, factors, advice_high, advice_watch, advice_ok):
    """Shows the risk score, the action level, a verdict and the warning signs found in the inputs."""
    cut = cutoff(n)
    level = "high" if score >= cut else ("watch" if score >= WATCH_LEVEL[n] else "normal")
    a, b, c = st.columns(3)
    a.metric("Risk score", f"{score:.0%}")
    b.metric("Action level", f"{cut:.0%}")
    c.metric("Verdict", {"high": "Act now", "watch": "Watch", "normal": "Normal"}[level])
    st.progress(min(max(score, 0.0), 1.0))
    if level == "high":
        st.error(advice_high)
    elif level == "watch":
        st.warning(advice_watch)
    else:
        st.success(advice_ok)
    st.markdown("**Warning signs in these inputs:** " + ("; ".join(factors) if factors else "none of the usual warning signs"))
    st.caption("The risk score ranks cases from low to high. It is not an exact probability. "
               f"A score at or above {cut:.0%} puts the case in the group that is acted on in the project plan; "
               f"{WATCH_LEVEL[n]:.0%} to {cut:.0%} is the watch tier.")
    return {"high": "Act now", "watch": "Watch", "normal": "Normal"}[level]


def add_to_log(objective, case, result, verdict):
    if "log" not in st.session_state:
        st.session_state["log"] = []
    st.session_state["log"].append({"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "objective": objective,
                                    "case": case, "result": result, "verdict": verdict})


def feats_of(n):
    return PACKAGE[n][2]


TODAY = datetime.date.today()


# ----------------------------------------------------------------------------------------------
# 4. PAGES
# ----------------------------------------------------------------------------------------------
def page_home():
    st.title("⛽ Meridian Downstream Energy")
    st.subheader("Operations and Risk Predictive Support System")
    st.write("Five prediction tools built from the Meridian machine learning project. Pick one in the menu on the left, "
             "enter the details, and press the button to get a score and a suggested action.")
    rows = []
    names = {1: "Cut unplanned downtime and maintenance cost", 2: "Maximise refinery yield and gross margin",
             3: "Raise on-time delivery and cut distribution cost", 4: "Reduce off-spec product and rework cost",
             5: "Detect product loss, theft and metering errors"}
    for n in range(1, 6):
        o = REGISTRY["objectives"][str(n)]
        r = o["test_results"]
        if n == 2:
            res = f"error {r['MAE']:.2f} USD/bbl, R2 {r['R2']:.2f}"
        else:
            res = f"ROC AUC {r['ROC_AUC']:.2f}, PR AUC {r['PR_AUC']:.2f}"
        rows.append({"#": n, "Business objective": names[n], "Model": o["chosen_model"], "Test result": res})
    st.table(pd.DataFrame(rows).set_index("#"))
    with st.expander("Please read: how to use these scores"):
        st.markdown(
            "- The scores support decisions; engineers, planners and supervisors still make them.\n"
            "- Breakdown and off-spec models are **ranking tools**: they are right more often than chance, not every time.\n"
            "- For inventory, the simple **gap over 1%** rule was more accurate than the machine learning model, so the page shows both.\n"
            "- The project data is synthetic, and the money figures use assumed costs (see the `COST` box at the top of app.py).")
    saved = REGISTRY.get("scikit_learn_version")
    if saved and saved != sklearn.__version__:
        st.warning(f"The models were saved with scikit-learn {saved}, but this computer has {sklearn.__version__}. "
                   f"If you see errors, run: pip install scikit-learn=={saved}")


# ---------------------------------------------------------------- Objective 1
def page_maintenance():
    st.header("1. Maintenance risk")
    st.write("Will this asset have a breakdown next month? Enter its latest sensor readings and history.")
    f = feats_of(1)
    with st.form("form1"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**The asset**")
            atype, d_type = category(f, "Asset type", "asset_type", "a_type", default="Pump")
            crit, d_crit = category(f, "Criticality (A = most critical)", "criticality", "a_crit", default="B")
            site, d_site = category(f, "Site type", "site_type", "a_site", default="Terminal")
            country, d_country = category(f, "Country", "country", "a_country", default="Nigeria")
            maker, d_maker = category(f, "Manufacturer", "manufacturer", "a_maker", default="Siemens")
            age = st.number_input("Asset age (years)", 0.0, 60.0, 13.0, 0.5)
            life = st.number_input("Design life (years)", 1.0, 60.0, 25.0, 1.0)
        with c2:
            st.markdown("**Latest sensor readings**")
            vib = st.number_input("Vibration (mm/s)", 0.0, 30.0, 3.1, 0.1)
            vib3 = st.number_input("Vibration, 3-month average (mm/s)", 0.0, 30.0, 3.1, 0.1)
            temp = st.number_input("Temperature (°C)", 0.0, 250.0, 75.6, 0.5)
            temp3 = st.number_input("Temperature, 3-month average (°C)", 0.0, 250.0, 75.5, 0.5)
            press = st.number_input("Pressure (bar)", 0.0, 60.0, 12.5, 0.5)
            lube = st.number_input("Lubricant condition score (0-100)", 0.0, 100.0, 83.0, 1.0)
        with c3:
            st.markdown("**Recent history**")
            hours = st.number_input("Operating hours this month", 0.0, 744.0, 650.0, 10.0)
            month = st.number_input("Month of year (1-12)", 1, 12, TODAY.month, 1)
            bd6 = st.number_input("Breakdowns in the last 6 months", 0, 10, 0, 1)
            pm6 = st.number_input("Preventive jobs in the last 6 months", 0, 10, 0, 1)
            since = st.number_input("Months since last breakdown (36 = none)", 0.0, 36.0, 36.0, 1.0)
        go = st.form_submit_button("Check this asset")
    if not go:
        return
    values = {**d_type, **d_crit, **d_site, **d_country, **d_maker,
              "asset_age_years": age, "life_used_ratio": age / life,
              "avg_vibration_mm_s": vib, "vib_avg_3m": vib3, "avg_temperature_c": temp, "temp_avg_3m": temp3,
              "avg_pressure_bar": press, "press_avg_3m": press, "lubricant_condition_score": lube, "lube_avg_3m": lube,
              "operating_hours": hours, "utilisation_pct": hours / (calendar.monthrange(TODAY.year, int(month))[1] * 24) * 100,
              "month_of_year": month, "breakdowns_last_6m": bd6, "preventive_jobs_last_6m": pm6, "months_since_last_breakdown": since,
              "vib_alert": float(vib >= 4.5), "lube_poor": float(lube < 60)}
    factors = []
    if temp3 >= 79.8: factors.append("high 3-month temperature (the hottest quarter of assets break down about 7x as often)")
    if vib >= 4.5: factors.append("vibration alert (about 3.7x the normal breakdown rate)")
    if lube < 60: factors.append("poor lubricant condition (about 3.4x)")
    if age >= 19: factors.append("old asset (about 1.6x)")
    if bd6 > 0: factors.append(f"{int(bd6)} breakdown(s) in the last 6 months")
    score = risk_score(1, values)
    verdict = show_risk(1, score, factors,
                        f"Schedule a preventive inspection or job now. A preventive job costs about ${COST['preventive_job_usd']:,}; "
                        f"a breakdown costs about ${COST['breakdown_repair_usd']:,} plus roughly {COST['breakdown_downtime_h']} hours of downtime.",
                        "Elevated risk. Add this asset to the next inspection round and watch vibration and temperature.",
                        "No action needed beyond the normal maintenance plan.")
    add_to_log("1 Maintenance", f"{atype}, {country}", f"risk {score:.0%}", verdict)


# ---------------------------------------------------------------- Objective 2
def page_margin():
    st.header("2. Margin forecast")
    st.write("What gross margin will this refinery run earn? Enter the crude, plant conditions and market prices.")
    f = feats_of(2)
    with st.form("form2"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Crude and site**")
            grade, d_grade = category(f, "Crude grade", "crude_grade", "m_grade", default="Bonny Light")
            country, d_country = category(f, "Country", "country", "m_country", default="Nigeria")
            api = st.number_input("Crude API gravity", 10.0, 60.0, 34.8, 0.1)
            sulphur = st.number_input("Crude sulphur (%)", 0.0, 5.0, 0.24, 0.05)
            cost = st.number_input("Crude cost (USD/bbl)", 20.0, 150.0, 74.0, 0.5)
        with c2:
            st.markdown("**Plant conditions**")
            thr = st.number_input("Throughput (barrels)", 1000.0, 600000.0, 152000.0, 1000.0)
            util = st.number_input("Capacity utilisation (%)", 10.0, 120.0, 85.0, 1.0)
            furnace = st.number_input("Furnace outlet temperature (°C)", 200.0, 500.0, 365.0, 1.0)
            colp = st.number_input("Column pressure (bar)", 0.5, 5.0, 1.8, 0.1)
            energy = st.number_input("Energy use (GJ/bbl)", 0.1, 3.0, 0.65, 0.05)
            down = st.number_input("Unplanned downtime (hours)", 0.0, 48.0, 1.0, 0.5)
        with c3:
            st.markdown("**Market and date**")
            crack = st.number_input("Crack spread today (USD/bbl)", -5.0, 40.0, 13.7, 0.1)
            crack7 = st.number_input("Crack spread, 7-day average (USD/bbl)", -5.0, 40.0, 13.65, 0.1)
            prev = st.number_input("Average margin of the last 7 days (USD/bbl)", -10.0, 40.0, 12.0, 0.1)
            day = st.date_input("Run date", TODAY)
        go = st.form_submit_button("Forecast margin")
    if not go:
        return
    values = {**d_grade, **d_country, "crude_api_gravity": api, "crude_sulphur_pct": sulphur, "crude_cost_usd_per_bbl": cost,
              "throughput_bbl": thr, "capacity_utilisation": util / 100, "furnace_outlet_temp_c": furnace, "column_pressure_bar": colp,
              "energy_use_gj_per_bbl": energy, "unplanned_downtime_hours": down, "crack_spread_usd_per_bbl": crack,
              "crack_spread_7d_avg": crack7, "margin_prev_7d_avg": prev, "month": day.month, "quarter": (day.month - 1) // 3 + 1,
              "day_of_week": day.weekday(), "energy_x_sulphur": energy * sulphur,
              "sour_crude": float(sulphur > 1.5), "heavy_crude": float(api < 32), "had_downtime": float(down > 0)}
    model, X = run_model(2, values)
    pred = float(model.predict(X)[0])
    err = float(REGISTRY["objectives"]["2"]["test_results"]["MAE"])
    a, b, c = st.columns(3)
    a.metric("Forecast margin", f"{pred:.2f} USD/bbl", f"{pred - AVERAGE_MARGIN:+.2f} vs average ({AVERAGE_MARGIN})")
    b.metric("Likely range", f"{pred - err:.2f} to {pred + err:.2f}", "typical error ±" + f"{err:.2f}", delta_color="off")
    c.metric("Margin for this run", f"${pred * thr:,.0f}", f"{thr:,.0f} barrels", delta_color="off")
    if pred >= AVERAGE_MARGIN + 1:
        st.success("Above-average margin expected. Protect this run: avoid downtime and keep throughput steady.")
    elif pred <= AVERAGE_MARGIN - 1:
        st.warning("Below-average margin expected. Check the crude choice, crack spread and energy use before committing the run.")
    else:
        st.info("Margin is expected to be close to the average.")
    notes = []
    if sulphur > 1.5: notes.append("sour crude (earned about 3.5 USD/bbl less than sweet crude in the data)")
    if api < 32: notes.append("heavy crude")
    if crack < 12: notes.append("low crack spread (the strongest driver of margin)")
    if down > 0: notes.append("unplanned downtime (costs throughput even if margin per barrel holds)")
    st.markdown("**Things pulling the forecast down:** " + ("; ".join(notes) if notes else "none of the usual ones"))
    st.caption("This is a forecast, not a promise. On the project's test data it was off by about this much on average.")
    add_to_log("2 Margin", f"{grade}, {country}", f"{pred:.2f} USD/bbl", "Above avg" if pred >= AVERAGE_MARGIN + 1 else "Below avg" if pred <= AVERAGE_MARGIN - 1 else "Average")


# ---------------------------------------------------------------- Objective 3
def page_delivery():
    st.header("3. Delivery risk")
    st.write("Will this fuel trip arrive more than 30 minutes late? Enter the trip details at dispatch.")
    f = feats_of(3)
    with st.form("form3"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Route and load**")
            product, d_prod = category(f, "Product", "product", "d_prod", default="AGO")
            site, d_site = category(f, "Origin site type", "site_type", "d_site", default="Depot")
            country, d_country = category(f, "Origin country", "country", "d_country", default="Nigeria")
            region, d_region = category(f, "Destination region", "destination_region", "d_region", default="West Africa")
            volume = st.number_input("Volume (litres)", 1000.0, 60000.0, 30000.0, 1000.0)
        with c2:
            st.markdown("**Conditions**")
            distance = st.number_input("Distance (km)", 1.0, 1500.0, 108.0, 5.0)
            traffic = st.slider("Traffic index (0 = clear, 10 = gridlock)", 0.0, 10.0, 5.0, 0.5)
            weather, d_weather = category(f, "Weather", "weather", "d_weather", default="Clear")
            promised = st.number_input("Promised trip time (hours)", 0.5, 72.0, 4.5, 0.5)
        with c3:
            st.markdown("**Truck, driver, time**")
            truck = st.number_input("Truck age (years)", 0.0, 30.0, 7.0, 1.0)
            driver = st.number_input("Driver experience (years)", 0.0, 45.0, 8.0, 1.0)
            day = st.date_input("Dispatch date", TODAY)
            hour = st.slider("Dispatch hour (0-23)", 0, 23, 10)
        go = st.form_submit_button("Check this trip")
    if not go:
        return
    bad_weather = weather in ("Rain", "Storm", "Harmattan dust")
    values = {**d_prod, **d_site, **d_country, **d_region, **d_weather,
              "volume_litres": volume, "distance_km": distance, "traffic_index": traffic, "promised_hours": promised,
              "promised_speed_kmh": distance / promised, "truck_age_years": truck, "driver_experience_years": driver,
              "dispatch_hour": hour, "dispatch_day": day.weekday(), "dispatch_month": day.month,
              "traffic_x_distance": traffic * distance, "is_weekend": float(day.weekday() >= 5),
              "peak_hour": float(hour in (6, 7, 8, 9, 16, 17, 18, 19)), "long_haul": float(distance > 200),
              "old_truck": float(truck > 10), "novice_driver": float(driver < 3), "bad_weather": float(bad_weather)}
    factors = []
    if traffic >= 9: factors.append("very heavy traffic (such trips are late about 7x as often as light-traffic trips)")
    if distance >= 175: factors.append("long distance (about 3x)")
    if bad_weather: factors.append(f"bad weather: {weather.lower()} (about 2.4x)")
    if truck > 10: factors.append("older truck")
    if driver < 3: factors.append("novice driver")
    if distance / promised > 35: factors.append("tight promise (very high average speed needed)")
    score = risk_score(3, values)
    verdict = show_risk(3, score, factors,
                        f"High chance of a late arrival. Add a time buffer to the promised time, or tell the customer early. "
                        f"(A late trip is assumed to cost about ${COST['late_penalty_usd']} in penalties.)",
                        "Some risk. Keep an eye on traffic and weather, and prepare a revised time if needed.",
                        "Low risk of lateness. Dispatch as planned.")
    add_to_log("3 Delivery", f"{product}, {distance:.0f} km", f"risk {score:.0%}", verdict)


# ---------------------------------------------------------------- Objective 4
def page_quality():
    st.header("4. Off-spec risk")
    st.write("Will this blend batch fail the product specification? Enter the batch details before release.")
    f = feats_of(4)
    with st.form("form4"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Product and feed**")
            product, d_prod = category(f, "Product", "product", "q_prod", exclude=("product_group",), default="AGO")
            grade, d_grade = category(f, "Feed crude grade", "crude_grade", "q_grade", default="Bonny Light")
            site, d_site = category(f, "Site type", "site_type", "q_site", default="Terminal")
            country, d_country = category(f, "Country", "country", "q_country", default="Nigeria")
        with c2:
            st.markdown("**Blend settings**")
            sulphur = st.number_input("Feed crude sulphur (%)", 0.0, 5.0, 0.24, 0.05)
            dose = st.number_input("Additive dose (ppm)", 50.0, 400.0, 200.0, 1.0)
            temp = st.number_input("Blend temperature (°C)", 10.0, 80.0, 38.0, 0.5)
        with c3:
            st.markdown("**Quality indicators**")
            water = st.number_input("Water content (ppm)", 0.0, 1500.0, 40.0, 1.0)
            storage = st.number_input("Storage time (days)", 0.0, 60.0, 7.0, 1.0)
            day = st.date_input("Test date", TODAY)
        go = st.form_submit_button("Check this batch")
    if not go:
        return
    gasoline = product == "PMS"
    values = {**d_prod, **d_grade, **d_site, **d_country,
              "feed_crude_sulphur_pct": sulphur, "additive_dose_ppm": dose, "blend_temp_c": temp, "storage_days": storage,
              "water_content_ppm": water, "dose_deviation_ppm": abs(dose - TYPICAL_DOSE_PPM), "blend_temp_deviation": temp - TYPICAL_BLEND_TEMP_C,
              "test_month": day.month, "test_quarter": (day.month - 1) // 3 + 1,
              "high_water": float(water > 80), "long_storage": float(storage > 14), "sour_feed": float(sulphur > 1.0),
              "product_group_Gasoline": float(gasoline), "product_group_Middle_distillate": float(not gasoline)}
    factors = []
    if water > 80: factors.append("high water content (such batches fail about 4.7x as often)")
    elif water >= 60: factors.append("water content in the top quarter (about 6x the failure rate of the driest batches)")
    if sulphur > 1.0: factors.append("sour feed crude (about 2.7x)")
    if storage > 14: factors.append("long storage (about 1.5x)")
    if abs(dose - TYPICAL_DOSE_PPM) > 40: factors.append("additive dose far from the usual setting")
    score = risk_score(4, values)
    verdict = show_risk(4, score, factors,
                        f"Hold the batch for an extra check and blend correction. Rework of an off-spec batch costs about "
                        f"${COST['rework_usd']:,}; an extra check is assumed to cost about ${COST['quality_check_usd']}.",
                        "Some risk. Re-test water content and sulphur before release.",
                        "Low risk. Release as normal.")
    add_to_log("4 Quality", f"{product}, water {water:.0f} ppm", f"risk {score:.0%}", verdict)


# ---------------------------------------------------------------- Objective 5
def page_inventory():
    st.header("5. Inventory check")
    st.write("Does this tank-day look like a loss, theft or metering problem? Compare the book stock with the measured stock.")
    f = feats_of(5)
    with st.form("form5"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Tank**")
            product, d_prod = category(f, "Product", "product", "i_prod", default="AGO")
            site, d_site = category(f, "Site type", "site_type", "i_site", default="Depot")
            country, d_country = category(f, "Country", "country", "i_country", default="Nigeria")
            day = st.date_input("Record date", TODAY)
        with c2:
            st.markdown("**Volumes (kilolitres)**")
            opening = st.number_input("Opening stock", 1.0, 200000.0, 19000.0, 100.0)
            receipts = st.number_input("Receipts", 0.0, 100000.0, 0.0, 50.0)
            dispatches = st.number_input("Dispatches", 0.0, 100000.0, 500.0, 10.0)
            book = st.number_input("Book closing stock (what the records say)", 1.0, 200000.0, 18500.0, 100.0)
            measured = st.number_input("Measured closing stock (what the gauge says)", 1.0, 200000.0, 18500.0, 100.0)
        with c3:
            st.markdown("**Meter and history**")
            temp = st.number_input("Tank temperature (°C)", 0.0, 70.0, 30.0, 0.5)
            calib = st.number_input("Days since meter calibration", 0.0, 1000.0, 180.0, 10.0)
            mean14 = st.number_input("This tank's usual gap, last 14 days (%)", -5.0, 5.0, 0.0, 0.05)
            std14 = st.number_input("Spread of that gap, last 14 days (%)", 0.0, 5.0, 0.13, 0.01)
        go = st.form_submit_button("Check this tank-day")
    if not go:
        return
    gap_kl = measured - book
    gap_pct = gap_kl / book * 100
    arith = opening + receipts - dispatches - book
    values = {**d_prod, **d_site, **d_country,
              "opening_stock_kl": opening, "receipts_kl": receipts, "dispatches_kl": dispatches, "book_closing_kl": book,
              "measured_closing_kl": measured, "tank_temp_c": temp, "days_since_meter_calibration": calib,
              "gap_kl": gap_kl, "gap_pct": gap_pct, "abs_gap_pct": abs(gap_pct), "measured_to_book_ratio": measured / book,
              "book_arithmetic_gap": arith, "dispatch_rate_pct": dispatches / opening * 100, "received_today": float(receipts > 0),
              "gap_pct_mean_14d": mean14, "gap_pct_std_14d": std14, "gap_zscore": (gap_pct - mean14) / max(std14, 0.05),
              "record_month": day.month, "record_day": day.weekday()}
    model, X = run_model(5, values)
    model_flag = int(model.predict(X)[0]) == -1                  # Isolation Forest: -1 means "unusual"
    unusual = float(-model.decision_function(X)[0])               # higher = more unusual
    rule_flag = abs(gap_pct) > 1
    recording_error = abs(gap_pct) > 50

    a, b, c = st.columns(3)
    a.metric("Gap (measured minus book)", f"{gap_kl:,.0f} kl", f"{gap_pct:+.2f}%", delta_color="off")
    b.metric("Gap rule (over 1%)", "FLAG" if rule_flag else "OK")
    c.metric("Machine learning model", "FLAG" if model_flag else "OK", f"unusualness {unusual:+.2f}", delta_color="off")
    if recording_error:
        st.error("The measured stock is more than 50% away from book. This is almost certainly a **recording or reading error** "
                 "(for example a wrong digit or unit). Re-check the gauge reading and the entry before investigating a loss.")
    elif rule_flag:
        st.error("The gap is over 1% of book stock. Investigate this tank-day: check meter calibration, unrecorded receipts or "
                 "dispatches, leaks and possible theft. The gap rule is the main control; it was the most accurate in the project.")
    elif model_flag:
        st.warning("The gap is under 1%, but the model finds the pattern unusual. Use it as a second opinion: review the tank's "
                   "recent history if staff time allows.")
    else:
        st.success("No sign of a problem. Book and measured stock agree.")
    notes = []
    if abs(arith) > 0.01 * opening: notes.append(f"the book figures do not add up (opening + receipts - dispatches differs from book closing by {arith:,.0f} kl)")
    if abs(gap_pct - mean14) / max(std14, 0.05) > 3: notes.append("this gap is far outside the tank's usual range")
    if calib > 270: notes.append("meter not calibrated for a long time (note: in the project data this did not explain gaps by itself)")
    st.markdown("**Other things noticed:** " + ("; ".join(notes) if notes else "nothing else"))
    add_to_log("5 Inventory", f"{product}, {site}", f"gap {gap_pct:+.2f}%",
               "Recording error" if recording_error else "Investigate" if rule_flag else "Review" if model_flag else "OK")


# ---------------------------------------------------------------- Log
def page_log():
    st.header("Prediction log")
    st.write("Every check you run in this session is listed here. Download it to keep a record.")
    log = st.session_state.get("log", [])
    if not log:
        st.info("Nothing yet. Run a check on one of the pages.")
        return
    table = pd.DataFrame(log)
    st.dataframe(table)
    st.download_button("Download log as CSV", table.to_csv(index=False).encode("utf-8"), "meridian_prediction_log.csv", "text/csv")
    if st.button("Clear log"):
        st.session_state["log"] = []
        st.rerun()


# ----------------------------------------------------------------------------------------------
# 5. MENU
# ----------------------------------------------------------------------------------------------
PAGES = {"Home": page_home, "1  Maintenance risk": page_maintenance, "2  Margin forecast": page_margin,
         "3  Delivery risk": page_delivery, "4  Off-spec risk": page_quality, "5  Inventory check": page_inventory,
         "Prediction log": page_log}
st.sidebar.title("Meridian Downstream")
st.sidebar.caption("Operations and Risk Predictive Support System")
choice = st.sidebar.radio("Go to", list(PAGES))
PAGES[choice]()
st.sidebar.caption("Decision support only. Built from the Meridian ML project (synthetic data).")
```

---

## 21. Key Findings and Insights

**Objective 1: breakdowns**
- Breakdowns are rare (4.6% of asset-months), so this is a ranking tool. Servicing the riskiest 10% catches 38% of breakdowns, 3.8x better than random. PR AUC is only 0.155 with a wide range (0.11 to 0.23).
- **Heat is the strongest warning.** Assets in the hottest quarter (3-month average from 79.8 C) break down 9.8% of the time next month against 1.4% in the coolest quarter: 7.1x.
- A vibration alert means 12.4% breakdown risk against 3.3% without (3.7x). Poor lubricant means 15.2% against 4.5% (3.4x, but a small group of 66 cases).
- Age matters less (1.6x between oldest and youngest quarter).

**Objective 2: margin**
- **Crack spread is the engine.** Alone it correlates 0.83 with margin.
- Crude grade swings margin from 8.1 (Basrah Medium) to 13.4 (WTI), a gap of 5.3 USD/bbl. Sour crude earns 8.90 against 12.40 for sweet, a 3.50 penalty.
- Unplanned downtime shows no effect on margin per barrel (11.51 either way). Its cost sits in lost throughput.
- Ridge beat the tree models. The forecast is within 0.86 USD/bbl, 52% better than last week's margin.

**Objective 3: late deliveries**
- **Traffic dominates.** Trips in the heaviest-traffic quarter are late 26.3% of the time against 3.7% in the lightest: 7.0x.
- Long distance (3.1x), bad weather (2.4x), older trucks (1.5x) and novice drivers (1.3x) matter less.
- Flagging the riskiest 20% catches 62% of late trips.

**Objective 4: off-spec batches**
- **Water is the clearest lever.** Batches in the wettest quarter (from 60.2 ppm) fail 16.4% of the time against 2.7% in the driest: 6.1x. High-water batches fail 4.7x as often.
- Sour feed crude (2.7x) and long storage (1.5x) follow.
- The model is useful but not sharp: PR AUC 0.162 (0.11 to 0.25).

**Objective 5: inventory**
- **A plain rule beat machine learning.** "Gap over 1% of book" scores PR AUC 0.987 against 0.401 for Isolation Forest, and finds 85% to 98% of every anomaly type.
- Meter calibration age does not explain gaps: 3.3% of recently calibrated tank-days have a big gap against 3.0% for the longest-uncalibrated. A calibration schedule alone would not fix this.
- 32 tank-days hold a reading more than 50% away from book (for example 10x too high). Those are recording errors that need entry validation.

**Across the project**
- **Signal decides what works.** Margin is highly predictable. Late trips and loss are predictable enough to act on. Breakdowns and off-spec batches are weaker and should be used as rankings.
- **Simple beat complex twice.** Ridge won margin and a rule won inventory.
- **Delivery is the thinnest business case.** It can lose money if actions cost more than assumed.

<p align="center">
  <img src="images/key_findings.png" alt="Key findings by objective" width="100%">
</p>
<p align="center"><em>Key findings by objective</em></p>

---

## 22. Recommendations

In priority order, by size of value and ease of starting:

1. **Run a monthly risk-ranked maintenance round (Asset Integrity and Maintenance).** Score every asset and service the riskiest 10%. Pilot at one or two sites with a 0.75 cut-off. Meanwhile make vibration-alert and high-temperature assets an automatic inspection trigger, no model needed. Judge it on a full quarter.
2. **Tighten water control in blending and add a pre-release check (Quality and Blending).** Review the water limit and where water enters the blend first, then trial the batch check list (0.58 cut-off) at one site. A check pays for itself once the flagged group is above 3.1% off-spec.
3. **Add traffic-aware promised times (Supply and Distribution).** A traffic buffer is cheap and needs no model. Use the model flag only for actions costing under about $81 per flagged trip, and confirm the real late penalty first.
4. **Use the margin forecast in weekly planning (Refining and Planning).** Compare it with planners' own estimates for four weeks, refit monthly, and use the 3.5 USD/bbl sour-crude gap to judge crude price discounts.
5. **Make the daily "gap over 1%" rule the inventory control (Terminal Operations and Security).** Investigate every flagged tank-day, keep Isolation Forest as a second opinion, and reject any reading more than 50% from book at entry.
6. **Before going live (Data and Analytics):** refit on all history, replace assumed costs with Finance's real ones, and monitor precision and recall monthly.

| When | What |
|---|---|
| Days 0 to 30 | Water-control review, traffic buffers, vibration and temperature triggers, gap rule on, entry validation, refit |
| Days 30 to 90 | Pilot preventive rounds and batch checks at one or two sites, confirm costs with Finance |
| Day 90 onward | Scale what the pilots prove, add margin forecast to weekly planning, start monthly monitoring |

---

## 23. Impact

Annual figures are scaled from each test window to 365 days. They are indicative until Finance confirms the cost assumptions.

| Area | Measure | Now | With the plan | Change |
|---|---|---|---|---|
| Maintenance | Breakdowns per year | 218 | 168 | **-23%** (1,248 downtime hours and $1.5M repair cost avoided, for 422 extra preventive jobs) |
| Delivery | Late trips per year | 314 | 236 | **-25%** (on-time rate 89.2% to 91.8%, for 580 flagged trips to handle) |
| Quality | Off-spec batches per year | 131 | 87 | **-34%** (off-spec rate 6.5% to 4.3%, $514,699 rework avoided, for 399 extra checks) |

**Net value per year under three sets of assumptions**

| Objective | Conservative | Base | Optimistic |
|---|---|---|---|
| 1. Maintenance | $711,371 | $3,801,906 | $6,907,548 |
| 3. Delivery | -$23,117 | $23,624 | $70,415 |
| 4. Quality | $107,784 | $414,989 | $542,857 |
| **Total** | **$796,038** | **$4,240,519** | **$7,520,819** |

Resampling each test period 500 times gives a 90% range of **$2.96M to $5.72M** around the base total. Even the conservative case stays positive.

**What the answer is most sensitive to:** the share of flagged breakdowns actually avoided (a swing of $4.6M) and the downtime cost per hour (a swing of $3.1M).

**Break-even:** maintenance stops paying if fewer than 11% of flagged breakdowns are avoided (assumed 60%). Quality needs 14% (assumed 70%). Delivery needs 20% (assumed 40%), the thinnest margin.

**Margin:** every 0.10 USD/bbl gained on last year's 331 million barrels is worth about $33.1M. The forecast does not create that gain, it shows planners where it is. **Inventory:** the gap rule puts about 332 tank-days a year under investigation (3.3%).

### Charts

<p align="center">
  <img src="images/impact.png" alt="Impact: operational change and value under three scenarios" width="100%">
</p>
<p align="center"><em>Impact: operational change and value under three scenarios</em></p>

---

## 24. Conclusion

I built a full prediction system for a downstream oil and gas business: eight messy tables cleaned and joined in a star schema, five modelling tables, a date-based split, four models per objective plus three anomaly detectors, all saved with joblib and served in a seven-page Streamlit app.

**Four of five objectives met their success bars.** Margin is forecast to within 0.86 USD/bbl (R² 0.85). Late trips are caught at 62% by flagging the riskiest 20%, breakdowns at 38% by servicing the riskiest 10%, and off-spec batches at 48% by checking the riskiest 20%. At the stated assumptions the maintenance, delivery and quality plans return about **$4.2M a year net**, with maintenance as 90% of it. The fifth objective was only partly met, and I kept that in: a one-line "gap over 1%" rule outperformed all three anomaly models.

**Limits to keep in mind**
- The data is synthetic with planted patterns, so effects are cleaner than real operations will be.
- Dollar values use assumed unit costs (downtime per hour, late penalty, inspection cost, fuel price), not company records.
- Each objective was tested on one later period. Breakdown and off-spec results have wide ranges: use them to rank risk, not to decide alone.
- The saved models were trained on the training split only and need a refit on all data before real use.
- Delivery is the weakest business case.

**Next steps:** refit on all history, replace assumed costs with real ones, pilot maintenance rounds and batch checks at one or two sites, add gap-based features to the inventory model, and monitor flags against what actually happened.

---

**Author:** Adewale Oshinukan | Data Scientist | [GitHub](https://github.com/waleoshin75-sketch) | [Live App](https://meridiandownstreamenergyoperationsandriskpredictivesupportsys.streamlit.app/)
