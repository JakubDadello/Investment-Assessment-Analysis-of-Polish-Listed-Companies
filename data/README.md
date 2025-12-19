# Data Folder

This folder contains all the datasets used in the project.

---

## 1. raw_data.csv
- **Description:** Financial data for 370 Polish listed companies from the Warsaw Stock Exchange (GPW)
- **Source:** Official GPW website
- **Date collected:** 19 Dec 2025
- **Columns:**
  - `net_income`: Net profit in PLN
  - `net_cash_flow`: Net cash flow in PLN
  - `roe`: Return on equity
  - `roa`: Return on assets
  - `ebitda`: EBITDA in PLN
  - `sector`: Industry sector
  - `cumulation`: Reporting type (quarterly, semi-annual, or three-quarter reports)

---

## 2. initial_labeling_data.csv
- **Description:** Investment level labels for GPW companies
- **Labels:** low / medium / high
- **Methodology:**
  - For each sector, median values of financial indicators (`roe`, `roa`, `net_income`, `ebitda`) were calculated
  - Companies were labeled based on their position relative to the sector median
- **Notes:**
  - Each company in `raw_data.csv` has exactly one label
  - No missing labels

---

## 3. <third_file>.csv
- **Description:** …
- **Source:** …
- **Columns:** …
- **Notes:** …
