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

## 3. preprocessed_data.csv
- **Description:** Preprocessed dataset ready for machine learning. Combines features from `initial_labeling_data.csv` with preprocessing applied to numerical and categorical variables.  
- **Preprocessing steps:**
  - **Numerical features:** `net_income`, `net_cash_flow`, `roe`, `roa`, `ebitda`, `cumulation`
    - Missing values imputed using **median**
    - Features standardized using **StandardScaler**
  - **Categorical features:** `sector`
    - Missing values imputed using **most frequent value**
    - One-hot encoded using **OneHotEncoder** (ignoring unknown categories)
- **Features:** Contains all input features after preprocessing (numerical scaled + categorical one-hot encoded)  
- **Target:** Includes the investment level label (`low`, `medium`, `high`) from `initial_labeling_data.csv`  
- **Usage:** Ready for modeling in Python or other ML frameworks

