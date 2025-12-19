# Description of the project
The analysis of a company's financial data, taking into account the characteristics of its sector, is among the key factors considered in investment-related decision-making processes. The analytical project presented in this presentation focuses on collecting selected, critical, and structured financial and non-financial data concerning Polish publicly listed companies. Subsequently, a comprehensive process of data exploration and visualization was carried out.

---

# Purpose of the project
The purpose of this project was to develop an efficient classification model capable of accurately categorizing the investment assessment of companies based on the mentioned determinants.

---

# Project issues
1. How can the investment evaluation of a publicly listed company be conducted based on selected financial indicators and information regarding its sector affiliation?
2. What type of analytical model, based on machine learning algorithms, ensures sufficient effectiveness in Solving this Business problem?

---

# Business Understanding
This project concerns fundamental analysis, understood within the framework of economic sciences and financial market theory as a set of methods used to evaluate the value of a company’s financial assets, particularly equity instruments.
Fundamental analysis is based on the company’s so-called “fundamentals,” closely linked to the concept of “intrinsic value.” Serving cognitive, descriptive, explanatory, and predictive functions, it provides a valuable repertoire of analytical techniques and tools for entities investing capital in financial markets.

Due to the multidimensional and interdisciplinary nature of fundamental analysis, the project was focused on one of its core stages — financial analysis.
Financial analysis is the process of examining a company’s financial condition in order to determine its investment attractiveness and the efficiency of its business operations. It is based on the assessment of selected financial ratios, with data sourced from the financial statements of the companies under analysis.

Financial Analysis Areas: 
1. Balance sheet analysis (assets and liabilities)
2. Profitability analysis (income statement/profit and loss account)
3. Cash flow analysis (cash flow statement)
4. Liquidity analysis (e.g., current ratio)
5. Profitability ratios analysis (e.g., ROE)
6. Debt/leverage analysis (e.g., long-term debt ratio)

---

# Data Understanding
The Data Understanding process was carried out for each type of data. In this project, these included both financial and non-financial data.
The tools used during the Data Understanding phase included the PostgreSQL database management system and the pgAdmin interface. In the next step, the preprocessed data, enriched with investment evaluation labels, were imported into an Excel file. Power BI was then used for their visualization.

Within this project, for each company listed on the Warsaw Stock Exchange (WSE), numerical data were collected illustrating the values of five fundamental financial indicators for the year 2025:
1. Net profit (net income/loss)
2. Net cash flow
3. Return on equity (ROE)
4. Return on assets (ROA)
5. Operating profit plus depreciation (EBITDA)

The selection of the above indicators was driven by the need to cover all areas of financial analysis that are highly relevant. They encompass the profitability, liquidity, and efficiency of the company under analysis.

The primary source of the collected financial data was the individual financial statements of the analyzed companies, as included in their periodic reports.
Depending on formal legal requirements and the companies’ internal reporting and accounting policies, for some companies the analyzed reports covered the first quarter of 2025. For the remaining companies, the examined period encompassed the first half of the year, so the statements were of a cumulative nature.

The non-financial data collected for the purposes of this project are categorical in nature and cover thirteen sectors:
1. agriculture and food industry;
2. chemicals and health products;
3. commercial services;
4. communication technologies and media;
5. consumer goods and e-commerce;
6. energy technology;
7. financial and insurance;
8. health and life sciences;
9. hospitality industry;
10. manufacturing industry;
11. real estate and construction;
12. technology and engineering;
13. transport and motorization.






