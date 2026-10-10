# SA1-Applied-Multivariate-Data-Analysis

**Study title:** Physicochemical Differences, Dimensionality, and Latent Structure of Portuguese Vinho Verde Wines: A MANOVA, Principal Component, and Exploratory Factor Analysis of the UCI Wine Quality Dataset

**Author:** Kobe Illana A. Atanacio — APM1220 Applied Multivariate Data Analysis, Summative Assessment 1

## Research questions
1. **MANOVA** – Do red and white wines differ significantly in their combined physicochemical characteristics? (H0: equal mean vectors of six selected variables.)
2. **PCA** – Can the 11 physicochemical measurements be summarised by fewer principal components that retain a substantial share of the variability?
3. **EFA** – What latent dimensions explain the relationships among the 11 physicochemical measurements?

## Dataset
UCI Wine Quality dataset (Portuguese *Vinho Verde*): 1,599 red + 4,898 white = 6,497 wines, 11 physicochemical variables, a sensory `quality` score (not analysed) and a `color` label added when the files are combined.
- Source: https://archive.ics.uci.edu/dataset/186/wine+quality (DOI 10.24432/C56S3T), licence **CC BY 4.0**
- Attribution: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Data set]. UCI Machine Learning Repository. Original study: Cortez et al. (2009), *Decision Support Systems, 47*(4), 547–553, https://doi.org/10.1016/j.dss.2009.05.016

## Statistical methods
Exploratory analysis (descriptives, IQR/Mahalanobis outliers, correlations) · MANOVA (Pillai/Wilks/Hotelling/Roy, Mardia, Box's M, permutation test, sensitivity analyses, univariate ANOVAs, discriminant function) · PCA on the correlation matrix (eigenvalues, scree, parallel analysis, broken-stick) · EFA (KMO, Bartlett, parallel analysis, MAP, minres extraction, oblimin rotation).

## Summary of major findings
- **MANOVA:** red and white wines differ strongly on six variables (Pillai's V = .763, F(6, 6490) = 3,481.81, p < .001, partial η² = .763); total sulfur dioxide and volatile acidity discriminate most; 97.8% cross-validated classification accuracy. Robust to transformation, outlier/duplicate removal, balancing, and permutation testing.
- **PCA:** three components retain 64.4% of the variance — PC1 sulfur dioxide/preservation vs. volatile acidity & salts (tracks wine colour), PC2 density–alcohol (body/sweetness), PC3 acidity/pH.
- **EFA:** three nearly orthogonal factors (53.1% of variance) — preservation/sulfur dioxide, density–alcohol body, acidity. KMO = .405 (marginal), density is a Heywood case; results are exploratory.

## Software
Python 3.12; pandas, numpy, scipy, statsmodels, scikit-learn, pingouin, factor_analyzer, matplotlib, seaborn, reportlab (see `requirements.txt`). Random seed = 42.

## Reproducing the analysis
```bash
pip install -r requirements.txt
python code/SA1_Wine_Analysis.py   # data import, all analyses -> output/figures, output/tables, output/results.json
python code/make_report.py         # builds report/SA1_Wine_Quality_Report.pdf
```
The script first tries `ucimlrepo.fetch_ucirepo(id=186)`; if the UCI server is unreachable it uses `data/winequality-red.csv` and `data/winequality-white.csv`.

## Structure
```
data/    winequality-red.csv, winequality-white.csv
code/    SA1_Wine_Analysis.py, make_report.py
output/  figures/, tables/, results.json, analysis_log.txt
report/  SA1_Wine_Quality_Report.pdf
```
