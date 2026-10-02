# Principal Component Analysis of Cross-Country Well-Being: World Happiness Report 2021

| | |
|---|---|
| **Name** | Kobe Illana A. Atanacio |
| **Section** | Section 2 |
| **Course** | APM1220 |
| **Assessment title** | Principal Component Analysis of Cross-Country Well-Being (World Happiness Report 2021) |

## Dataset Source

**World Happiness Report 2021** (Sustainable Development Solutions Network; survey data from the Gallup World Poll), provided as `world-happiness-report-2021.csv`. It contains 149 countries. Country name and regional indicator are used only for labeling and are excluded from the PCA.

## Brief Description of the PCA Analysis

The analysis uses Python (pandas, NumPy, scikit-learn, matplotlib, seaborn) to test whether seven correlated well-being indicators can be summarized by fewer principal components:

- **Variables:** Ladder score, Logged GDP per capita, Social support, Healthy life expectancy, Freedom to make life choices, Generosity, and Perceptions of corruption.
- **Method:** The variables were standardized (mean 0, SD 1), and PCA was performed through the eigen-decomposition of the **correlation matrix**. The result was verified against `sklearn.decomposition.PCA`.
- **Number of components:** The first two components were retained, based on the scree plot (elbow at PC3), the Kaiser criterion (eigenvalues 3.914 and 1.289 exceed 1), and a cumulative variance of 74.3%.
- **PC1 (55.9% of variance):** An overall socioeconomic well-being dimension, with high loadings for Ladder score, GDP, life expectancy and social support, and a negative contribution from perceived corruption.
- **PC2 (18.4% of variance):** A generosity and freedom versus perceived-corruption dimension that is largely independent of income.
- **Conclusion:** Cross-country well-being in this dataset is essentially two-dimensional, so seven variables can be reduced to two components with 74.3% of the variance retained.

All numerical results are reported to three decimal places.

## Files

| File | Description |
|---|---|
| `PCA_World_Happiness_2021.ipynb` | Notebook with code, output, graphs, and answers for Parts A–D |
| `PCA_World_Happiness_2021.html` | HTML export of the executed notebook |
| `world-happiness-report-2021.csv` | Dataset (keep in the same folder to re-run the notebook) |
| `README.md` | This file |
