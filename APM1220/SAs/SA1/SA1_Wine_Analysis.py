#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APM1220 - Applied Multivariate Data Analysis | Summative Assessment 1
Multivariate Analysis of the Wine Quality Dataset (MANOVA, PCA, EFA)
Author: Kobe Illana A. Atanacio

Run from anywhere:   python code/SA1_Wine_Analysis.py
Outputs:             output/figures/*.png, output/tables/*.csv, output/results.json
Requirements:        pandas numpy scipy statsmodels scikit-learn matplotlib seaborn
                     pingouin factor_analyzer ucimlrepo
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pingouin as pg
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity


# --- compatibility shim: factor_analyzer 0.5.x calls check_array(force_all_finite=...), which newer
# scikit-learn renamed to ensure_all_finite. This keeps the script working on both versions.
import factor_analyzer.factor_analyzer as _fa_mod
_orig_check_array = _fa_mod.check_array
def _check_array_compat(X, *a, **kw):
    try:
        return _orig_check_array(X, *a, **kw)
    except TypeError:
        if "force_all_finite" in kw:
            kw["ensure_all_finite"] = kw.pop("force_all_finite")
        return _orig_check_array(X, *a, **kw)
_fa_mod.check_array = _check_array_compat

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
pd.set_option("display.width", 200, "display.max_columns", 30)
sns.set_theme(style="whitegrid", context="notebook")

# ----------------------------------------------------------------------------
# 0. SETUP
# ----------------------------------------------------------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA, FIG, TAB = (os.path.join(BASE, p) for p in ("data", "output/figures", "output/tables"))
for p_ in (DATA, FIG, TAB):
    os.makedirs(p_, exist_ok=True)

FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
            "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
R = {}                      # results container -> results.json


def save_table(df_, name, index=True):
    df_.round(4).to_csv(os.path.join(TAB, name + ".csv"), index=index)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close()


def header(txt):
    print("\n" + "=" * 90 + f"\n{txt}\n" + "=" * 90)


# ----------------------------------------------------------------------------
# PART A. DATA IMPORT, PREPARATION, EXPLORATORY ANALYSIS
# ----------------------------------------------------------------------------
header("PART A - DATA PREPARATION AND EXPLORATORY ANALYSIS")


def load_wine():
    """Primary route: UCI repository via ucimlrepo (id=186). Fallback: local CSVs in data/."""
    try:
        from ucimlrepo import fetch_ucirepo
        wq = fetch_ucirepo(id=186)
        d = wq.data.original.copy()
        d.columns = [c.strip() for c in d.columns]
        d["color"] = d["color"].str.strip().str.capitalize()
        for col in ("red", "white"):  # keep local copies for reproducibility
            f = os.path.join(DATA, f"winequality-{col}.csv")
            if not os.path.exists(f):
                d[d["color"] == col.capitalize()].drop(columns="color").to_csv(f, sep=";", index=False)
        return d[FEATURES + ["quality", "color"]], "UCI repository via ucimlrepo (id=186)"
    except Exception as e:
        print(f"[info] ucimlrepo unavailable ({type(e).__name__}); using local CSVs in data/")
        red = pd.read_csv(os.path.join(DATA, "winequality-red.csv"), sep=";")
        white = pd.read_csv(os.path.join(DATA, "winequality-white.csv"), sep=";")
        red["color"], white["color"] = "Red", "White"
        return pd.concat([red, white], ignore_index=True), "local winequality-red/white.csv (UCI files)"


df, source = load_wine()
df["color"] = pd.Categorical(df["color"], categories=["Red", "White"])
print("Data source:", source)
n_red, n_white = int((df["color"] == "Red").sum()), int((df["color"] == "White").sum())
R["A"] = dict(source=source, n=len(df), n_vars=df.shape[1], n_red=n_red, n_white=n_white)
print(df.shape, "| red:", n_red, "| white:", n_white)
print(df.dtypes)

# missing values & duplicates
R["A"]["missing_total"] = int(df.isna().sum().sum())
dup_mask = df.duplicated(keep="first")
R["A"]["duplicates_total"] = int(dup_mask.sum())
R["A"]["duplicates_red"] = int(dup_mask[df.color == "Red"].sum())
R["A"]["duplicates_white"] = int(dup_mask[df.color == "White"].sum())
R["A"]["n_unique_rows"] = int((~dup_mask).sum())
print("Missing:", R["A"]["missing_total"], "| duplicates (beyond first):", R["A"]["duplicates_total"],
      f"(red {R['A']['duplicates_red']}, white {R['A']['duplicates_white']})")
df_nodup = df[~dup_mask].reset_index(drop=True)


def describe(d):
    t = d[FEATURES].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    t["skew"] = d[FEATURES].skew()
    t["excess_kurt"] = d[FEATURES].kurt()
    t["CV_%"] = 100 * t["std"] / t["mean"]
    return t


desc_all = describe(df)
save_table(desc_all, "A1_descriptives_overall")
save_table(describe(df[df.color == "Red"]), "A2_descriptives_red")
save_table(describe(df[df.color == "White"]), "A3_descriptives_white")
print(desc_all.round(3))
save_table(df.groupby("color")["quality"].describe(), "A4_quality_by_color")

# univariate outliers (Tukey 1.5*IQR; |z| > 3)
out_rows = []
for v in FEATURES:
    q1, q3 = df[v].quantile([.25, .75]); iqr = q3 - q1
    n_iqr = int(((df[v] < q1 - 1.5 * iqr) | (df[v] > q3 + 1.5 * iqr)).sum())
    n_ext = int(((df[v] < q1 - 3 * iqr) | (df[v] > q3 + 3 * iqr)).sum())
    z = (df[v] - df[v].mean()) / df[v].std()
    out_rows.append([v, n_iqr, 100 * n_iqr / len(df), n_ext, int((z.abs() > 3).sum()), float(z.abs().max())])
uout = pd.DataFrame(out_rows, columns=["variable", "n_IQR_outliers", "pct_IQR", "n_extreme_3IQR", "n_abs_z_gt3", "max_abs_z"]).set_index("variable")
save_table(uout, "A5_univariate_outliers")
print(uout.round(2))


def mahal(X_):
    Xc_ = X_ - X_.mean(0)
    S_inv = np.linalg.inv(np.cov(X_, rowvar=False))
    return np.einsum("ij,jk,ik->i", Xc_, S_inv, Xc_)


d2_all = pd.Series(index=df.index, dtype=float)
for c in ("Red", "White"):
    m = df.color == c
    d2_all[m] = mahal(df.loc[m, FEATURES].values)
cut11 = stats.chi2.ppf(0.999, len(FEATURES))
R["A"]["mv_outliers_11"] = int((d2_all > cut11).sum())
R["A"]["mv_outliers_11_cut"] = float(cut11)
print(f"Multivariate outliers (11 vars, within-colour D2 > {cut11:.2f}):", R["A"]["mv_outliers_11"])

corr = df[FEATURES].corr()
save_table(corr, "A6_correlation_pearson")
save_table(df[FEATURES].corr(method="spearman"), "A7_correlation_spearman")
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().rename("r").reset_index()
pairs.columns = ["var1", "var2", "r"]
pairs = pairs.reindex(pairs.r.abs().sort_values(ascending=False).index)
save_table(pairs, "A8_correlation_pairs_sorted", index=False)
print(pairs.head(10).round(3))
R["A"]["n_pairs_abs_r_ge_30"] = int((pairs.r.abs() >= .30).sum())
R["A"]["n_pairs_abs_r_ge_60"] = int((pairs.r.abs() >= .60).sum())
R["A"]["top_pairs"] = pairs.head(8).values.tolist()


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


rows = []
for v in FEATURES:
    a, b = df.loc[df.color == "White", v], df.loc[df.color == "Red", v]
    t, p_ = stats.ttest_ind(a, b, equal_var=False)
    rows.append([v, a.mean(), b.mean(), cohen_d(a, b), t, p_])
cdiff = pd.DataFrame(rows, columns=["variable", "mean_white", "mean_red", "cohen_d", "welch_t", "p"]).set_index("variable")
save_table(cdiff, "A9_color_differences_all_variables")
print(cdiff.round(3))

# ---- figures A ----
pal = {"Red": "#a4161a", "White": "#d9b44a"}
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, v in zip(axes.ravel(), FEATURES):
    for c in ("Red", "White"):
        sns.kdeplot(df.loc[df.color == c, v], ax=ax, color=pal[c], fill=True, alpha=.35, label=c, common_norm=False)
    ax.set_title(v); ax.set_xlabel("")
axes.ravel()[-1].axis("off")
axes.ravel()[0].legend(title="Colour")
fig.suptitle("Density plots of the 11 physicochemical variables by wine colour", y=1.01)
save_fig("fig01_density_by_color")

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, v in zip(axes.ravel(), FEATURES):
    sns.boxplot(data=df, x="color", y=v, ax=ax, palette=pal, fliersize=1.5, linewidth=1)
    ax.set_xlabel(""); ax.set_title(v)
axes.ravel()[-1].axis("off")
fig.suptitle("Boxplots of the 11 physicochemical variables by wine colour", y=1.01)
save_fig("fig02_boxplots_by_color")

plt.figure(figsize=(9, 7.5))
mask = np.triu(np.ones_like(corr, dtype=bool), 1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, annot_kws={"size": 8},
            linewidths=.5, cbar_kws={"label": "Pearson r"})
plt.title("Correlation heatmap of the 11 physicochemical variables (n = 6,497)")
save_fig("fig03_correlation_heatmap")

# scatterplot matrix of the six variables later used as MANOVA dependent variables
_dv_plot = ["fixed acidity", "volatile acidity", "residual sugar", "chlorides", "total sulfur dioxide", "sulphates"]
g_ = sns.pairplot(df[_dv_plot + ["color"]], hue="color", palette=pal, corner=True, diag_kind="kde", height=1.7,
                  plot_kws=dict(s=4, alpha=.35, linewidth=0))
g_.figure.suptitle("Scatterplot matrix of six physicochemical variables by wine colour", y=1.01)
plt.savefig(os.path.join(FIG, "fig03b_scatterplot_matrix.png"), dpi=200, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------
# PART B. MANOVA
# ----------------------------------------------------------------------------
header("PART B - MANOVA")
DV = ["fixed acidity", "volatile acidity", "residual sugar", "chlorides", "total sulfur dioxide", "sulphates"]
R["B"] = dict(DV=DV)

vif = pd.Series([variance_inflation_factor(df[FEATURES].values, i) for i in range(len(FEATURES))], index=FEATURES)
cand = pd.DataFrame({"cohen_d": cdiff["cohen_d"], "VIF_11_vars": vif})
cand["selected"] = cand.index.isin(DV)
save_table(cand, "B0_DV_selection_evidence")
print(cand.round(3))
dv_corr = df[DV].corr()
save_table(dv_corr, "B0b_DV_correlations_pooled")
resid = df[DV] - df.groupby("color")[DV].transform("mean")
dv_corr_w = resid.corr()
save_table(dv_corr_w, "B0c_DV_correlations_within_group")
offdiag = ~np.eye(len(DV), dtype=bool)
R["B"]["max_abs_r_within"] = float(dv_corr_w.where(offdiag).abs().max().max())
R["B"]["max_abs_r_pooled"] = float(dv_corr.where(offdiag).abs().max().max())
R["B"]["vif_DV_only"] = {v: float(variance_inflation_factor(df[DV].values, i)) for i, v in enumerate(DV)}
R["B"]["det_within_corr"] = float(np.linalg.det(dv_corr_w.values))
print("Within-group DV correlations\n", dv_corr_w.round(3))

grp = {c: df.loc[df.color == c, DV].values for c in ("Red", "White")}


def mardia(X_):
    n_, p_ = X_.shape
    Xc_ = X_ - X_.mean(0)
    S = np.cov(X_, rowvar=False, bias=True)
    D = Xc_ @ np.linalg.inv(S) @ Xc_.T
    b1 = (D ** 3).sum() / n_ ** 2
    b2 = (np.diag(D) ** 2).sum() / n_
    df_s = p_ * (p_ + 1) * (p_ + 2) / 6
    chi_s = n_ * b1 / 6
    z_k = (b2 - p_ * (p_ + 2)) / np.sqrt(8 * p_ * (p_ + 2) / n_)
    return dict(b1=float(b1), chi2_skew=float(chi_s), df_skew=float(df_s), p_skew=float(stats.chi2.sf(chi_s, df_s)),
                b2=float(b2), z_kurt=float(z_k), p_kurt=float(2 * stats.norm.sf(abs(z_k))))


R["B"]["mardia"] = {c: mardia(grp[c]) for c in grp}
print("Mardia:", json.dumps(R["B"]["mardia"], indent=1))
sk = pd.DataFrame({c: df.loc[df.color == c, DV].skew() for c in ("Red", "White")})
ku = pd.DataFrame({c: df.loc[df.color == c, DV].kurt() for c in ("Red", "White")})
uni_norm = pd.concat([sk.add_prefix("skew_"), ku.add_prefix("kurt_")], axis=1)
save_table(uni_norm, "B1_DV_skew_kurtosis_by_group")
print(uni_norm.round(2))

cut_dv = stats.chi2.ppf(0.999, len(DV))
d2_dv = pd.Series(index=df.index, dtype=float)
for c in ("Red", "White"):
    m = df.color == c
    d2_dv[m] = mahal(df.loc[m, DV].values)
out_dv = d2_dv > cut_dv
R["B"]["mv_outliers"] = dict(cut=float(cut_dv), total=int(out_dv.sum()),
                             red=int(out_dv[df.color == "Red"].sum()), white=int(out_dv[df.color == "White"].sum()))
print("MV outliers (DV set):", R["B"]["mv_outliers"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, c in zip(axes, ("Red", "White")):
    d2 = np.sort(d2_dv[df.color == c].values)
    th = stats.chi2.ppf((np.arange(1, len(d2) + 1) - .5) / len(d2), len(DV))
    ax.scatter(th, d2, s=4, color=pal[c], edgecolor="k", linewidth=.1)
    lim = max(th.max(), d2.max()); ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.axhline(cut_dv, color="grey", ls=":", lw=1)
    ax.set_title(f"{c} wines (n = {len(d2):,})")
    ax.set_xlabel(r"Theoretical $\chi^2_6$ quantiles"); ax.set_ylabel(r"Observed Mahalanobis $D^2$")
fig.suptitle("Chi-square Q-Q plots of Mahalanobis distances (six MANOVA dependent variables)", y=1.02)
save_fig("fig04_mahalanobis_qq")

def box_m(groups):
    """Box's M test (log-determinant form, numerically stable for large n; chi-square approximation)."""
    g_ = len(groups); p_ = groups[0].shape[1]
    ns = np.array([len(x) for x in groups]); N_ = ns.sum()
    covs = [np.cov(x, rowvar=False) for x in groups]
    Sp_ = sum((n_i - 1) * S_i for n_i, S_i in zip(ns, covs)) / (N_ - g_)
    M = (N_ - g_) * np.linalg.slogdet(Sp_)[1] - sum((n_i - 1) * np.linalg.slogdet(S_i)[1] for n_i, S_i in zip(ns, covs))
    c = (np.sum(1 / (ns - 1)) - 1 / (N_ - g_)) * (2 * p_ ** 2 + 3 * p_ - 1) / (6 * (p_ + 1) * (g_ - 1))
    chi2_ = M * (1 - c); df_ = p_ * (p_ + 1) * (g_ - 1) / 2
    return dict(M=float(M), chi2=float(chi2_), df=int(df_), p=float(stats.chi2.sf(chi2_, df_)))


R["B"]["boxm"] = box_m([grp["Red"], grp["White"]])
cov_det = {c: float(np.linalg.det(np.cov(grp[c], rowvar=False))) for c in grp}
R["B"]["cov_det"] = cov_det
R["B"]["cov_det_ratio"] = cov_det["White"] / cov_det["Red"]
lev = []
for v in DV:
    W, p_ = stats.levene(df.loc[df.color == "Red", v], df.loc[df.color == "White", v], center="median")
    lev.append([v, df.loc[df.color == "Red", v].var(), df.loc[df.color == "White", v].var(), W, p_])
levdf = pd.DataFrame(lev, columns=["variable", "var_red", "var_white", "Brown_Forsythe_W", "p"]).set_index("variable")
save_table(levdf, "B2_levene_by_DV")
print("Box's M:", R["B"]["boxm"], "| det ratio W/R:", round(R["B"]["cov_det_ratio"], 4))
print(levdf.round(4))

# --- the MANOVA ---
mod = MANOVA.from_formula(" + ".join(f"Q('{v}')" for v in DV) + " ~ C(color)", data=df)
mv = mod.mv_test()
tab = mv.results["C(color)"]["stat"]
print(tab)
N = len(df); p = len(DV)
manova_rows = []
for stat_name in ["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]:
    r_ = tab.loc[stat_name]
    manova_rows.append([stat_name, r_["Value"], r_["F Value"], r_["Num DF"], r_["Den DF"], r_["Pr > F"]])
manova_tab = pd.DataFrame(manova_rows, columns=["test", "value", "F", "df1", "df2", "p"]).set_index("test")
save_table(manova_tab, "B3_manova_tests")
print(manova_tab)
wilks = float(manova_tab.loc["Wilks' lambda", "value"])
pillai = float(manova_tab.loc["Pillai's trace", "value"])
Wm = np.cov(grp["Red"], rowvar=False) * (len(grp["Red"]) - 1) + np.cov(grp["White"], rowvar=False) * (len(grp["White"]) - 1)
Sp = Wm / (N - 2)
dvec = grp["White"].mean(0) - grp["Red"].mean(0)
D2 = float(dvec @ np.linalg.inv(Sp) @ dvec)
n1, n2 = len(grp["Red"]), len(grp["White"])
T2 = n1 * n2 / N * D2
R["B"]["manova"] = {k: manova_tab.loc[k].to_dict() for k in manova_tab.index}
R["B"]["effect"] = dict(partial_eta2=1 - wilks, pillai=pillai, mahalanobis_D2=D2, mahalanobis_D=float(np.sqrt(D2)),
                        hotelling_T2=float(T2), N=N, p=p)
print("Partial eta2 =", 1 - wilks, "| Mahalanobis D =", np.sqrt(D2), "| T2 =", T2)


def pillai_two_group(X_, g):
    """Pillai's V for two groups: V = (n1 n2 / N) d' T^-1 d  (T = total SSCP)."""
    X_ = np.asarray(X_, float); g = np.asarray(g)
    n_, p_ = X_.shape
    Xc_ = X_ - X_.mean(0); Tm = Xc_.T @ Xc_
    a, b = X_[g == 0], X_[g == 1]
    d = b.mean(0) - a.mean(0)
    V = len(a) * len(b) / n_ * d @ np.linalg.solve(Tm, d)
    F = V / (1 - V) * (n_ - p_ - 1) / p_
    return V, F, stats.f.sf(F, p_, n_ - p_ - 1)


def run_sens(label, d, dvs, transform=None):
    X_ = d[dvs].values
    if transform: X_ = transform(X_)
    g = (d.color == "White").astype(int).values
    V, F, pv = pillai_two_group(X_, g)
    return [label, len(d), V, F, pv]


sens = [run_sens("Primary analysis", df, DV),
        run_sens("log10-transformed DVs", df, DV, np.log10),
        run_sens("Multivariate outliers removed", df[~out_dv], DV),
        run_sens("Duplicate rows removed", df_nodup, DV)]
bal = pd.concat([df[df.color == "Red"], df[df.color == "White"].sample(n=n_red, random_state=SEED)])
sens.append(run_sens("Balanced subsample (1,599 vs 1,599)", bal, DV))
sens.append(run_sens("Eight-DV set (adds alcohol and density)", df, DV + ["alcohol", "density"]))
sens_df = pd.DataFrame(sens, columns=["analysis", "N", "Pillai_V", "F", "p"]).set_index("analysis")
save_table(sens_df, "B4_manova_sensitivity")
print(sens_df)
R["B"]["boxm_balanced"] = box_m([bal.loc[bal.color == "Red", DV].values, bal.loc[bal.color == "White", DV].values])

Xp = df[DV].values; gp = (df.color == "White").astype(int).values
Xc_p = Xp - Xp.mean(0); Tm_inv = np.linalg.inv(Xc_p.T @ Xc_p)
obs_V = pillai_two_group(Xp, gp)[0]
perm = np.empty(10000)
for i in range(10000):
    gs = rng.permutation(gp)
    d_ = Xp[gs == 1].mean(0) - Xp[gs == 0].mean(0)
    perm[i] = (gs == 0).sum() * (gs == 1).sum() / len(gs) * d_ @ Tm_inv @ d_
R["B"]["perm"] = dict(obs_V=float(obs_V), perm_max=float(perm.max()), perm_mean=float(perm.mean()),
                      p=float((1 + (perm >= obs_V).sum()) / (10000 + 1)), n_perm=10000)
print("Permutation:", R["B"]["perm"])
R["B"]["sens"] = sens_df.reset_index().values.tolist()

alpha_bonf = 0.05 / len(DV)
rows = []
for v in DV:
    a, b = df.loc[df.color == "Red", v], df.loc[df.color == "White", v]
    F, pv = stats.f_oneway(a, b)
    grand = df[v].mean()
    ssb = len(a) * (a.mean() - grand) ** 2 + len(b) * (b.mean() - grand) ** 2
    sst = ((df[v] - grand) ** 2).sum()
    wf = pg.welch_anova(df, dv=v, between="color")
    rows.append([v, a.mean(), a.std(), b.mean(), b.std(), F, pv, ssb / sst, cohen_d(b, a),
                 float(wf["F"].iloc[0]), float(wf["p-unc" if "p-unc" in wf.columns else "p_unc"].iloc[0]), bool(pv < alpha_bonf)])
anova = pd.DataFrame(rows, columns=["variable", "M_red", "SD_red", "M_white", "SD_white", "F_1_6495", "p", "eta2",
                                    "cohen_d", "Welch_F", "Welch_p", "sig_Bonferroni"]).set_index("variable")
save_table(anova, "B5_univariate_anovas")
print(anova.round(4))
R["B"]["bonf_alpha"] = alpha_bonf

c_raw = np.linalg.solve(Sp, dvec)
c_std = c_raw * np.sqrt(np.diag(Sp))
struct = (Sp @ c_raw) / np.sqrt(np.diag(Sp) * (c_raw @ Sp @ c_raw))
disc = pd.DataFrame({"std_coef": c_std, "structure_r": struct}, index=DV)
disc["rank_abs_structure"] = disc["structure_r"].abs().rank(ascending=False).astype(int)
save_table(disc, "B6_discriminant_function")
print(disc.round(3))
cv = cross_val_score(LinearDiscriminantAnalysis(), df[DV].values, (df.color == "White").astype(int).values,
                     cv=StratifiedKFold(10, shuffle=True, random_state=SEED))
R["B"]["lda_cv_acc"] = float(cv.mean())
R["B"]["majority_baseline"] = float(max(n_red, n_white) / len(df))
print("LDA 10-fold CV accuracy:", cv.mean(), "baseline:", R["B"]["majority_baseline"])

fig, ax = plt.subplots(figsize=(8, 4.6))
order = anova["cohen_d"].sort_values().index
ax.barh(order, anova.loc[order, "cohen_d"], color=["#a4161a" if x < 0 else "#d9b44a" for x in anova.loc[order, "cohen_d"]], edgecolor="k")
ax.axvline(0, color="k", lw=1)
for s in (-.8, .8):
    ax.axvline(s, color="grey", ls=":", lw=1)
ax.set_xlabel("Cohen's d (White minus Red); dotted lines mark |d| = 0.8")
ax.set_title("Standardised mean differences between white and red wines\n(six MANOVA dependent variables)")
save_fig("fig05_manova_cohens_d")

# ----------------------------------------------------------------------------
# PART C. PCA (correlation matrix of the 11 standardised variables)
# ----------------------------------------------------------------------------
header("PART C - PRINCIPAL COMPONENT ANALYSIS")
X = df[FEATURES].values
n, p = X.shape
Z = (X - X.mean(0)) / X.std(0, ddof=1)
Rm = np.corrcoef(X, rowvar=False)
kmo_vec, kmo_total = calculate_kmo(df[FEATURES])
chi_b, p_b = calculate_bartlett_sphericity(df[FEATURES])
R["C"] = dict(det_R=float(np.linalg.det(Rm)), kmo=float(kmo_total), bartlett_chi2=float(chi_b),
              bartlett_df=int(p * (p - 1) / 2), bartlett_p=float(p_b))
kmo_tab = pd.DataFrame({"MSA": kmo_vec}, index=FEATURES)
save_table(kmo_tab, "C0_KMO_per_variable")
print("det(R) =", R["C"]["det_R"], "| KMO =", kmo_total, "| Bartlett chi2 =", chi_b, "p =", p_b)
print(kmo_tab.round(3).T)

evals, evecs = np.linalg.eigh(Rm)
idx = np.argsort(evals)[::-1]
evals, evecs = evals[idx], evecs[:, idx]
load_pca = evecs * np.sqrt(evals)
for j in range(p):
    if load_pca[np.abs(load_pca[:, j]).argmax(), j] < 0:
        load_pca[:, j] *= -1; evecs[:, j] *= -1
prop = evals / evals.sum(); cum = prop.cumsum()


def parallel_pca(n_, p_, iters=1000, q=95):
    sims_ = np.empty((iters, p_))
    for i in range(iters):
        sims_[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n_, p_)), rowvar=False)))[::-1]
    return sims_.mean(0), np.percentile(sims_, q, axis=0)


pa_mean, pa_95 = parallel_pca(n, p)
broken = np.array([sum(1 / k for k in range(i, p + 1)) / p for i in range(1, p + 1)])
eig_tab = pd.DataFrame({"eigenvalue": evals, "proportion": prop, "cumulative": cum, "PA_mean": pa_mean,
                        "PA_95th": pa_95, "broken_stick_prop": broken,
                        "kaiser_gt1": evals > 1, "PA_retain": evals > pa_95},
                       index=[f"PC{i}" for i in range(1, p + 1)])
save_table(eig_tab, "C1_pca_eigenvalues")
print(eig_tab.round(3))
R["C"].update(eig=evals.tolist(), prop=prop.tolist(), cum=cum.tolist(), pa95=pa_95.tolist(),
              n_kaiser=int((evals > 1).sum()), n_pa=int((evals > pa_95).sum()), n_broken_stick=int((prop > broken).sum()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ks = np.arange(1, p + 1)
axes[0].plot(ks, evals, "o-", color="#7b2d26", label="Observed eigenvalues")
axes[0].plot(ks, pa_95, "s--", color="grey", label="Parallel analysis (95th percentile)")
axes[0].axhline(1, color="k", ls=":", lw=1, label="Kaiser criterion (λ = 1)")
axes[0].set_xlabel("Component"); axes[0].set_ylabel("Eigenvalue"); axes[0].set_xticks(ks); axes[0].legend()
axes[0].set_title("(a) Scree plot with parallel analysis")
axes[1].bar(ks, prop * 100, color="#c9a227", edgecolor="k", label="Individual")
axes[1].plot(ks, cum * 100, "o-", color="#7b2d26", label="Cumulative")
axes[1].axhline(80, color="grey", ls=":", lw=1)
axes[1].set_xlabel("Component"); axes[1].set_ylabel("% variance explained"); axes[1].set_xticks(ks); axes[1].legend()
axes[1].set_title("(b) Variance explained")
fig.suptitle("PCA component-retention evidence", y=1.02)
save_fig("fig06_pca_scree_variance")

K_PCA = int(R["C"]["n_pa"])
R["C"]["K"] = K_PCA
nshow = max(K_PCA, 4)
ldg = pd.DataFrame(load_pca[:, :nshow], index=FEATURES, columns=[f"PC{i}" for i in range(1, nshow + 1)])
ldg["communality_K"] = (load_pca[:, :K_PCA] ** 2).sum(1)
save_table(ldg, "C2_pca_loadings")
print(ldg.round(3))
scores = Z @ evecs
for j in range(4):
    df[f"PC{j+1}"] = scores[:, j]
sc_tab = df.groupby("color")[[f"PC{j+1}" for j in range(4)]].agg(["mean", "std"])
save_table(sc_tab, "C3_pc_scores_by_color")
print(sc_tab.round(3))
R["C"]["scores_d"] = {f"PC{j+1}": cohen_d(df.loc[df.color == "White", f"PC{j+1}"], df.loc[df.color == "Red", f"PC{j+1}"]) for j in range(4)}

plt.figure(figsize=(6.5, 6))
sns.heatmap(ldg.iloc[:, :K_PCA], annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, linewidths=.5, cbar_kws={"label": "Loading"})
plt.title(f"PCA loadings (first {K_PCA} components)")
save_fig("fig07_pca_loadings_heatmap")


def loading_and_biplot(a, b, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    ax = axes[0]
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="grey", ls=":"))
    for i, v in enumerate(FEATURES):
        ax.arrow(0, 0, load_pca[i, a], load_pca[i, b], head_width=.025, color="#7b2d26", length_includes_head=True)
        ax.text(load_pca[i, a] * 1.1, load_pca[i, b] * 1.1, v, fontsize=8, ha="center")
    ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
    ax.set_xlabel(f"PC{a+1} ({prop[a]*100:.1f}%)"); ax.set_ylabel(f"PC{b+1} ({prop[b]*100:.1f}%)")
    ax.set_title("(a) Loading plot")
    ax = axes[1]
    for c in ("Red", "White"):
        m = (df.color == c).values
        ax.scatter(scores[m, a] / evals[a] ** .5, scores[m, b] / evals[b] ** .5, s=5, alpha=.35, color=pal[c], label=c)
    sc = 3.0
    for i, v in enumerate(FEATURES):
        ax.arrow(0, 0, load_pca[i, a] * sc, load_pca[i, b] * sc, head_width=.07, color="k", length_includes_head=True, lw=.8)
        ax.text(load_pca[i, a] * sc * 1.1, load_pca[i, b] * sc * 1.1, v, fontsize=8, ha="center")
    ax.axhline(0, color="k", lw=.4); ax.axvline(0, color="k", lw=.4)
    ax.set_xlabel(f"PC{a+1} (standardised score)"); ax.set_ylabel(f"PC{b+1} (standardised score)")
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.legend(markerscale=3, title="Colour"); ax.set_title("(b) Biplot (scores coloured by wine colour)")
    fig.suptitle(title, y=1.0)
    save_fig(name)


loading_and_biplot(0, 1, "fig08_pca_loading_biplot_PC1_PC2", "PCA loading plot and biplot: PC1 vs PC2")
loading_and_biplot(0, 2, "fig09_pca_loading_biplot_PC1_PC3", "PCA loading plot and biplot: PC1 vs PC3")


def eigs(d):
    e = np.sort(np.linalg.eigvalsh(np.corrcoef(d[FEATURES].values, rowvar=False)))[::-1]
    return e / e.sum()


sens_pca = pd.DataFrame({"All (n=6,497)": eigs(df), "No duplicates": eigs(df_nodup),
                         "Red only": eigs(df[df.color == "Red"]), "White only": eigs(df[df.color == "White"])}).iloc[:6]
sens_pca.index = [f"PC{i}" for i in range(1, 7)]
save_table(sens_pca, "C4_pca_sensitivity_prop_variance")
print(sens_pca.round(3))
R["C"]["sens_prop"] = sens_pca.round(4).to_dict()

# ----------------------------------------------------------------------------
# PART D. EXPLORATORY FACTOR ANALYSIS
# ----------------------------------------------------------------------------
header("PART D - EXPLORATORY FACTOR ANALYSIS")
R["D"] = dict(kmo=float(kmo_total), bartlett_chi2=float(chi_b), bartlett_p=float(p_b), bartlett_df=int(p * (p - 1) / 2))


def reduced_eigs(Rmat):
    Rr = Rmat.copy(); np.fill_diagonal(Rr, 1 - 1 / np.diag(np.linalg.inv(Rmat)))
    return np.sort(np.linalg.eigvalsh(Rr))[::-1]


obs_fa = reduced_eigs(Rm)
sims = np.empty((1000, p))
for i in range(1000):
    sims[i] = reduced_eigs(np.corrcoef(rng.standard_normal((n, p)), rowvar=False))
fa_pa95, fa_pamean = np.percentile(sims, 95, axis=0), sims.mean(0)
R["D"]["n_pa"] = int((obs_fa > fa_pa95).sum()); R["D"]["n_kaiser"] = int((evals > 1).sum())
fa_eig_tab = pd.DataFrame({"reduced_corr_eigenvalue": obs_fa, "PA_95th": fa_pa95, "PA_mean": fa_pamean,
                           "PCA_eigenvalue": evals}, index=[f"F{i}" for i in range(1, p + 1)])
save_table(fa_eig_tab, "D1_efa_eigenvalues_parallel")
print(fa_eig_tab.round(3))
R["D"]["obs_fa"] = obs_fa.tolist(); R["D"]["fa_pa95"] = fa_pa95.tolist()

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(ks, evals, "o-", color="#7b2d26", label="PCA eigenvalues (unit diagonal)")
ax.plot(ks, obs_fa, "^-", color="#2b6f8e", label="Common-factor eigenvalues (SMC diagonal)")
ax.plot(ks, fa_pa95, "s--", color="grey", label="Parallel analysis (95th pct., factor)")
ax.axhline(0, color="k", lw=.6); ax.axhline(1, color="k", ls=":", lw=1)
ax.set_xticks(ks); ax.set_xlabel("Factor number"); ax.set_ylabel("Eigenvalue"); ax.legend()
ax.set_title("Scree plot and parallel analysis for exploratory factor analysis")
save_fig("fig10_efa_scree_parallel")


def fit_efa(k, rotation="oblimin", data=None, method="minres"):
    data = df[FEATURES] if data is None else data
    fa_ = FactorAnalyzer(n_factors=k, method=method, rotation=None if k == 1 else rotation)
    fa_.fit(data)
    L_ = fa_.loadings_.copy()
    phi_ = fa_.phi_.copy() if (rotation in ("oblimin", "promax") and k > 1 and getattr(fa_, "phi_", None) is not None) else np.eye(k)
    for j in range(k):
        if L_[np.abs(L_[:, j]).argmax(), j] < 0:
            L_[:, j] *= -1
            phi_[j, :] *= -1; phi_[:, j] *= -1
    return fa_, L_, phi_


def rmsr(L_, phi_):
    res = Rm - L_ @ phi_ @ L_.T
    off = res[np.triu_indices(p, 1)]
    return float(np.sqrt((off ** 2).mean())), float((np.abs(off) > .05).mean() * 100)


# Velicer's minimum average partial (MAP) test
mp = []
for m_ in range(0, 8):
    if m_ == 0:
        Rs = Rm
    else:
        A = evecs[:, :m_] * np.sqrt(evals[:m_]); C = Rm - A @ A.T; dd = np.sqrt(np.diag(C)); Rs = C / np.outer(dd, dd)
    mp.append(float((Rs[np.triu_indices(p, 1)] ** 2).mean()))
R["D"]["map"] = mp; R["D"]["map_min_at"] = int(np.argmin(mp))
print("MAP (m = 0..7):", np.round(mp, 5), "-> minimum at", R["D"]["map_min_at"])

# Maximum-likelihood fit indices for k = 1..5 (used as supporting evidence only; data are non-normal)
chi0 = -(n - 1 - (2 * p + 5) / 6) * np.linalg.slogdet(Rm)[1]; df0 = p * (p - 1) / 2
cmp_rows = []
for k in range(1, 6):
    # rotation-invariant fit of the minres solution
    L_u = FactorAnalyzer(n_factors=k, method="minres", rotation=None).fit(df[FEATURES]).loadings_
    r_, pct_ = rmsr(L_u, np.eye(k))
    comm_u = (L_u ** 2).sum(1)
    fa_ml = FactorAnalyzer(n_factors=k, method="ml", rotation=None).fit(df[FEATURES])
    Lm = fa_ml.loadings_; psi = fa_ml.get_uniquenesses()
    Sig = Lm @ Lm.T + np.diag(psi)
    Fml = np.linalg.slogdet(Sig)[1] - np.linalg.slogdet(Rm)[1] + np.trace(np.linalg.solve(Sig, Rm)) - p
    chi2_k = (n - 1 - (2 * p + 5) / 6 - 2 * k / 3) * Fml
    df_k = ((p - k) ** 2 - (p + k)) / 2
    rmsea = float(np.sqrt(max(chi2_k - df_k, 0) / (df_k * (n - 1))))
    tli = float(((chi0 / df0) - (chi2_k / df_k)) / ((chi0 / df0) - 1))
    bic = float(chi2_k - df_k * np.log(n))
    # interpretability under promax (stable for k >= 4): salient (>=.40) indicators per factor
    _, L_pm, phi_pm = fit_efa(k, "promax")
    sal = [(np.abs(L_pm[:, j]) >= .40).sum() for j in range(k)]
    cmp_rows.append([k, r_, pct_, float((comm_u > 1).sum()), float(chi2_k), float(df_k), rmsea, tli, bic, int(min(sal)),
                     mp[k] if k < len(mp) else np.nan])
cmp = pd.DataFrame(cmp_rows, columns=["k", "RMSR_minres", "pct_resid_gt_.05", "n_Heywood_minres", "ML_chi2", "ML_df", "RMSEA",
                                      "TLI", "BIC", "min_salient_per_factor_promax", "MAP"]).set_index("k")
save_table(cmp, "D2_efa_solution_comparison")
print(cmp.round(3))
R["D"]["compare"] = cmp.round(4).reset_index().values.tolist()

# KMO sensitivity: iteratively drop the variable with the lowest MSA
drop_rows, cur = [], list(FEATURES)
for _ in range(4):
    kv, kt = calculate_kmo(df[cur])
    worst = cur[int(np.argmin(kv))]
    drop_rows.append([len(cur), kt, worst, float(np.min(kv))])
    cur.remove(worst)
kmo_sens = pd.DataFrame(drop_rows, columns=["n_variables", "overall_KMO", "lowest_MSA_variable(next dropped)", "lowest_MSA"])
save_table(kmo_sens, "D2b_kmo_sensitivity", index=False)
print(kmo_sens.round(3))
R["D"]["kmo_sens"] = kmo_sens.values.tolist()

# ---- final solution --------------------------------------------------------
# Decision (see report, Section 6.4): K = 3. Parallel analysis on reduced correlations suggests 7 (an artefact of n = 6,497 that
# yields one-indicator factors and Heywood cases), MAP suggests 2, the common-factor scree elbow and Kaiser criterion
# suggest 3, and only the 3-factor solution is both interpretable (>= 3 salient indicators per factor) and consistent with PCA.
K_EFA = 3
R["D"]["K"] = K_EFA
fa, L, phi = fit_efa(K_EFA, "oblimin")
fa_v, L_v, _ = fit_efa(K_EFA, "varimax")
comm = np.diag(L @ phi @ L.T); uniq = 1 - comm
struct_m = L @ phi
cols = [f"F{j+1}" for j in range(K_EFA)]
pattern = pd.DataFrame(L, index=FEATURES, columns=cols)
pattern["communality"] = comm; pattern["uniqueness"] = uniq
pattern["max_abs_loading"] = np.abs(L).max(1)
pattern["n_loadings_ge_.30"] = (np.abs(L) >= .30).sum(1)
save_table(pattern, "D3_efa_pattern_oblimin")
save_table(pd.DataFrame(struct_m, index=FEATURES, columns=cols), "D4_efa_structure_matrix")
save_table(pd.DataFrame(phi, index=cols, columns=cols), "D5_efa_factor_correlations")
save_table(pd.DataFrame(L_v, index=FEATURES, columns=cols).assign(communality=(L_v ** 2).sum(1)), "D6_efa_varimax_for_comparison")
print(pattern.round(3)); print("Factor correlations\n", pd.DataFrame(phi, index=cols, columns=cols).round(3))
ss = np.array([(L[:, j] * struct_m[:, j]).sum() for j in range(K_EFA)])
R["D"].update(ss=ss.tolist(), prop_var=(ss / p).tolist(), cum_var=float(ss.sum() / p), phi=phi.tolist())
R["D"]["rmsr"], R["D"]["pct_resid"] = rmsr(L, phi)
R["D"]["heywood"] = [v for v, c_ in zip(FEATURES, comm) if c_ > 1]
R["D"]["max_abs_phi"] = float(np.abs(phi[np.triu_indices(K_EFA, 1)]).max()) if K_EFA > 1 else 0.0


def congruence(A, B):
    return [float((A[:, j] @ B[:, j]) / np.sqrt((A[:, j] ** 2).sum() * (B[:, j] ** 2).sum())) for j in range(A.shape[1])]


R["D"]["congruence_oblimin_varimax"] = congruence(L, L_v)
print("RMSR:", R["D"]["rmsr"], "| %|resid|>.05:", R["D"]["pct_resid"], "| Heywood:", R["D"]["heywood"], "| var:", R["D"]["cum_var"])

plt.figure(figsize=(6.5, 6))
sns.heatmap(pd.DataFrame(L, index=FEATURES, columns=cols), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            linewidths=.5, cbar_kws={"label": "Pattern loading"})
plt.title(f"EFA pattern loadings (minres, oblimin; {K_EFA} factors)")
save_fig("fig11_efa_pattern_heatmap")

# sensitivity: factor structure on within-colour-centred data
Xw = df[FEATURES] - df.groupby("color")[FEATURES].transform("mean")
_, Lw, _ = fit_efa(K_EFA, "oblimin", data=Xw)
cong = np.array([[abs((L[:, i] @ Lw[:, j]) / np.sqrt((L[:, i] ** 2).sum() * (Lw[:, j] ** 2).sum())) for j in range(K_EFA)] for i in range(K_EFA)])
R["D"]["within_color_congruence_max"] = cong.max(1).tolist()
save_table(pd.DataFrame(Lw, index=FEATURES, columns=cols), "D7_efa_within_color_pattern")
print("Congruence with within-colour solution:", cong.round(3))

with open(os.path.join(BASE, "output", "results.json"), "w") as f:
    json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("\nDone. Figures:", len(os.listdir(FIG)), "| Tables:", len(os.listdir(TAB)))
