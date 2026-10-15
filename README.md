# Vasicek mean regression — code and data

Reproducibility material for:

> **Vasicek Mean Regression for Bounded Data: Modeling, Inference, Diagnostics, and Computational Studies**
> Víctor Leiva, Bruna Alves, Josmar Mazucheli and Francisco Bustillos.

Section, table and figure numbers below refer to the revised manuscript (CMES_92843, first revision).
The SAS code of the Monte Carlo study of Parts I to III (Sections 5 and 6), with its archived output,
travels with the submission and is not reproduced here.

Everything in Python needs only `numpy` and `scipy` (`matplotlib` for the figures). `oecd_refit.R`
needs the R packages `gamlss` and `vasicekreg` (whose Vasicek family is `NVASIM`).

---

## `oecd/` — the application of Section 8

| file | what it is |
|---|---|
| `oecd_bli_2017.csv` | the 38 countries and four variables, already on the modelling scale |
| `fit_all.py` | fits six unit regression models to those data and compares them (Tables 8 to 10) |
| `PROVENANCE.txt` | where the data come from, how each column was mapped, and what was checked |

`fit_all.py` starts by reproducing the Vasicek fit reported in the article and stops if it fails:
eight published quantities to four decimal places, including the log-likelihood 38.3997 and the
shape estimate 0.1119. It then fits six families, all with five parameters and a logit link on the
conditional mean, and verifies by numerical integration that each fitted density integrates to one,
so that the maximized log-likelihoods are comparable.

```
python oecd/fit_all.py
```

The mean of the unit-Weibull model has no closed form. It is evaluated with a trapezoid rule on a
logarithmic scale, whose relative error is below 1e-12 against a high-precision reference; the
Gauss-Laguerre rule of the first version of this file was replaced because its relative error
reached 1.0 for small values of the scale parameter.

### On the provenance of the data

The 2017 edition of the OECD Better Life Index is no longer served by the official route: the
legacy OECD.Stat portal was retired and the current SDMX API serves only the current edition,
whose values differ. The file here comes from a third-party mirror, and it is described as such.
It was cross-checked against archived snapshots of the official site, and the strong check is the
numerical reproduction described above. `PROVENANCE.txt` gives the literal URL and the checksum.

---

## `part_iv/` — the misspecification study of Section 7

Each competing law (beta, Kumaraswamy, simplex) is matched to the Vasicek distribution in the mean
and in the variance at every observation, so that only the shape of the density differs, and the
test of Shapiro and Wilk on the quantile residual and that of Anderson and Darling on the generalized
Cox-Snell residual are applied to the fitted Vasicek model (Tables 5 to 7, B1 and B2).

| file | what it is |
|---|---|
| `msp2_core.py` | Vasicek fit, the two test statistics, the alternative densities and their maximized log-likelihoods |
| `msp2_design.py` | the matching: dispersion parameters tabulated per theta, and the samplers |
| `feasibility.py`, `feasibility.json` | checks that each competitor can reproduce the Vasicek variance over the covariate range |
| `msp2_run.py` | the study: phase A the two tests, phase B the likelihood benchmark and the Neyman-Pearson ceiling |
| `msp2_report.py` | merges the blocks and prints the tables |
| `msp2_rules.py` | the calibrated likelihood ratio test against the AIC selection rule |
| `msp2_results_*.json` | the results, one file per block |

All seeds derive from the master seed 20260812 (`msp2_run.py`).

---

## `revision_R1/` — computations added in the first revision

Run from the folder `revision_R1/`; outputs are written to the current folder, and the outputs used
in the article are in `results/`.

| file | what it reproduces |
|---|---|
| `mcse_est.py`, `mcse_resid.py`, `part1_resid_sd.py` | Monte Carlo standard errors of Section 5.3 and Table 1. `mcse_est.py` reads the tables of the submitted manuscript (LaTeX source as argument) and `mcse_resid.py` the archived SAS output (folder as argument); `part1_resid_sd.py` simulates the Part I residual summaries, which are pivotal |
| `matching_accuracy.py` | measured accuracy of the variance matching of Part IV (Section 5.4) |
| `mc_oecd_design.py` | Monte Carlo study at the covariate design of the application, Table 12: `python mc_oecd_design.py 5000 ../oecd/oecd_bli_2017.csv` |
| `get_data.py` | downloads the eight data sets of the R package `betareg` and checks their SHA-256 |
| `six_models.py`, `run_datasets.py` | the six models fitted to those data sets, Table 13: `python run_datasets.py` |
| `portfolio_mechanism.py` | Vasicek against beta on portfolio default fractions, Table 14 (third column) |
| `portfolio_six.py`, `fast_invert.py` | the six models on the first 300 of those samples, Table 14 (last three columns): `python portfolio_six.py 300 m N shard nshards` |
| `check_fast_fit.py` | checks that the faster mean inversion of `fast_invert.py` attains the same maximum as the bisection of `six_models.py` (outputs `results/chk_*.log`) |
| `summarize_ps6.py` | the rows of Table 14 from the files in `results/` |
| `oecd_refit.R` | the fits of Table 8 in `gamlss`, with p-values from the normal and from the t distribution |
| `figures_r1.py` | Figures 1 and 2 |

For long runs, set `OMP_NUM_THREADS=1` when several processes run in parallel.

---

## `misspecification/` — preliminary exploration

The two scripts here are the first exploration of the question that Part IV answers, with the
alternatives matched at the centre of the design only. They are kept for the record; the study
reported in the article is the one in `part_iv/`.

```
python misspecification/misspec.py 1000
python misspecification/misspec2.py 600
```

---

Issues and corrections: open an issue in this repository.
