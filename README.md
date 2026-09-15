# Vasicek mean regression — computational verification

Computational verification and reproducibility material for:

> **Vasicek mean regression: Estimation and diagnostics**
> Bruna Alves, Josmar Mazucheli, Víctor Leiva and Francisco Bustillos.

This repository holds the material produced for the verification of the article.
The SAS code of the Monte Carlo study of Sections 5.2 to 5.4 travels with the
submission and is not reproduced here.

Everything below runs with Python and needs only `numpy` and `scipy`.

---

## `oecd/` — the application of Section 6

| file | what it is |
|---|---|
| `oecd_bli_2017.csv` | the 38 countries and four variables, already on the modelling scale |
| `fit_all.py` | fits six unit regression models to those data and compares them |
| `PROVENANCE.txt` | where the data come from, how each column was mapped, and what was checked |

`fit_all.py` starts by reproducing the fit reported in the article and stops if it
fails. That reproduction is the check that validates the data set: eight published
quantities to four decimal places, including the log-likelihood 38.3997, the shape
estimate 0.1119, and the four coefficients with their standard errors. A different
data set would not reproduce them.

It then fits six families, all with five parameters, all with a logit link on the
conditional mean, and verifies by numerical integration that each fitted density
integrates to one over the unit interval, so that the maximized log-likelihoods are
comparable. Output: the ordering by AIC, the Akaike weights, and Vuong tests
against the Vasicek model.

```
python oecd/fit_all.py
```

### On the provenance of the data

The 2017 edition of the OECD Better Life Index is no longer served by the official
route: the legacy OECD.Stat portal was retired and the current SDMX API serves only
the current edition, whose values differ. The file here comes from a third-party
mirror, and it is described as such. It was cross-checked by sampling against
archived snapshots of the official site, and the strong check is the numerical
reproduction described above. `PROVENANCE.txt` gives the literal URL, the checksum
and the dating trap that catches auditors who compare against the wrong snapshot.

---

## `misspecification/` — do the residual diagnostics detect anything?

The Monte Carlo study of the article generates Vasicek data and fits the Vasicek
model, so it measures calibration under the null and never power. This is the
missing experiment: the response is generated from other families and the Vasicek
model is always fitted.

| file | what it is |
|---|---|
| `misspec.py` | moments of both residuals and rejection rates with the textbook critical values |
| `misspec2.py` | the same, with **both tests calibrated** to five percent under the correct model |
| `results_moments.txt` | output of the first, 1000 replications |
| `results_calibrated.txt` | output of the second |

```
python misspecification/misspec.py 1000
python misspecification/misspec2.py 600
```

The second script exists because the first showed that the Anderson-Darling test on
the Cox-Snell residual, used with its textbook critical value, has a type I error of
0.002 to 0.007 against a nominal 0.05. Comparing power at that point would flatter
the quantile residual for the wrong reason, so each threshold is taken from the
95th percentile of its own null distribution and both tests then have five percent
by construction.

---

## What this material shows

1. In the intercept-only model the fitted quantile residual is exactly the
   studentized sample of the probit of the response. Its sample mean is zero and
   its sample variance is `n/(n-1)` in every replication, and this holds when the
   response is generated from beta, Kumaraswamy or simplex data as well. Two of the
   four reported moments therefore have no power against any alternative.
2. The skewness and the excess kurtosis do carry signal, and its sign separates the
   families: positive under beta and Kumaraswamy, negative under simplex.
3. Against a wider set of unit regression models fitted to the same data, all with
   five parameters and the same link, the Vasicek model ranks fifth of six.

---

Issues and corrections: open an issue in this repository.

---

## `bbm/` — metric completion of the Bender-Brody-Muller Hamiltonian

An unrelated line of work kept in the same repository: a four-step examination of whether the
metric completion of the BBM Hamiltonian can discretise its spectrum, and whether the result has
anything to do with the Riemann zeros. See [`bbm/README.md`](bbm/README.md).

Short answer: no, and the obstruction is structural — in the dilation representation the metric
is a gauge factor that cancels out of the eigenvalue equation.

