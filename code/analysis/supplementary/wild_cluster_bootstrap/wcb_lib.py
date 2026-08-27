"""
wcb_lib.py
----------
Reusable wild cluster bootstrap (WCB) primitives shared by
run_wild_cluster_bootstrap.py. Two independent code paths, matched to what
each hypothesis actually needs:

1. Single linear restriction (one coefficient, or a difference between two
   named coefficients, e.g. contrast diffs, IN=DN equality) ->
   `wcb_single_restriction`, a thin wrapper around the third-party
   `wildboottest` package's `WildboottestCL` class (Roodman et al. 2019
   "fast and wild" algorithm; the same engine behind Stata's `boottest` /
   R's `fwildclusterboot`). This is the "prefer a well-tested existing
   implementation" path.

2. Joint multi-restriction Wald tests (q > 1 restrictions simultaneously,
   e.g. an omnibus "all condition dummies = 0" test) -> `wcb_joint_wald`,
   a from-scratch WCR (null-imposed wild cluster restricted) bootstrap,
   because WildboottestCL's public R-vector interface only supports a
   single linear combination (R must be a length-k vector, not a q x k
   matrix). Implements the standard Cameron-Gelbach-Miller (2008) /
   Roodman et al. (2019) joint-restriction recipe:
     - fit the restricted model imposing R*beta = r exactly (closed form),
     - re-weight the restricted residuals by a cluster-level bootstrap
       weight (Rademacher or Webb six-point) to build B pseudo-outcomes,
     - refit the unrestricted OLS on each pseudo-outcome (X is fixed, so
       (X'X)^{-1}X' is precomputed once and reused -- no per-iteration
       formula refit),
     - recompute a cluster-robust (CR1, small-sample-adjusted to match
       statsmodels' cov_type="cluster" default) Wald/F statistic each time,
     - p-value = share of bootstrap statistics >= the observed statistic.

Both paths use Webb six-point weights by default (recommended for small
cluster counts) and impose the null on the bootstrap DGP (WCR), matching
the brief. A fixed seed is threaded through every call for reproducibility.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from wildboottest.wildboottest import WildboottestCL, draw_weights

SEED_BASE = 20260824  # fixed WCB reproducibility seed (today's date, YYYYMMDD)
DEFAULT_B = 9999
DEFAULT_WEIGHTS = "webb"


def aligned_cluster(fit_result, df, cluster_col):
    """Cluster labels for the exact rows statsmodels retained after any
    listwise deletion, in the same row order as fit_result.model.exog.
    `row_labels` is the original DataFrame index of the retained rows."""
    labels = fit_result.model.data.row_labels
    return df.loc[labels, cluster_col].values, labels


def make_R_vector(xnames, pos_name=None, neg_name=None):
    """R vector of length k for a single restriction: +1 at pos_name's
    column, -1 at neg_name's column (either may be None -> implicit 0,
    e.g. an omitted reference level), all else 0."""
    k = len(xnames)
    R = np.zeros(k)
    if pos_name is not None:
        R[xnames.index(pos_name)] = 1.0
    if neg_name is not None:
        R[xnames.index(neg_name)] = -1.0
    return R


def wcb_single_restriction(exog, endog, cluster, R, B=DEFAULT_B,
                            weights_type=DEFAULT_WEIGHTS, seed_offset=0,
                            impose_null=True, bootstrap_type="11"):
    """Single-restriction wild cluster bootstrap via WildboottestCL.
    R is a length-k vector; the tested restriction is R'beta = 0.
    Cluster labels are factorized to integer codes first -- WildboottestCL's
    numba-jitted inner loop cannot type a raw string/object array."""
    codes, _ = pd.factorize(np.asarray(cluster))
    boot = WildboottestCL(
        X=np.asarray(exog, dtype=float),
        Y=np.asarray(endog, dtype=float).ravel(),
        cluster=codes.astype(np.int64),
        R=np.asarray(R, dtype=float),
        B=B,
        seed=SEED_BASE + seed_offset,
    )
    boot.get_scores(bootstrap_type=bootstrap_type, impose_null=impose_null,
                     adj=True, cluster_adj=True)
    _, _, full_enum_warn = boot.get_weights(weights_type=weights_type)
    boot.get_numer()
    boot.get_denom()
    boot.get_tboot()
    boot.get_vcov()
    boot.get_tstat()
    boot.get_pvalue(pval_type="two-tailed")
    return {
        "wcb_t": float(boot.t_stat),
        "wcb_p": float(boot.pvalue),
        "B_requested": int(B),
        "B_effective": int(boot.B),
        "G": int(boot.G),
        "weights_type": weights_type,
        "full_enumeration": bool(full_enum_warn),
    }


def _cluster_indicator(codes, G):
    n = len(codes)
    return sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, G))


def wcb_joint_wald(exog, endog, cluster, R, r=None, B=DEFAULT_B,
                    weights_type=DEFAULT_WEIGHTS, seed_offset=0,
                    adj=True, cluster_adj=True):
    """Joint (q > 1 restriction) WCR wild cluster bootstrap Wald/F test for
    H0: R @ beta = r. R is a (q, k) matrix. Returns the observed Wald/F
    statistic (matches statsmodels' cluster-robust Wald F exactly -- see
    the QC cross-check in run_wild_cluster_bootstrap.py) and its
    null-imposed wild-cluster-bootstrap p-value."""
    X = np.asarray(exog, dtype=float)
    y = np.asarray(endog, dtype=float).ravel()
    N, k = X.shape
    R = np.atleast_2d(np.asarray(R, dtype=float))
    q = R.shape[0]
    if r is None:
        r = np.zeros(q)

    codes, uniques = pd.factorize(np.asarray(cluster))
    G = len(uniques)
    D = _cluster_indicator(codes, G)

    XtX_inv = np.linalg.inv(X.T @ X)
    P = XtX_inv @ X.T  # k x N, reused for every bootstrap refit (X is fixed)

    beta_hat = P @ y
    resid_hat = y - X @ beta_hat

    def cr1_vcov(resid):
        Xu = X * resid[:, None]
        S = np.asarray(D.T @ Xu)  # G x k, per-cluster scores
        meat = S.T @ S
        vcov = XtX_inv @ meat @ XtX_inv
        if adj:
            vcov = vcov * (N - 1) / (N - k)
        if cluster_adj:
            vcov = vcov * G / (G - 1)
        return vcov

    def wald_stat(beta, resid):
        vcov = cr1_vcov(resid)
        Rb_r = R @ beta - r
        mid = R @ vcov @ R.T
        return float(Rb_r @ np.linalg.solve(mid, Rb_r) / q)

    vcov_hat = cr1_vcov(resid_hat)
    W_obs = wald_stat(beta_hat, resid_hat)

    # Restricted fit imposing R beta = r exactly (closed-form Lagrangian solution)
    RXtXinvRt = R @ XtX_inv @ R.T
    lam = np.linalg.solve(RXtXinvRt, R @ beta_hat - r)
    beta_r = beta_hat - XtX_inv @ R.T @ lam
    resid_r = y - X @ beta_r
    Xbeta_r = X @ beta_r

    rng = np.random.default_rng(SEED_BASE + seed_offset)
    v0, B_eff = draw_weights(weights_type, full_enumeration=False,
                              N_G_bootcluster=G, boot_iter=B, rng=rng)
    w_by_row = v0[codes, :]  # N x B_eff, cluster weight broadcast to its rows

    W_boot = np.empty(B_eff)
    for b in range(B_eff):
        y_star = Xbeta_r + w_by_row[:, b] * resid_r
        beta_star = P @ y_star
        resid_star = y_star - X @ beta_star
        W_boot[b] = wald_stat(beta_star, resid_star)

    p_boot = float(np.mean(W_boot >= W_obs))
    return {
        "W_obs": W_obs, "wcb_p": p_boot, "q": q,
        "B_requested": int(B), "B_effective": int(B_eff), "G": int(G),
        "beta_hat": beta_hat, "vcov_hat": vcov_hat,
        "weights_type": weights_type,
    }


def fmt_p(p, B=DEFAULT_B):
    """Bootstrap p-values are a Monte Carlo mean over B draws (see
    WildboottestCL.get_pvalue / wcb_joint_wald's plain np.mean count) with no
    add-one correction, so a value of exactly 0.0 means 'no draw was as
    extreme', not 'the true p-value is zero'. Render that floor honestly."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p == 0.0:
        return f"<{1.0 / B:.4f}"
    return f"{p:.3g}"


def classify_conclusion(p_asym, p_wcb, alpha=0.05):
    same_side = (p_asym < alpha) == (p_wcb < alpha)
    if same_side:
        return "same conclusion"
    if p_asym < alpha and p_wcb >= alpha:
        return "asymptotic significant, WCB not significant"
    if p_asym >= alpha and p_wcb < alpha:
        return "asymptotic not significant, WCB significant"
    return "same conclusion"
