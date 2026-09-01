"""Model implementation for our analysis"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# Helper functions to summarize key stats across realizations


def summarize_across_realizations(pairs, compute_fn, max_lag=10, **kwargs):
    """Runs compute_fn (e.g. compute_granger_causality) on each input (x, y) pair
    stacking the results into one long table tagged by a realization id."""
    tables = []
    for i, (x, y) in enumerate(pairs):
        table = compute_fn(x, y, max_lag=max_lag, **kwargs)
        table["realization"] = i
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def summarize_best_lag_and_hit_rate(
    long_table, stat_col="f_stat", p_col="p_value", alpha=0.05
):
    """Given a long table of summary stats, computes best lagper realization
    and hit rate by lag"""
    best_lag_per_realization = long_table.loc[
        long_table.groupby("realization")[stat_col].idxmax()
    ][["realization", "lag"]].rename(columns={"lag": "best_lag"})
    hit_rate_by_lag = (
        long_table.assign(hit=long_table[p_col] < alpha)
        .groupby("lag")["hit"]
        .mean()
        .reset_index(name="hit_rate")
    )
    return best_lag_per_realization, hit_rate_by_lag


# Helper functions for the implementation of models 2 through 5


def _fit_ols_rss(design, target):
    """Computes OLS via least squares. Returns residual sum of squares
    and number of predictors including the intercept"""
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    return np.sum(residuals**2), design.shape[1]


def discretize(series, n_bins=8):
    """Bins a continuous series into n_bins of equal-frequency"""
    return pd.qcut(pd.Series(series), q=n_bins, labels=False, duplicates="drop")


def mutual_information(a, b):
    """Computes the mutual information between a and b"""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    n = len(df)

    joint = df.groupby(["a", "b"]).size() / n
    marginal_a = df.groupby("a").size() / n
    marginal_b = df.groupby("b").size() / n

    mi = 0.0
    for (a_val, b_val), p_ab in joint.items():
        mi += p_ab * np.log(p_ab / (marginal_a.loc[a_val] * marginal_b.loc[b_val]))
    return mi


def conditional_mutual_information(a, b, c):
    """Computes the conditional mutual information of a and b given c"""
    df = pd.DataFrame({"a": a, "b": b, "c": c}).dropna()
    n = len(df)

    joint_abc = df.groupby(["a", "b", "c"]).size() / n
    joint_ac = df.groupby(["a", "c"]).size() / n
    joint_bc = df.groupby(["b", "c"]).size() / n
    marginal_c = df.groupby("c").size() / n

    cmi = 0.0
    for (a_val, b_val, c_val), p_abc in joint_abc.items():
        p_ac = joint_ac.loc[(a_val, c_val)]
        p_bc = joint_bc.loc[(b_val, c_val)]
        p_c = marginal_c.loc[c_val]
        cmi += p_abc * np.log((p_abc * p_c) / (p_ac * p_bc))
    return cmi


def _jitter(series, scale=1e-6, seed=42):
    """Adds small Gaussian noise to break exact ties before quantile binning"""
    rng = np.random.default_rng(seed)
    s = pd.Series(series).reset_index(drop=True)
    std = s.std()
    if std == 0 or np.isnan(std):
        return s
    noise = rng.normal(loc=0.0, scale=scale * std, size=len(s))
    return s + noise


# Models


def compute_lagged_correlations(x, y, max_lag=10):
    """Computes correlation between Y and X_lagged"""
    x, y = pd.Series(x), pd.Series(y)
    lags = np.arange(-max_lag, max_lag + 1)

    rows = []
    for lag in lags:
        aligned = pd.concat([y, x.shift(lag)], axis=1, keys=["y", "x_shifted"]).dropna()
        correlation = aligned["y"].corr(aligned["x_shifted"])
        rows.append({"lag": lag, "correlation": correlation})

    return pd.DataFrame(rows)


def compute_granger_causality(x, y, max_lag=10):
    """Computes granger causality"""
    x, y = pd.Series(x), pd.Series(y)

    rows = []
    for k in range(1, max_lag + 1):
        # Building the aligned (y_now, y_lags, x_lags) table for every lag order
        data = {"y_now": y}
        for lag in range(1, k + 1):
            data[f"y_lag{lag}"] = y.shift(lag)
            data[f"x_lag{lag}"] = x.shift(lag)
        aligned = pd.DataFrame(data).dropna()

        target = aligned["y_now"].values
        y_lag_cols = [f"y_lag{lag}" for lag in range(1, k + 1)]
        x_lag_cols = [f"x_lag{lag}" for lag in range(1, k + 1)]
        n = len(aligned)

        # Setting up the "restricted" and "unrestricted" models
        restricted_design = np.column_stack([np.ones(n), aligned[y_lag_cols].values])
        unrestricted_design = np.column_stack(
            [np.ones(n), aligned[y_lag_cols].values, aligned[x_lag_cols].values]
        )

        rss_restricted, n_params_restricted = _fit_ols_rss(restricted_design, target)
        rss_unrestricted, n_params_unrestricted = _fit_ols_rss(
            unrestricted_design, target
        )

        df_num = n_params_unrestricted - n_params_restricted
        df_denom = n - n_params_unrestricted
        # Calculating f-stat
        f_stat = ((rss_restricted - rss_unrestricted) / df_num) / (
            rss_unrestricted / df_denom
        )
        p_value = sp_stats.f.sf(f_stat, df_num, df_denom)

        rows.append({"lag": k, "f_stat": f_stat, "p_value": p_value})

    return pd.DataFrame(rows)


def compute_tdmi_pooled(pairs, max_lag=30, n_bins=8, jitter_scale=1e-6, seed=42):
    """Computes TDMI across a group of events"""
    all_x = pd.concat([pd.Series(x) for x, y in pairs], ignore_index=True)
    all_y = pd.concat([pd.Series(y) for x, y in pairs], ignore_index=True)

    all_x_j = _jitter(all_x, scale=jitter_scale, seed=seed)
    all_y_j = _jitter(all_y, scale=jitter_scale, seed=seed + 1)
    _, x_bin_edges = pd.qcut(all_x_j, q=n_bins, retbins=True, duplicates="drop")
    _, y_bin_edges = pd.qcut(all_y_j, q=n_bins, retbins=True, duplicates="drop")

    lags = np.arange(1, max_lag + 1)
    rows = []
    for lag in lags:
        now_pooled, past_pooled = [], []
        for i, (x, y) in enumerate(pairs):
            x_j = _jitter(x, scale=jitter_scale, seed=seed + 100 + i)
            y_j = _jitter(y, scale=jitter_scale, seed=seed + 200 + i)
            x_binned = pd.cut(x_j, bins=x_bin_edges, labels=False, include_lowest=True)
            y_binned = pd.cut(y_j, bins=y_bin_edges, labels=False, include_lowest=True)
            aligned = pd.DataFrame(
                {"now": y_binned, "past": x_binned.shift(lag)}
            ).dropna()
            now_pooled.append(aligned["now"])
            past_pooled.append(aligned["past"])
        now_all = pd.concat(now_pooled, ignore_index=True)
        past_all = pd.concat(past_pooled, ignore_index=True)
        mi = mutual_information(now_all, past_all)
        rows.append({"lag": lag, "tdmi": mi})
    return pd.DataFrame(rows)


def compute_te_pooled(pairs, max_lag=30, n_bins=8, jitter_scale=1e-6, seed=42):
    """Computes Transfer Entropy (TE) across a group of events"""
    all_x = pd.concat([pd.Series(x) for x, y in pairs], ignore_index=True)
    all_y = pd.concat([pd.Series(y) for x, y in pairs], ignore_index=True)

    all_x_j = _jitter(all_x, scale=jitter_scale, seed=seed)
    all_y_j = _jitter(all_y, scale=jitter_scale, seed=seed + 1)
    _, x_bin_edges = pd.qcut(all_x_j, q=n_bins, retbins=True, duplicates="drop")
    _, y_bin_edges = pd.qcut(all_y_j, q=n_bins, retbins=True, duplicates="drop")

    lags = np.arange(1, max_lag + 1)
    rows = []
    for lag in lags:
        y_now_pooled, x_past_pooled, y_past_pooled = [], [], []
        for i, (x, y) in enumerate(pairs):
            x_j = _jitter(x, scale=jitter_scale, seed=seed + 100 + i)
            y_j = _jitter(y, scale=jitter_scale, seed=seed + 200 + i)
            x_binned = pd.cut(x_j, bins=x_bin_edges, labels=False, include_lowest=True)
            y_binned = pd.cut(y_j, bins=y_bin_edges, labels=False, include_lowest=True)
            aligned = pd.DataFrame(
                {
                    "y_now": y_binned,
                    "x_past": x_binned.shift(lag),
                    "y_past": y_binned.shift(lag),
                }
            ).dropna()
            y_now_pooled.append(aligned["y_now"])
            x_past_pooled.append(aligned["x_past"])
            y_past_pooled.append(aligned["y_past"])
        y_now_all = pd.concat(y_now_pooled, ignore_index=True)
        x_past_all = pd.concat(x_past_pooled, ignore_index=True)
        y_past_all = pd.concat(y_past_pooled, ignore_index=True)
        te = conditional_mutual_information(y_now_all, x_past_all, y_past_all)
        rows.append({"lag": lag, "transfer_entropy": te})
    return pd.DataFrame(rows)


def compute_te_nl_pooled(
    pairs, max_lag=30, n_bins=8, f=lambda z: z**2, jitter_scale=1e-6, seed=42
):
    """Computes nonlinear TE on a group of events (conditioned on the target's squared past)"""
    all_x = pd.concat([pd.Series(x) for x, y in pairs], ignore_index=True)
    all_y = pd.concat([pd.Series(y) for x, y in pairs], ignore_index=True)
    all_yf = f(all_y)

    all_x_j = _jitter(all_x, scale=jitter_scale, seed=seed)
    all_y_j = _jitter(all_y, scale=jitter_scale, seed=seed + 1)
    all_yf_j = _jitter(all_yf, scale=jitter_scale, seed=seed + 2)
    _, x_bin_edges = pd.qcut(all_x_j, q=n_bins, retbins=True, duplicates="drop")
    _, y_bin_edges = pd.qcut(all_y_j, q=n_bins, retbins=True, duplicates="drop")
    _, yf_bin_edges = pd.qcut(all_yf_j, q=n_bins, retbins=True, duplicates="drop")

    lags = np.arange(1, max_lag + 1)
    rows = []
    for lag in lags:
        y_now_pooled, x_past_pooled, yf_past_pooled = [], [], []
        for i, (x, y) in enumerate(pairs):
            yf = f(pd.Series(y).reset_index(drop=True))
            x_j = _jitter(x, scale=jitter_scale, seed=seed + 100 + i)
            y_j = _jitter(y, scale=jitter_scale, seed=seed + 200 + i)
            yf_j = _jitter(yf, scale=jitter_scale, seed=seed + 300 + i)
            x_binned = pd.cut(x_j, bins=x_bin_edges, labels=False, include_lowest=True)
            y_binned = pd.cut(y_j, bins=y_bin_edges, labels=False, include_lowest=True)
            yf_binned = pd.cut(
                yf_j, bins=yf_bin_edges, labels=False, include_lowest=True
            )
            aligned = pd.DataFrame(
                {
                    "y_now": y_binned,
                    "x_past": x_binned.shift(lag),
                    "yf_past": yf_binned.shift(lag),
                }
            ).dropna()
            y_now_pooled.append(aligned["y_now"])
            x_past_pooled.append(aligned["x_past"])
            yf_past_pooled.append(aligned["yf_past"])
        y_now_all = pd.concat(y_now_pooled, ignore_index=True)
        x_past_all = pd.concat(x_past_pooled, ignore_index=True)
        yf_past_all = pd.concat(yf_past_pooled, ignore_index=True)
        te_nl = conditional_mutual_information(y_now_all, x_past_all, yf_past_all)
        rows.append({"lag": lag, "te_nl": te_nl})
    return pd.DataFrame(rows)
