"""Code for the plots"""

import matplotlib.pyplot as plt
import numpy as np


def _asset_label(asset):
    return "S&P 500 CFD" if asset == "E_SandP-500" else asset


def plot_lagged_correlations(
    long_table,
    ax=None,
    title=None,
    leader="X",
    target="Y",
    tick_step=5,
    title_fontsize=15,
    label_fontsize=13,
    tick_fontsize=11,
):
    """Plots mean lagged correlations and their ranges on a bar chart"""
    summary = (
        long_table.groupby("lag")["correlation"].agg(["mean", "std"]).reset_index()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    leader_label = _asset_label(leader)
    target_label = _asset_label(target)

    colors = ["#2b5c8f" if val >= 0 else "#c44e52" for val in summary["mean"]]
    ax.bar(
        summary["lag"],
        summary["mean"],
        yerr=summary["std"],
        color=colors,
        width=0.7,
        capsize=1.5,
        error_kw={"elinewidth": 0.7, "ecolor": "black", "alpha": 0.5},
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
    ax.set_xlabel(
        f"Lead {target_label} <- | -> Lead {leader_label}",
        fontsize=label_fontsize,
    )
    ax.set_xticks([l for l in summary["lag"] if l % tick_step == 0])
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    return ax


def plot_fstat_and_best_lag(
    long_table,
    axes=None,
    leader="X",
    target="Y",
    group_label=None,
    title_fontsize=15,
    label_fontsize=13,
    tick_fontsize=11,
):
    """Plots f-stat values and distribution of best lags across events/control windows"""
    leader_label = _asset_label(leader)
    target_label = _asset_label(target)
    n_events = long_table["realization"].nunique()

    summary = long_table.groupby("lag")["f_stat"].agg(["mean", "std"]).reset_index()
    best_lag_per_event = long_table.loc[
        long_table.groupby("realization")["f_stat"].idxmax()
    ][["realization", "lag"]]

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax_fstat, ax_bestlag = axes

    ax_fstat.plot(
        summary["lag"], summary["mean"], marker="o", markersize=4, color="tab:blue"
    )
    ax_fstat.fill_between(
        summary["lag"],
        summary["mean"] - summary["std"],
        summary["mean"] + summary["std"],
        color="tab:blue",
        alpha=0.2,
        label="±1 std across events",
    )
    ax_fstat.set_xlabel("Lag order (k)", fontsize=label_fontsize)
    ax_fstat.set_ylabel("F-statistic", fontsize=label_fontsize)
    title_fstat = (
        f"{group_label} (n={n_events})"
        if group_label
        else f"Mean F-stat by lag (n={n_events})"
    )
    ax_fstat.set_title(title_fstat, fontsize=title_fontsize, fontweight="bold")
    ax_fstat.tick_params(axis="both", labelsize=tick_fontsize)
    ax_fstat.legend(fontsize=tick_fontsize)
    ax_fstat.grid(axis="y", linestyle=":", alpha=0.5)

    max_lag = int(long_table["lag"].max())
    weights = (
        np.ones(len(best_lag_per_event)) / n_events
    )  # counts -> proportion of events
    ax_bestlag.hist(
        best_lag_per_event["lag"],
        bins=range(1, max_lag + 2),
        weights=weights,
        align="left",
        color="tab:blue",
        edgecolor="white",
    )
    ax_bestlag.set_xlabel("Lag", fontsize=label_fontsize)
    ax_bestlag.set_ylabel("Proportion of events", fontsize=label_fontsize)
    if not group_label:
        ax_bestlag.set_title(
            f"Distribution of best lag (n={n_events})",
            fontsize=title_fontsize,
            fontweight="bold",
        )
    ax_bestlag.tick_params(axis="both", labelsize=tick_fontsize)
    ax_bestlag.grid(axis="y", linestyle=":", alpha=0.5)

    if standalone:
        fig.suptitle(
            f"Granger Causality: does {leader_label} help predict {target_label}?",
            fontsize=title_fontsize + 2,
            fontweight="bold",
            y=1.03,
        )
        plt.tight_layout()
        plt.show()
    return axes


def plot_tdmi(
    table,
    ax=None,
    leader="X",
    target="Y",
    group_label=None,
    title_fontsize=15,
    label_fontsize=13,
    tick_fontsize=11,
):
    """Plots pooled TDMI values (in nats)"""
    leader_label = _asset_label(leader)
    target_label = _asset_label(target)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.3))

    ax.plot(table["lag"], table["tdmi"], marker="o", color="tab:green", markersize=4)
    ax.axhline(0, color="#666666", linewidth=0.9)
    ax.set_xlabel("Lag", fontsize=label_fontsize)
    ax.set_ylabel("TDMI (nats)", fontsize=label_fontsize)

    if group_label:
        ax.set_title(group_label, fontsize=title_fontsize, fontweight="bold")
    else:
        ax.set_title(
            f"TDMI: I({target_label}$_t$; {leader_label}$_{{t-lag}}$)",
            fontsize=title_fontsize,
            fontweight="bold",
        )

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    if standalone:
        plt.tight_layout()
        plt.show()
    return ax


def plot_transfer_entropy(
    table,
    ax=None,
    leader="X",
    target="Y",
    group_label=None,
    title_fontsize=15,
    label_fontsize=13,
    tick_fontsize=11,
):
    """Plots pooled TE values (in nats)"""
    leader_label = _asset_label(leader)
    target_label = _asset_label(target)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.3))

    ax.plot(
        table["lag"],
        table["transfer_entropy"],
        marker="o",
        color="tab:purple",
        markersize=4,
    )
    ax.axhline(0, color="#666666", linewidth=0.9)
    ax.set_xlabel("Lag", fontsize=label_fontsize)
    ax.set_ylabel("Transfer entropy (nats)", fontsize=label_fontsize)

    if group_label:
        ax.set_title(group_label, fontsize=title_fontsize, fontweight="bold")
    else:
        ax.set_title(
            f"TE: I({target_label}$_t$; {leader_label}$_{{t-lag}}$ | {target_label}$_{{t-lag}}$)",
            fontsize=title_fontsize,
            fontweight="bold",
        )

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    if standalone:
        plt.tight_layout()
        plt.show()
    return ax


def plot_nonlinear_transfer_entropy(
    table,
    ax=None,
    leader="X",
    target="Y",
    group_label=None,
    title_fontsize=15,
    label_fontsize=13,
    tick_fontsize=11,
):
    """Plots nonlinear TE values (in nats)"""
    leader_label = _asset_label(leader)
    target_label = _asset_label(target)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.3))

    ax.plot(table["lag"], table["te_nl"], marker="o", color="tab:brown", markersize=4)
    ax.axhline(0, color="#666666", linewidth=0.9)
    ax.set_xlabel("Lag", fontsize=label_fontsize)
    ax.set_ylabel("TE$_{NL}$ (nats)", fontsize=label_fontsize)

    if group_label:
        ax.set_title(group_label, fontsize=title_fontsize, fontweight="bold")
    else:
        ax.set_title(
            f"TE$_{{NL}}$: I({target_label}$_t$; {leader_label}$_{{t-lag}}$ | f({target_label}$_{{t-lag}}$))",
            fontsize=title_fontsize,
            fontweight="bold",
        )

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    if standalone:
        plt.tight_layout()
        plt.show()
    return ax
