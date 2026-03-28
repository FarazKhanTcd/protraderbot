import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from training50 import (
    BASE_DIR,
    DEVICE,
    LOOKBACK,
    MEMORY_LEN,
    N_ROUNDS,
    N_STOCKS_PER_ROUND,
    REBALANCE_CANDLES,
    TICKERS,
    TRAIN_FRAC,
    GATPortfolioNet,
    SAVE_PATH,
    build_correlation_graph,
    build_feature_tensor,
    prepare_data,
)

TX_COST_BPS = 5
RESULTS_CSV = BASE_DIR / "backtest_results50.csv"
GROSS_PLOT = BASE_DIR / "gross_performance50.png"
NET_PLOT = BASE_DIR / "gross_vs_net50.png"
WEIGHTS_PLOT = BASE_DIR / "weights_heatmap50.png"


def backtest_gat_50(
    model,
    full_df,
    all_tickers,
    start_idx,
    n_passes=N_ROUNDS,
    n_stocks_per_pass=N_STOCKS_PER_ROUND,
    device=DEVICE,
):
    model.eval()
    model.to(device)

    rng = np.random.default_rng(42 + 999)
    inference_subsets = [list(rng.choice(all_tickers, size=n_stocks_per_pass, replace=False)) for _ in range(n_passes)]
    print("Inference subsets:")
    for i, subset in enumerate(inference_subsets, start=1):
        print(f"  Pass {i}: {subset}")

    effective_start = max(start_idx, LOOKBACK + 4)
    steps = list(range(effective_start, len(full_df) - REBALANCE_CANDLES, REBALANCE_CANDLES))
    records = []
    histories = [[] for _ in range(n_passes)]

    for idx in steps:
        weight_accumulator = {t: 0.0 for t in all_tickers}
        count_accumulator = {t: 0 for t in all_tickers}

        for p, subset in enumerate(inference_subsets):
            x = build_feature_tensor(full_df, subset, idx, LOOKBACK).to(device)
            ei, ea = build_correlation_graph(full_df, subset, idx, LOOKBACK)
            ei, ea = ei.to(device), ea.to(device)

            with torch.no_grad():
                w = model(x, ei, ea, history=histories[p], device=device)
            w_np = w.cpu().numpy()

            for t, wi in zip(subset, w_np):
                weight_accumulator[t] += float(wi)
                count_accumulator[t] += 1

            cur = full_df["Close"][subset].iloc[idx].values
            fut = full_df["Close"][subset].iloc[idx + REBALANCE_CANDLES].values
            rets_sub = (fut - cur) / (cur + 1e-8)
            histories[p].append((w.detach().cpu(), torch.tensor(rets_sub, dtype=torch.float32)))
            if len(histories[p]) > MEMORY_LEN:
                histories[p].pop(0)

        avg_w = np.array(
            [weight_accumulator[t] / max(count_accumulator[t], 1) for t in all_tickers],
            dtype=np.float32,
        )
        avg_w = avg_w / (avg_w.sum() + 1e-8)

        cur_all = full_df["Close"][all_tickers].iloc[idx].values
        fut_all = full_df["Close"][all_tickers].iloc[idx + REBALANCE_CANDLES].values
        rets_all = (fut_all - cur_all) / (cur_all + 1e-8)

        records.append({
            "date": full_df.index[idx],
            "portfolio_ret": float((avg_w * rets_all).sum()),
            **{t: float(wi) for t, wi in zip(all_tickers, avg_w)},
        })

    result = pd.DataFrame(records).set_index("date")
    result["cumulative"] = (1 + result["portfolio_ret"]).cumprod()
    return result


def backtest_equal_weight_rebalanced(full_df, tickers, start_idx):
    n = len(tickers)
    target_w = np.ones(n) / n
    effective_start = max(start_idx, LOOKBACK + 4)
    steps = list(range(effective_start, len(full_df) - REBALANCE_CANDLES, REBALANCE_CANDLES))
    records = []

    for idx in steps:
        cur = full_df["Close"][tickers].iloc[idx].values
        fut = full_df["Close"][tickers].iloc[idx + REBALANCE_CANDLES].values
        rets = (fut - cur) / (cur + 1e-8)

        drifted = target_w * (1 + rets)
        drifted_w = drifted / drifted.sum()
        turnover = float(np.abs(drifted_w - target_w).sum())

        records.append({
            "date": full_df.index[idx],
            "portfolio_ret": float((target_w * rets).sum()),
            "turnover": turnover,
        })

    result = pd.DataFrame(records).set_index("date")
    result["cumulative"] = (1 + result["portfolio_ret"]).cumprod()
    return result


def backtest_buy_and_hold(full_df, tickers, start_idx):
    n = len(tickers)
    effective_start = max(start_idx, LOOKBACK + 4)
    entry = full_df["Close"][tickers].iloc[effective_start].values
    shares = (1.0 / n) / entry
    steps = list(range(effective_start, len(full_df) - REBALANCE_CANDLES, REBALANCE_CANDLES))
    records = []

    for idx in steps:
        cur_val = (shares * full_df["Close"][tickers].iloc[idx].values).sum()
        next_val = (shares * full_df["Close"][tickers].iloc[idx + REBALANCE_CANDLES].values).sum()
        records.append({
            "date": full_df.index[idx],
            "portfolio_ret": float((next_val - cur_val) / (cur_val + 1e-8)),
        })

    result = pd.DataFrame(records).set_index("date")
    result["cumulative"] = (1 + result["portfolio_ret"]).cumprod()
    return result


def get_shares_outstanding(tickers):
    shares = {}
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        shares[ticker] = info.get("sharesOutstanding", info.get("impliedSharesOutstanding", 1e9))
    print("Shares outstanding used (billions, current snapshot):")
    for ticker, shares_out in shares.items():
        print(f"  {ticker:<6} {shares_out / 1e9:.2f}B")
    return shares


def backtest_cap_weighted(full_df, tickers, shares_outstanding, start_idx):
    effective_start = max(start_idx, LOOKBACK + 4)
    steps = list(range(effective_start, len(full_df) - REBALANCE_CANDLES, REBALANCE_CANDLES))
    shares_arr = np.array([shares_outstanding[t] for t in tickers], dtype=np.float64)

    records = []
    prev_w = None

    for idx in steps:
        cur = full_df["Close"][tickers].iloc[idx].values.astype(np.float64)
        fut = full_df["Close"][tickers].iloc[idx + REBALANCE_CANDLES].values.astype(np.float64)
        rets = (fut - cur) / (cur + 1e-8)

        mcaps = cur * shares_arr
        w = mcaps / (mcaps.sum() + 1e-8)

        turnover = float(np.abs(w - prev_w).sum()) if prev_w is not None else 0.0
        prev_w = w.copy()

        records.append({
            "date": full_df.index[idx],
            "portfolio_ret": float((w * rets).sum()),
            "turnover": turnover,
            **{t: float(wi) for t, wi in zip(tickers, w)},
        })

    result = pd.DataFrame(records).set_index("date")
    result["cumulative"] = (1 + result["portfolio_ret"]).cumprod()
    return result


def fetch_spy_returns(test_dates):
    start = test_dates[0] - pd.Timedelta(days=10)
    end = test_dates[-1] + pd.Timedelta(days=10)
    spy = yf.download(
        "SPY",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    close = spy["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    spy_weekly = close.resample("W-FRI").last().dropna()
    spy_ret = spy_weekly.pct_change().dropna()
    common = test_dates.intersection(spy_ret.index)
    result = pd.DataFrame({"portfolio_ret": spy_ret.loc[common].values.flatten()}, index=common)
    result["cumulative"] = (1 + result["portfolio_ret"]).cumprod()
    return result


def compute_metrics(result, name):
    r = result["portfolio_ret"]
    n_weeks = len(r)
    total = result["cumulative"].iloc[-1] - 1
    years = n_weeks / 52
    ann = (1 + total) ** (1 / max(years, 0.01)) - 1
    sharpe = r.mean() / (r.std() + 1e-8) * math.sqrt(52)
    mdd = ((result["cumulative"].cummax() - result["cumulative"]) / result["cumulative"].cummax()).max()
    return {
        "Strategy": name,
        "Total Return": f"{total * 100:+.2f}%",
        "Ann. Return": f"{ann * 100:+.2f}%",
        "Sharpe": f"{sharpe:.3f}",
        "Max Drawdown": f"{mdd * 100:.2f}%",
        "Weeks": n_weeks,
    }


def metrics_net(ret_series, cum_series, name):
    r = ret_series
    n_weeks = len(r)
    total = cum_series.iloc[-1] - 1
    years = n_weeks / 52
    ann = (1 + total) ** (1 / max(years, 0.01)) - 1
    sharpe = r.mean() / (r.std() + 1e-8) * math.sqrt(52)
    mdd = ((cum_series.cummax() - cum_series) / cum_series.cummax()).max()
    return {
        "Strategy": name,
        "Total Return": f"{total * 100:+.2f}%",
        "Ann. Return": f"{ann * 100:+.2f}%",
        "Sharpe": f"{sharpe:.3f}",
        "Max Drawdown": f"{mdd * 100:.2f}%",
    }


def main():
    weekly_df, train_df, test_df, split_idx = prepare_data()
    print(
        f"Weekly bars: {len(weekly_df)}  |  "
        f"{weekly_df.index[0].date()} -> {weekly_df.index[-1].date()}"
    )
    print(
        f"Train: {len(train_df)} bars  |  "
        f"{train_df.index[0].date()} -> {train_df.index[-1].date()}"
    )
    print(
        f"Test:  {len(test_df)} bars  |  "
        f"{test_df.index[0].date()} -> {test_df.index[-1].date()}"
    )

    model = GATPortfolioNet(n_stocks=N_STOCKS_PER_ROUND).to(DEVICE)
    state = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"Loaded weights from {SAVE_PATH}")

    gat_result = backtest_gat_50(model, weekly_df, TICKERS, start_idx=split_idx)
    print(
        f"\nGAT backtest (50 stocks): {len(gat_result)} steps  |  "
        f"{gat_result.index[0].date()} -> {gat_result.index[-1].date()}"
    )

    ew_result = backtest_equal_weight_rebalanced(weekly_df, TICKERS, start_idx=split_idx)
    print(
        f"Equal-Weight Rebal.: {len(ew_result)} steps  |  "
        f"{ew_result.index[0].date()} -> {ew_result.index[-1].date()}"
    )

    bh_result = backtest_buy_and_hold(weekly_df, TICKERS, start_idx=split_idx)
    print(
        f"Buy & Hold: {len(bh_result)} steps  |  "
        f"{bh_result.index[0].date()} -> {bh_result.index[-1].date()}"
    )

    shares_out = get_shares_outstanding(TICKERS)
    cap_result = backtest_cap_weighted(weekly_df, TICKERS, shares_out, start_idx=split_idx)
    print(
        f"\nApprox. Cap-Weighted: {len(cap_result)} steps  |  "
        f"{cap_result.index[0].date()} -> {cap_result.index[-1].date()}"
    )

    spy_result = fetch_spy_returns(gat_result.index)
    print(
        f"SPY reference: {len(spy_result)} steps  |  "
        f"{spy_result.index[0].date()} -> {spy_result.index[-1].date()}"
    )

    gross_metrics = [
        compute_metrics(gat_result, "GAT Portfolio"),
        compute_metrics(ew_result, "EW Rebalanced"),
        compute_metrics(bh_result, "Buy & Hold (1/N)"),
        compute_metrics(cap_result, "Approx. Cap-Weighted"),
        compute_metrics(spy_result, "S&P 500 (SPY) [ref]"),
    ]
    gross_df = pd.DataFrame(gross_metrics).set_index("Strategy")
    print("\n" + "=" * 72)
    print("OUT-OF-SAMPLE COMPARISON — GROSS (no transaction costs)")
    print("=" * 72)
    print(gross_df.to_string())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gat_result.index, gat_result["cumulative"], label="GAT Portfolio", linewidth=2.2, color="#2563eb")
    ax.plot(ew_result.index, ew_result["cumulative"], label="EW Rebalanced", linewidth=1.4, linestyle="--", color="#16a34a")
    ax.plot(bh_result.index, bh_result["cumulative"], label="Buy & Hold (1/N)", linewidth=1.4, linestyle="--", color="#9333ea")
    ax.plot(spy_result.index, spy_result["cumulative"], label="S&P 500 (ref)", linewidth=1.2, linestyle=":", color="grey")
    ax.set_ylabel("Cumulative Wealth ($1 invested)")
    ax.set_title(f"Out-of-Sample Performance (Gross)\n{gat_result.index[0].date()} -> {gat_result.index[-1].date()}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="grey", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(GROSS_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    tx_rate = TX_COST_BPS / 10_000
    gat_weights_arr = gat_result[TICKERS].values
    gat_turnover = np.concatenate([[0.0], np.abs(np.diff(gat_weights_arr, axis=0)).sum(axis=1)])
    cap_turnover = cap_result["turnover"].values
    ew_turnover = ew_result["turnover"].values

    gat_net = gat_result.copy()
    gat_net["turnover"] = gat_turnover
    gat_net["net_ret"] = gat_net["portfolio_ret"] - gat_turnover * tx_rate
    gat_net["cumulative_net"] = (1 + gat_net["net_ret"]).cumprod()

    cap_net = cap_result.copy()
    cap_net["net_ret"] = cap_net["portfolio_ret"] - cap_turnover * tx_rate
    cap_net["cumulative_net"] = (1 + cap_net["net_ret"]).cumprod()

    ew_net = ew_result.copy()
    ew_net["net_ret"] = ew_net["portfolio_ret"] - ew_turnover * tx_rate
    ew_net["cumulative_net"] = (1 + ew_net["net_ret"]).cumprod()

    bh_net = bh_result.copy()
    bh_net["net_ret"] = bh_net["portfolio_ret"]
    bh_net["cumulative_net"] = bh_net["cumulative"]

    spy_net = spy_result.copy()
    spy_net["net_ret"] = spy_net["portfolio_ret"]
    spy_net["cumulative_net"] = spy_net["cumulative"]

    print("Average weekly turnover:")
    print(f"  GAT Portfolio:             {gat_turnover.mean() * 100:.1f}%")
    print(f"  Approx. Cap-Weighted:      {cap_turnover.mean() * 100:.1f}%")
    print(f"  EW Rebalanced:             {ew_turnover.mean() * 100:.1f}%")
    print("  Buy & Hold:                0.0%")
    print("  SPY:                       0.0%")

    net_metrics = [
        metrics_net(gat_net["net_ret"], gat_net["cumulative_net"], "GAT Portfolio"),
        metrics_net(ew_net["net_ret"], ew_net["cumulative_net"], "EW Rebalanced"),
        metrics_net(bh_net["net_ret"], bh_net["cumulative_net"], "Buy & Hold (1/N)"),
        metrics_net(cap_net["net_ret"], cap_net["cumulative_net"], "Approx. Cap-Weighted"),
        metrics_net(spy_net["net_ret"], spy_net["cumulative_net"], "S&P 500 (SPY) [ref]"),
    ]
    net_df = pd.DataFrame(net_metrics).set_index("Strategy")
    print("\n" + "=" * 72)
    print(f"OUT-OF-SAMPLE COMPARISON — NET ({TX_COST_BPS}bps transaction costs)")
    print("=" * 72)
    print(net_df.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = {"GAT": "#2563eb", "Cap": "#dc2626", "EW": "#16a34a", "BH": "#9333ea", "SPY": "grey"}

    for ax, title, gat_c, ew_c, bh_c, cap_c, spy_c in [
        (axes[0], "Gross Returns (no costs)", gat_result["cumulative"], ew_result["cumulative"], bh_result["cumulative"], cap_result["cumulative"], spy_result["cumulative"]),
        (axes[1], f"Net Returns ({TX_COST_BPS}bps costs)", gat_net["cumulative_net"], ew_net["cumulative_net"], bh_net["cumulative_net"], cap_net["cumulative_net"], spy_net["cumulative_net"]),
    ]:
        ax.plot(gat_result.index, gat_c, label="GAT", lw=2.2, color=colors["GAT"])
        ax.plot(cap_result.index, cap_c, label="Approx. Cap-Weighted", lw=1.8, color=colors["Cap"])
        ax.plot(ew_result.index, ew_c, label="EW Rebal.", lw=1.4, ls="--", color=colors["EW"])
        ax.plot(bh_result.index, bh_c, label="Buy & Hold", lw=1.4, ls="--", color=colors["BH"])
        ax.plot(spy_result.index, spy_c, label="SPY (ref)", lw=1.2, ls=":", color=colors["SPY"])
        ax.set_title(title)
        ax.set_ylabel("Cumulative Wealth")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(NET_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    sector_colors = (
        ['#1d4ed8'] * 10 +
        ['#dc2626'] * 10 +
        ['#16a34a'] * 10 +
        ['#f59e0b'] * 10 +
        ['#7c3aed'] * 10
    )

    ax = axes[0]
    w_df = gat_result[TICKERS]
    ax.stackplot(w_df.index, *[w_df[t] for t in TICKERS], labels=TICKERS, colors=sector_colors, alpha=0.85)
    ax.set_ylabel("Weight")
    ax.set_title("GAT Portfolio Weights — All 50 Stocks (Out-of-Sample)")
    ax.legend(loc="upper left", ncol=10, fontsize=6)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cw_df = cap_result[TICKERS]
    ax.stackplot(cw_df.index, *[cw_df[t] for t in TICKERS], labels=TICKERS, colors=sector_colors, alpha=0.85)
    ax.set_ylabel("Weight")
    ax.set_title("Approx. Market-Cap Weights — All 50 Stocks")
    ax.legend(loc="upper left", ncol=10, fontsize=6)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(WEIGHTS_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    comparison = pd.DataFrame({
        "gat_gross_ret": gat_result["portfolio_ret"],
        "gat_net_ret": gat_net["net_ret"],
        "gat_turnover": gat_net["turnover"],
        "ew_gross_ret": ew_result["portfolio_ret"],
        "ew_net_ret": ew_net["net_ret"],
        "ew_turnover": ew_result["turnover"],
        "cap_gross_ret": cap_result["portfolio_ret"],
        "cap_net_ret": cap_net["net_ret"],
        "cap_turnover": cap_result["turnover"],
        "bh_ret": bh_result["portfolio_ret"],
        "spy_ret": spy_result["portfolio_ret"],
        "gat_gross_cum": gat_result["cumulative"],
        "gat_net_cum": gat_net["cumulative_net"],
        "ew_gross_cum": ew_result["cumulative"],
        "ew_net_cum": ew_net["cumulative_net"],
        "cap_gross_cum": cap_result["cumulative"],
        "cap_net_cum": cap_net["cumulative_net"],
        "bh_cum": bh_result["cumulative"],
        "spy_cum": spy_result["cumulative"],
    }, index=gat_result.index)
    for ticker in TICKERS:
        comparison[f"w_{ticker}"] = gat_result[ticker]

    comparison.to_csv(RESULTS_CSV)
    print(f"\nSaved results to {RESULTS_CSV}")
    print(f"Saved gross plot to {GROSS_PLOT}")
    print(f"Saved gross/net plot to {NET_PLOT}")
    print(f"Saved weights plot to {WEIGHTS_PLOT}")


if __name__ == "__main__":
    main()
