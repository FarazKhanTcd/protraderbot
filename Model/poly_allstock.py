import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]

    required = ["Date", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}")

    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def polynomial_direction_prediction(
    prices: pd.Series,
    window: int = 100,
    degree: int = 2,
    horizon: int = 7
) -> pd.DataFrame:
    values = prices.to_numpy(dtype=float)
    rows = []

    for t in range(window - 1, len(values) - horizon):
        past_window = values[t - window + 1:t + 1]
        x = np.arange(window)

        coeffs = np.polyfit(x, past_window, degree)
        poly = np.poly1d(coeffs)

        current_price = float(values[t])
        predicted_price = float(poly((window - 1) + horizon))
        actual_future_price = float(values[t + horizon])

        pred_direction = 1 if predicted_price > current_price else 0
        actual_direction = 1 if actual_future_price > current_price else 0

        pred_return = (predicted_price - current_price) / current_price
        actual_return = (actual_future_price - current_price) / current_price

        rows.append({
            "current_price": current_price,
            "predicted_price": predicted_price,
            "actual_future_price": actual_future_price,
            "pred_direction": pred_direction,
            "actual_direction": actual_direction,
            "pred_return": pred_return,
            "actual_return": actual_return,
        })

    return pd.DataFrame(rows)


def accuracy_from_results(results: pd.DataFrame) -> float:
    if results.empty:
        return float("nan")
    return (results["pred_direction"] == results["actual_direction"]).mean()


def always_up_accuracy(results: pd.DataFrame) -> float:
    if results.empty:
        return float("nan")
    return (results["actual_direction"] == 1).mean()


def confusion_counts(results: pd.DataFrame) -> dict:
    tp = int(((results["pred_direction"] == 1) & (results["actual_direction"] == 1)).sum())
    tn = int(((results["pred_direction"] == 0) & (results["actual_direction"] == 0)).sum())
    fp = int(((results["pred_direction"] == 1) & (results["actual_direction"] == 0)).sum())
    fn = int(((results["pred_direction"] == 0) & (results["actual_direction"] == 1)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def plot_accuracy_bar(summary_df: pd.DataFrame, output_path: Path) -> None:
    tickers = summary_df["Ticker"]
    x = np.arange(len(tickers))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, summary_df["PolyAcc"], width, label="Polynomial")
    plt.bar(x + width / 2, summary_df["AlwaysUpAcc"], width, label="Always Up")

    plt.xticks(x, tickers, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Polynomial vs Always-Up Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(counts: dict, ticker: str, output_path: Path) -> None:
    matrix = np.array([
        [counts["TN"], counts["FP"]],
        [counts["FN"], counts["TP"]]
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix)

    ax.set_title(f"{ticker} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Down", "Up"])
    ax.set_yticklabels(["Down", "Up"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_return_scatter(results: pd.DataFrame, ticker: str, output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(results["pred_return"], results["actual_return"], alpha=0.35)

    min_val = min(results["pred_return"].min(), results["actual_return"].min())
    max_val = max(results["pred_return"].max(), results["actual_return"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Predicted 7-day return")
    plt.ylabel("Actual 7-day return")
    plt.title(f"{ticker} Predicted vs Actual 7-day Return")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    data_dir = Path("../data/raw-stocks")
    output_dir = Path("../plots")
    output_dir.mkdir(exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir.resolve()}")

    summary_rows = []

    for csv_file in csv_files:
        ticker = csv_file.stem

        try:
            df = load_data(csv_file)
            results = polynomial_direction_prediction(
                prices=df["Close"],
                window=20,
                degree=2,
                horizon=7
            )

            poly_acc = accuracy_from_results(results)
            up_acc = always_up_accuracy(results)
            counts = confusion_counts(results)

            summary_rows.append({
                "Ticker": ticker,
                "Rows": len(df),
                "Predictions": len(results),
                "PolyAcc": poly_acc,
                "AlwaysUpAcc": up_acc,
                "Delta_vs_Up": poly_acc - up_acc,
                "TP": counts["TP"],
                "TN": counts["TN"],
                "FP": counts["FP"],
                "FN": counts["FN"],
            })

            plot_confusion_matrix(
                counts,
                ticker,
                output_dir / f"{ticker}_confusion_matrix.png"
            )

            plot_return_scatter(
                results,
                ticker,
                output_dir / f"{ticker}_return_scatter.png"
            )

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Ticker").reset_index(drop=True)

    print("\nComparison table:\n")
    print(summary_df.round(4))

    summary_df.to_csv("poly_vs_always_up_summary.csv", index=False)

    plot_accuracy_bar(summary_df, output_dir / "accuracy_comparison.png")

    print("\nSaved summary to poly_vs_always_up_summary.csv")
    print(f"Saved plots in {output_dir.resolve()}")


if __name__ == "__main__":
    main()