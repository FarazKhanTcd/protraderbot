import numpy as np
import pandas as pd
from pathlib import Path

def print_direction_stats(results: pd.DataFrame) -> None:
    tp = ((results["pred_direction"] == 1) & (results["actual_direction"] == 1)).sum()
    tn = ((results["pred_direction"] == 0) & (results["actual_direction"] == 0)).sum()
    fp = ((results["pred_direction"] == 1) & (results["actual_direction"] == 0)).sum()
    fn = ((results["pred_direction"] == 0) & (results["actual_direction"] == 1)).sum()

    print("\nDirection statistics:")
    print(f"TP (up/up):   {tp}")
    print(f"TN (down/down): {tn}")
    print(f"FP (up/down): {fp}")
    print(f"FN (down/up): {fn}")
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]

    # Keep only the columns we need
    needed_cols = ["Date", "Close"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found columns: {list(df.columns)}")

    df = df[["Date", "Close"]].copy()

    # Remove rows where Date is not a real date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # Convert Close to numeric, removing bad rows like the 'AAPL' row
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).copy()

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def polynomial_direction_prediction(
    prices: pd.Series,
    window: int = 30,
    degree: int = 2,
    horizon: int = 7
) -> pd.DataFrame:
    results = []
    values = prices.to_numpy(dtype=float)

    for t in range(window - 1, len(values) - horizon):
        past_window = values[t - window + 1:t + 1]

        x = np.arange(window)
        y = past_window

        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)

        future_x = (window - 1) + horizon
        predicted_price = float(poly(future_x))

        current_price = float(values[t])
        actual_future_price = float(values[t + horizon])

        pred_direction = 1 if predicted_price > current_price else 0
        actual_direction = 1 if actual_future_price > current_price else 0

        results.append({
            "index_t": t,
            "current_price": current_price,
            "predicted_price_t_plus_7": predicted_price,
            "actual_price_t_plus_7": actual_future_price,
            "pred_direction": pred_direction,
            "actual_direction": actual_direction,
        })

    return pd.DataFrame(results)


def evaluate_accuracy(results: pd.DataFrame) -> float:
    if results.empty:
        return 0.0
    return (results["pred_direction"] == results["actual_direction"]).mean()


if __name__ == "__main__":
    csv_path = "../data/raw-stocks/GOOGL.csv"

    df = load_data(csv_path)

    results = polynomial_direction_prediction(
        prices=df["Close"],
        window=30,
        degree=2,
        horizon=7
    )

    accuracy = evaluate_accuracy(results)

    print("Cleaned data:")
    print(df.head())
    print("\nPrediction results:")
    print(results.head())
    print(f"\nDirectional accuracy: {accuracy:.4f}")

    output_path = "poly_prediction_results_AAPL.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    print_direction_stats(results)