# Graph Attention Portfolio Optimizer

A **Graph Neural Network--based portfolio optimization system** that
learns relationships between stocks and generates **optimal portfolio
allocations**.

This project includes:

-   📊 **Portfolio Rebalancer GUI** for generating portfolio weights
-   🧠 **Graph Attention Network (GAT) model** for asset allocation
-   📈 **Training & backtesting pipeline**

The model combines **temporal stock features** with **graph-based
relationships between assets** to produce diversified portfolio weights.

------------------------------------------------------------------------

# Project Structure

    repo_root/

    InferenceApplication/
    │
    ├── full_model.pth                # Trained model checkpoint
    ├── model.py                      # Model architecture
    ├── portfolio_rebalancer.py      # GUI application
    ├── portfolio_data.json          # Portfolio history storage
    └── sp100_tickers.csv            # Example stock universe

    training and backtesting/
    └── 50stocks/
        ├── training50.py            # Model training script
        ├── backtesting50.py         # Backtesting script
        └── downloadstocks50.py      # Historical data downloader

    notebooks/
    └── GAT_Portfolio_Comparison_Results.ipynb

------------------------------------------------------------------------

# Installation

## 1. Clone the Repository

``` bash
git clone https://github.com/FarazKhanTcd/protraderbot.git
cd protraderbot
```

------------------------------------------------------------------------

## 2. Create a Virtual Environment

### Linux / Mac

``` bash
python -m venv venv
source venv/bin/activate
```

### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

------------------------------------------------------------------------

## 3. Install Dependencies

``` bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pandas numpy yfinance scikit-learn matplotlib tqdm
```

If `torch-geometric` installation fails, follow the official guide:

https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

------------------------------------------------------------------------

# Using the Portfolio Rebalancer Application

The GUI allows users to **select stocks and generate portfolio
allocations using the trained model**.

The system:

1.  Downloads recent market data
2.  Builds feature tensors
3.  Constructs the correlation graph
4.  Runs the trained GAT model
5.  Produces optimized portfolio weights

------------------------------------------------------------------------

## Step 1: Run the Application

Navigate to the application folder:

``` bash
cd InferenceApplication
```

Run the GUI:

``` bash
python portfolio_rebalancer.py
```

------------------------------------------------------------------------

## Step 2: Load Stock Universe

Click:

**Load Stock Universe**

Select a CSV containing stock tickers.

Example file:

    sp100_tickers.csv

Expected format:

    Ticker
    AAPL
    MSFT
    NVDA
    AMZN

------------------------------------------------------------------------

## Step 3: Load the Model

Click:

**Load Model (.pth)**

Select:

    full_model.pth

------------------------------------------------------------------------

## Step 4: Select Stocks

Choose **exactly 10 stocks** from the universe.

Actions:

-   Double‑click a stock to add it
-   Click **Add →**
-   Remove stocks with **← Remove**

Selected stocks appear in the **Selected panel**.

------------------------------------------------------------------------

## Step 5: Run Inference

Click:

**Run Inference**

The application will:

-   Fetch recent market data from Yahoo Finance
-   Convert daily data into weekly bars
-   Build model feature tensors
-   Construct the correlation graph
-   Run the portfolio model
-   Generate allocation weights

------------------------------------------------------------------------

## Step 6: View Recommended Portfolio

Example output:

  Ticker   Weight
  -------- --------
  NVDA     18.4%
  MSFT     16.1%
  AAPL     13.7%
  GOOGL    12.9%
  META     10.2%
  JPM      8.1%
  XOM      7.3%
  PG       5.8%
  COST     4.3%
  HD       3.2%

------------------------------------------------------------------------

## Step 7: Export Portfolio

Click:

**Save Weights CSV**

This will:

-   export the portfolio weights
-   record the run in `portfolio_data.json`

------------------------------------------------------------------------

# Portfolio History

Each run stores:

-   timestamp
-   selected stocks
-   portfolio weights

The model can use previous runs to build **portfolio memory**, allowing
it to consider historical portfolio performance.

------------------------------------------------------------------------

# Training the Model

The repository includes scripts for **training the portfolio model from
scratch** using historical market data.

Training uses **50 large-cap stocks across multiple sectors**.

------------------------------------------------------------------------

## Step 1: Download Historical Data

Navigate to the training directory:

``` bash
cd "training and backtesting/50stocks"
```

Run:

``` bash
python downloadstocks50.py
```

This downloads OHLCV data for the training universe.

Example files:

    AAPL.csv
    MSFT.csv
    NVDA.csv
    ...

------------------------------------------------------------------------

## Step 2: Train the Model

Run the training script:

``` bash
python training50.py
```

The script performs:

1.  Data loading and cleaning
2.  Weekly resampling of stock data
3.  Feature engineering
4.  Graph construction based on correlations
5.  Model training using portfolio performance metrics

The trained model will be saved as:

    gat_portfolio_weights50.pth

------------------------------------------------------------------------

# Training Objective

The model is optimized using a **Sharpe ratio--based loss function**.

    Loss = -SharpeRatio + λ × DiversificationPenalty

Diversification is enforced using the **Herfindahl‑Hirschman Index
(HHI)**.

------------------------------------------------------------------------

# Feature Engineering

Each stock uses a **52‑week feature window** containing:

1.  Weekly return
2.  4‑week momentum
3.  Rolling volatility
4.  High‑low price ratio
5.  Volume z‑score
6.  Drawdown from peak

Feature tensor shape:

    (n_stocks, lookback, features)

    Example:
    (10, 52, 6)

------------------------------------------------------------------------

# Graph Construction

Stocks are connected through **correlation graphs**.

For each stock:

-   compute correlations with all other assets
-   keep the **top‑k strongest correlations**

This graph structure is used by the **Graph Attention Network**.

------------------------------------------------------------------------

# Model Architecture

### Temporal Encoder

Encodes each stock's historical features using:

    BiGRU + Attention Pooling

### Graph Attention Network

Captures cross‑stock relationships using:

    GATConv layers

### Portfolio Memory

Maintains recent `(weights, returns)` pairs so the model can adapt based
on past performance.

------------------------------------------------------------------------

# Backtesting

To evaluate model performance:

``` bash
python backtesting50.py
```

This simulates **weekly portfolio rebalancing** over historical data.

------------------------------------------------------------------------

# Notebook Experiments

The notebook

    GAT_Portfolio_Comparison_Results.ipynb

contains experiments comparing the GAT model to baseline portfolio
strategies.

------------------------------------------------------------------------
