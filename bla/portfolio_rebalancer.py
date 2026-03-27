"""
Portfolio Rebalancer GUI
========================
A Tkinter application that:
  - Loads a universe of stocks from a CSV / Excel file
  - Lets the user pick exactly 10 tickers
  - Fetches features and runs inference with a .pkl model
  - Saves the last selection and a full history of rebalancing runs

Drop-in points are clearly marked with  >>>  DROP-IN  <<<
"""

import os
import json
import pickle
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
# ── Model definition (keep model.py in the same folder) ──────────────────
try:
    from model import (GATPortfolioNet, StockTemporalEncoder,
                       PortfolioMemory, project_weights)
    _MODEL_DEF_LOADED = True
except ImportError:
    _MODEL_DEF_LOADED = False  # user will be prompted to browse for model.py
# ─────────────────────────────────────────────────────────────────────────────
# >>>  DROP-IN 1  <<<
#
# Replace this function with your real feature-construction logic.
# Inputs:
#   tickers  : list[str]  – e.g. ["AAPL", "MSFT", ...]
#   start    : str        – ISO date, e.g. "2023-01-01"
#   end      : str        – ISO date, e.g. "2024-01-01"
# Returns:
#   pd.DataFrame with one row per ticker and one column per feature,
#   in the same order / shape the model expects.
# ─────────────────────────────────────────────────────────────────────────────
def build_features(tickers, start, end, lookback=52, top_k=3):
    """
    Full pipeline:
      - Fetch OHLCV from Yahoo Finance
      - Convert to weekly
      - Build feature tensor (n_stocks, lookback, 6)
      - Build correlation graph (edge_index, edge_weight)

    Returns:
        X          : torch.Tensor (n_stocks, lookback, 6)
        edge_index : torch.LongTensor (2, E)
        edge_weight: torch.FloatTensor (E,)
    """
    

    # --- Normalize tickers for Yahoo ---
    tickers = [t.replace(".", "-") for t in tickers]

    # --- Fetch daily data ---
    data = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    # --- Build MultiIndex DataFrame like training ---
    frames = {}
    for t in tickers:
        df = data[t].copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            frames[(col, t)] = df[col]

    daily_df = pd.DataFrame(frames)
    daily_df.columns = pd.MultiIndex.from_tuples(daily_df.columns)
    daily_df = daily_df.ffill().dropna()

    # --- Resample to weekly (Friday) ---
    resampled = {}
    for t in tickers:
        ohlcv = daily_df.xs(t, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]]
        rs = ohlcv.resample("W-FRI").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            resampled[(col, t)] = rs[col]

    weekly_df = pd.DataFrame(resampled)
    weekly_df.columns = pd.MultiIndex.from_tuples(weekly_df.columns)
    weekly_df = weekly_df.ffill().dropna()

    # --- Check enough data ---
    if len(weekly_df) < lookback + 5:
        raise ValueError("Not enough data for lookback window")

    end_idx = len(weekly_df)

    # =========================
    # FEATURE TENSOR
    # =========================
    closes  = weekly_df["Close"][tickers].iloc[end_idx - lookback - 4 : end_idx]
    volumes = weekly_df["Volume"][tickers].iloc[end_idx - lookback - 4 : end_idx]
    highs   = weekly_df["High"][tickers].iloc[end_idx - lookback - 4 : end_idx]
    lows    = weekly_df["Low"][tickers].iloc[end_idx - lookback - 4 : end_idx]

    feat_list = []

    for t in tickers:
        c = closes[t].values.astype(np.float32)
        v = volumes[t].values.astype(np.float32)
        h = highs[t].values.astype(np.float32)
        l = lows[t].values.astype(np.float32)

        weekly_ret = np.diff(c) / (c[:-1] + 1e-8)
        mom_4w     = (c[4:] - c[:-4]) / (c[:-4] + 1e-8)

        weekly_ret = weekly_ret[-lookback:]
        mom_4w     = mom_4w[-lookback:]

        vol = np.array([
            weekly_ret[max(0, i-3):i+1].std()
            for i in range(len(weekly_ret))
        ])

        hl_ratio = ((h[1:] - l[1:]) / (c[1:] + 1e-8))[-lookback:]

        v_slice  = v[1:][-lookback:]
        v_zscore = (v_slice - v_slice.mean()) / (v_slice.std() + 1e-8)

        c_tail    = c[1:][-lookback:]
        roll_high = np.maximum.accumulate(c_tail)
        drawdown  = (c_tail - roll_high) / (roll_high + 1e-8)

        feats = np.stack(
            [weekly_ret, mom_4w, vol, hl_ratio, v_zscore, drawdown],
            axis=1
        )

        feat_list.append(feats)

    arr = np.stack(feat_list)

    mean = arr.mean(axis=0, keepdims=True)
    std  = np.where(arr.std(axis=0, keepdims=True) < 1e-4, 1.0, arr.std(axis=0, keepdims=True))
    arr  = np.clip(np.nan_to_num((arr - mean) / std), -3.0, 3.0)

    X = torch.tensor(arr, dtype=torch.float32)

    # =========================
    # CORRELATION GRAPH
    # =========================
    returns = weekly_df["Close"][tickers].iloc[end_idx - lookback : end_idx].pct_change().dropna()
    corr = returns.corr().values.astype(np.float32)

    np.fill_diagonal(corr, 0)

    adj = np.zeros_like(corr)

    for i in range(len(tickers)):
        top_k_idx = np.argsort(np.abs(corr[i]))[-top_k:]
        adj[i, top_k_idx] = np.abs(corr[i, top_k_idx])

    edge_index, edge_weight = dense_to_sparse(torch.tensor(adj, dtype=torch.float32))

    return X, edge_index, edge_weight

# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, feature_df: pd.DataFrame) -> dict:
    """
    Run the loaded model and return a dict {ticker: weight}.
    Adjust the call to model.predict / model.transform / etc. as needed.
    """
    # >>>  DROP-IN 2  <<<
    # Replace the lines below with how your model actually produces weights.
    # Example for a sklearn-style model:
    #   raw = model.predict(feature_df.values)
    #
    # For now we call predict and softmax-normalise so weights sum to 1.
    import numpy as np

    try:
        raw = model.predict(feature_df.values)          # shape (n_tickers,)
    except Exception:
        raw = model.predict(feature_df)                  # some models want a df

    raw = np.array(raw, dtype=float)
    # Softmax normalisation → all positive, sum = 1
    e = np.exp(raw - raw.max())
    weights = e / e.sum()
    return {ticker: float(w) for ticker, w in zip(feature_df.index, weights)}


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE = "portfolio_data.json"


def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"last_selection": [], "history": []}


def save_data(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────
class PortfolioApp(tk.Tk):
    MAX_PICKS = 10
    ACCENT   = "#1B6CA8"
    BG       = "#F0F4F8"
    PANEL    = "#FFFFFF"
    BORDER   = "#D1DCE8"
    GREEN    = "#2E7D32"
    RED      = "#C62828"
    TEXT     = "#1A202C"
    SUBTEXT  = "#4A5568"

    def __init__(self):
        super().__init__()
        self.title("Portfolio Rebalancer")
        self.geometry("1200x780")
        self.minsize(960, 640)
        self.configure(bg=self.BG)
        self.model_def_path  = None 
        # State
        self.data        = load_data()
        self.model       = None
        self.all_tickers : list[str] = []
        self.selected    : list[str] = []
        #self.history = []
        self._apply_style()
        self._build_ui()

        # Restore last selection
        if self.data["last_selection"]:
            self._restore_selection(self.data["last_selection"])

    # ── Styles ──────────────────────────────────────────────────────────────
    def _apply_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame",       background=self.BG)
        s.configure("Panel.TFrame", background=self.PANEL,
                    relief="flat", borderwidth=1)
        s.configure("TLabel",       background=self.BG,   foreground=self.TEXT)
        s.configure("Panel.TLabel", background=self.PANEL, foreground=self.TEXT)
        s.configure("Sub.TLabel",   background=self.PANEL, foreground=self.SUBTEXT,
                    font=("Helvetica", 9))
        s.configure("H1.TLabel",    background=self.BG,   foreground=self.TEXT,
                    font=("Helvetica", 18, "bold"))
        s.configure("H2.TLabel",    background=self.PANEL, foreground=self.TEXT,
                    font=("Helvetica", 12, "bold"))
        s.configure("Accent.TButton", font=("Helvetica", 10, "bold"),
                    foreground="white", background=self.ACCENT,
                    padding=(12, 6))
        s.map("Accent.TButton",
              background=[("active", "#145080"), ("disabled", "#A0AEC0")])
        s.configure("TButton", font=("Helvetica", 9), padding=(8, 4))
        s.configure("TEntry",  fieldbackground="white", padding=4)
        s.configure("Treeview",
                    background=self.PANEL, fieldbackground=self.PANEL,
                    foreground=self.TEXT, rowheight=24,
                    font=("Helvetica", 10))
        s.configure("Treeview.Heading",
                    background=self.BG, foreground=self.SUBTEXT,
                    font=("Helvetica", 9, "bold"))
        s.map("Treeview", background=[("selected", self.ACCENT)],
              foreground=[("selected", "white")])

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        hdr = ttk.Frame(self, padding=(20, 14, 20, 10))
        hdr.pack(fill="x")

        ttk.Label(hdr, text="Portfolio Rebalancer", style="H1.TLabel").pack(side="left")

        hdr_btns = ttk.Frame(hdr)
        hdr_btns.pack(side="right")

        ttk.Button(hdr_btns, text="📂  Load Stock Universe",
                   command=self._load_universe).pack(side="left", padx=4)
        ttk.Button(hdr_btns, text="📂  Load Model (.pth)",
                   command=self._load_model).pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Load a stock universe and model to begin.")
        ttk.Label(hdr, textvariable=self.status_var,
                  style="Sub.TLabel",
                  background=self.BG).pack(side="bottom", anchor="w")

        sep = tk.Frame(self, height=1, bg=self.BORDER)
        sep.pack(fill="x")

        # ── Main three-column layout ────────────────────────────────────────
        body = ttk.Frame(self, padding=16)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=2, minsize=280)
        body.columnconfigure(1, weight=1, minsize=220)
        body.columnconfigure(2, weight=2, minsize=280)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1, minsize=180)

        # Col 0 – Stock universe
        self._build_universe_panel(body)
        # Col 1 – Selected stocks
        self._build_selection_panel(body)
        # Col 2 – Weights result
        self._build_weights_panel(body)
        # Bottom – History
        self._build_history_panel(body)

    # ── Universe panel (col 0) ───────────────────────────────────────────────
    def _build_universe_panel(self, parent):
        frm = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        frm.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        frm.rowconfigure(2, weight=1)
        frm.columnconfigure(0, weight=1)

        ttk.Label(frm, text="Stock Universe", style="H2.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6))

        # Search
        sf = ttk.Frame(frm, style="Panel.TFrame")
        sf.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        sf.columnconfigure(0, weight=1)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._filter_universe)
        ttk.Entry(sf, textvariable=self.search_var).grid(
            row=0, column=0, sticky="ew")
        ttk.Label(sf, text="🔍", style="Panel.TLabel").grid(
            row=0, column=1, padx=(4, 0))

        # Listbox with scrollbar
        lf = ttk.Frame(frm, style="Panel.TFrame")
        lf.grid(row=2, column=0, sticky="nsew")
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        self.universe_lb = tk.Listbox(
            lf, selectmode="extended",
            font=("Helvetica", 10),
            bg=self.PANEL, fg=self.TEXT,
            selectbackground=self.ACCENT, selectforeground="white",
            activestyle="none", bd=0, highlightthickness=1,
            highlightbackground=self.BORDER,
            relief="flat", exportselection=False)
        self.universe_lb.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lf, command=self.universe_lb.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.universe_lb.configure(yscrollcommand=sb.set)
        self.universe_lb.bind("<Double-Button-1>", self._add_selected_from_lb)
        self.universe_lb.bind("<Return>", self._add_selected_from_lb)

        # Counter + add button
        bf = ttk.Frame(frm, style="Panel.TFrame")
        bf.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self.universe_count_var = tk.StringVar(value="0 stocks loaded")
        ttk.Label(bf, textvariable=self.universe_count_var,
                  style="Sub.TLabel").pack(side="left")
        ttk.Button(bf, text="Add →",
                   command=self._add_selected_from_lb).pack(side="right")

    # ── Selection panel (col 1) ──────────────────────────────────────────────
    def _build_selection_panel(self, parent):
        frm = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        frm.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=(0, 8))
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        hdr = ttk.Frame(frm, style="Panel.TFrame")
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(hdr, text="Selected  (0 / 10)",
                  style="H2.TLabel").pack(side="left")
        self.pick_label = hdr.winfo_children()[-1]   # keep ref

        lf = ttk.Frame(frm, style="Panel.TFrame")
        lf.grid(row=1, column=0, sticky="nsew")
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        self.pick_lb = tk.Listbox(
            lf, selectmode="extended",
            font=("Helvetica", 10, "bold"),
            bg=self.PANEL, fg=self.TEXT,
            selectbackground=self.RED, selectforeground="white",
            activestyle="none", bd=0, highlightthickness=1,
            highlightbackground=self.BORDER,
            relief="flat", exportselection=False)
        self.pick_lb.grid(row=0, column=0, sticky="nsew")
        sb2 = ttk.Scrollbar(lf, command=self.pick_lb.yview)
        sb2.grid(row=0, column=1, sticky="ns")
        self.pick_lb.configure(yscrollcommand=sb2.set)
        self.pick_lb.bind("<Double-Button-1>", self._remove_from_picks)
        self.pick_lb.bind("<Delete>", self._remove_from_picks)

        bf = ttk.Frame(frm, style="Panel.TFrame")
        bf.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(bf, text="← Remove",
                   command=self._remove_from_picks).pack(side="left")
        ttk.Button(bf, text="Clear All",
                   command=self._clear_picks).pack(side="right")

        # Run button
        self.run_btn = ttk.Button(
            frm, text="▶  Run Inference",
            style="Accent.TButton",
            command=self._run_inference,
            state="disabled")
        self.run_btn.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(frm,
                  text="Double-click to add/remove",
                  style="Sub.TLabel").grid(row=4, column=0, pady=(4, 0))

    # ── Weights panel (col 2) ─────────────────────────────────────────────────
    def _build_weights_panel(self, parent):
        frm = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        frm.grid(row=0, column=2, sticky="nsew", pady=(0, 8))
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        ttk.Label(frm, text="Recommended Weights",
                  style="H2.TLabel").grid(row=0, column=0, sticky="w",
                                          pady=(0, 6))

        cols = ("Ticker", "Weight %", "Bar")
        self.weights_tree = ttk.Treeview(frm, columns=cols,
                                         show="headings", height=12)
        for col, w in zip(cols, (80, 80, 200)):
            self.weights_tree.heading(col, text=col)
            self.weights_tree.column(col, width=w, anchor="center"
                                     if col != "Bar" else "w")
        self.weights_tree.grid(row=1, column=0, sticky="nsew")
        sb3 = ttk.Scrollbar(frm, command=self.weights_tree.yview)
        sb3.grid(row=1, column=1, sticky="ns")
        self.weights_tree.configure(yscrollcommand=sb3.set)

        bf = ttk.Frame(frm, style="Panel.TFrame")
        bf.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(bf, text="💾  Save Weights CSV",
                   command=self._export_weights_csv).pack(side="right")
        self.run_ts_var = tk.StringVar(value="No run yet.")
        ttk.Label(bf, textvariable=self.run_ts_var,
                  style="Sub.TLabel").pack(side="left")

    # ── History panel (bottom row) ───────────────────────────────────────────
    def _build_history_panel(self, parent):
        frm = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        frm.grid(row=1, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        hdr = ttk.Frame(frm, style="Panel.TFrame")
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(hdr, text="Run History", style="H2.TLabel").pack(side="left")
        ttk.Button(hdr, text="🗑  Clear History",
                   command=self._clear_history).pack(side="right")

        cols = ["Timestamp", "Tickers"] + [f"W{i+1}" for i in range(10)]
        self.hist_tree = ttk.Treeview(frm, columns=cols, show="headings",
                                      height=6)
        self.hist_tree.heading("Timestamp", text="Timestamp")
        self.hist_tree.column("Timestamp", width=140, anchor="center")
        self.hist_tree.heading("Tickers", text="Tickers")
        self.hist_tree.column("Tickers", width=420, anchor="w")
        for i in range(10):
            col = f"W{i+1}"
            self.hist_tree.heading(col, text=col)
            self.hist_tree.column(col, width=68, anchor="center")

        self.hist_tree.grid(row=1, column=0, sticky="nsew")
        sb4 = ttk.Scrollbar(frm, command=self.hist_tree.yview)
        sb4.grid(row=1, column=1, sticky="ns")
        self.hist_tree.configure(yscrollcommand=sb4.set)
        self.hist_tree.bind("<ButtonRelease-1>", self._on_history_select)

        ttk.Label(frm,
                  text="Click a history row to restore that selection.",
                  style="Sub.TLabel").grid(row=2, column=0, sticky="w",
                                           pady=(4, 0))
        self._refresh_history_tree()
    def _load_model_def(self):
        """
        Dynamically import the .py file that defines the model class
        (e.g. GATPortfolioNet).  Must be done BEFORE loading the .pth file
        so PyTorch can find the class during unpickling.
        """
        path = filedialog.askopenfilename(
            title="Select Python file with model class definition",
            filetypes=[("Python file", "*.py"), ("All", "*.*")])
        if not path:
            return
        try:
            self._import_model_def(path)
            self.model_def_path = path
            self._set_status(
                f"Model definition loaded: '{os.path.basename(path)}' — "
                f"now load the .pth file.")
        except Exception as exc:
            messagebox.showerror("Definition Load Error", str(exc))
 
    def _import_model_def(self, path: str):
        """
        Load a .py file as a module and inject its contents into sys.modules
        under both its own name AND '__main__' so torch.load can find any
        class regardless of how it was originally saved.
        """
        module_name = os.path.splitext(os.path.basename(path))[0]
        spec   = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
 
        # Register under its real name
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
 
        # Also copy every public attribute into __main__ so that
        # torch.load(...) can resolve  __main__.GATPortfolioNet  (or any
        # other class the model file defines).
        import __main__
        for attr, val in vars(module).items():
            if not attr.startswith("__"):
                setattr(__main__, attr, val)
 
    # ── Load model ───────────────────────────────────────────────────────────
    def _load_model(self):
        # Ensure model classes are registered in __main__ before unpickling
        if not _MODEL_DEF_LOADED:
            if self.model_def_path and os.path.exists(self.model_def_path):
                try:
                    self._import_model_def(self.model_def_path)
                except Exception:
                    pass
            else:
                messagebox.showwarning(
                    "model.py not found",
                    "model.py was not found next to this script.\n"
                    "Use the 'Load model.py' button to browse for it first.")
                return
 
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint (.pth)",
            filetypes=[("PyTorch checkpoint", "*.pth *.pt"),
                       ("Pickle", "*.pkl *.pickle"),
                       ("All", "*.*")])
        if not path:
            return
        try:
            import torch
            self.model = torch.load(path, map_location="cpu",
                                    weights_only=False)
            self.model.eval()
            self._set_status(
                f"Model loaded: '{os.path.basename(path)}'  "
                f"({type(self.model).__name__})")
            self._update_run_btn()
        except ImportError:
            messagebox.showerror("Missing Dependency",
                                 "PyTorch is not installed.\n"
                                 "Run: pip install torch")
        except Exception as exc:
            messagebox.showerror("Model Load Error", str(exc))
    # ── Load universe ────────────────────────────────────────────────────────
    def _load_universe(self):
        path = filedialog.askopenfilename(
            title="Select Stock Universe File",
            filetypes=[("CSV / Excel", "*.csv *.xlsx *.xls"), ("All", "*.*")])
        if not path:
            return
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            # >>>  DROP-IN 3  <<<
            # Adjust the column name below to match whatever your file uses
            # for tickers  (e.g. "Symbol", "Ticker", "ticker", etc.)
            ticker_col = None
            for candidate in ("Ticker", "ticker", "Symbol", "symbol",
                              "TICKER", "SYMBOL"):
                if candidate in df.columns:
                    ticker_col = candidate
                    break
            if ticker_col is None:
                ticker_col = df.columns[0]   # fall back to first column

            self.all_tickers = (
                df[ticker_col].dropna().astype(str).str.strip().tolist()
            )
            self._refresh_universe_list()
            self.universe_count_var.set(
                f"{len(self.all_tickers)} stocks loaded")
            self._set_status(
                f"Universe loaded: {len(self.all_tickers)} tickers from "
                f"'{os.path.basename(path)}'")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    

    # ── Universe list helpers ────────────────────────────────────────────────
    def _refresh_universe_list(self):
        q = self.search_var.get().upper()
        self.universe_lb.delete(0, "end")
        for t in self.all_tickers:
            if q in t.upper():
                self.universe_lb.insert("end", t)

    def _filter_universe(self, *_):
        self._refresh_universe_list()

    # ── Add / remove picks ───────────────────────────────────────────────────
    def _add_selected_from_lb(self, _event=None):
        idxs = self.universe_lb.curselection()
        for i in idxs:
            ticker = self.universe_lb.get(i)
            if ticker not in self.selected:
                if len(self.selected) >= self.MAX_PICKS:
                    messagebox.showwarning(
                        "Limit Reached",
                        f"You can select at most {self.MAX_PICKS} stocks.")
                    break
                self.selected.append(ticker)
        self._refresh_pick_list()

    def _remove_from_picks(self, _event=None):
        idxs = list(self.pick_lb.curselection())[::-1]
        for i in idxs:
            self.selected.pop(i)
        self._refresh_pick_list()

    def _clear_picks(self):
        self.selected.clear()
        self._refresh_pick_list()

    def _refresh_pick_list(self):
        self.pick_lb.delete(0, "end")
        for t in self.selected:
            self.pick_lb.insert("end", t)
        n = len(self.selected)
        self.pick_label.configure(
            text=f"Selected  ({n} / {self.MAX_PICKS})",
            foreground=self.RED if n == self.MAX_PICKS else self.TEXT)
        self._update_run_btn()
        # Persist selection
        self.data["last_selection"] = list(self.selected)
        save_data(self.data)

    def _restore_selection(self, tickers: list):
        self.selected = list(tickers)
        self._refresh_pick_list()

    def _update_run_btn(self):
        ready = (
            len(self.selected) == self.MAX_PICKS
            and self.model is not None
        )
        self.run_btn.configure(state="normal" if ready else "disabled")

    # ── Inference ────────────────────────────────────────────────────────────
    def build_full_history(self, memory_len=12):
        """
        Builds multi-step history from portfolio_data.json.
    
        Returns:
            history = [(weights_t, returns_t), ...]
            length ≤ memory_len
        """
    
    
        history_data = self.data.get("history", [])
        if len(history_data) < 2:
            return []  # need at least 2 runs to compute returns
    
        # Take last MEMORY_LEN + 1 runs (because returns are between pairs)
        history_data = history_data[-(memory_len + 1):]
    
        history = []
    
        try:
            for i in range(len(history_data) - 1):
                entry_prev = history_data[i]
                entry_next = history_data[i + 1]
    
                tickers_prev = entry_prev["tickers"]
                weights_prev = dict(zip(tickers_prev, entry_prev["weights"]))
    
                # Use CURRENT selection as reference universe
                weights = []
                returns = []
    
                # Fetch price range between the two timestamps
                start = entry_prev["timestamp"][:10]
                end   = entry_next["timestamp"][:10]
    
                tickers = list(set(tickers_prev) | set(self.selected))
    
                data = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    interval="1d",
                    group_by="ticker",
                    progress=False
                )
    
                for t in self.selected:
                    # --- Weight (from previous run) ---
                    w = weights_prev.get(t, 0.0)
                    weights.append(w)
    
                    # --- Return between runs ---
                    try:
                        t_yf = t.replace(".", "-")
                        prices = data[t_yf]["Close"].dropna()
    
                        if len(prices) < 2:
                            r = 0.0
                        else:
                            r = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                    except Exception:
                        r = 0.0
    
                    returns.append(r)
    
                weights_tensor = torch.tensor(weights, dtype=torch.float32)
                returns_tensor = torch.tensor(returns, dtype=torch.float32)
    
                history.append((weights_tensor, returns_tensor))
    
        except Exception as e:
            print("Full history build failed:", e)
            return []
    
        return history
    def _run_inference(self):
        if len(self.selected) != self.MAX_PICKS:
            messagebox.showwarning("Selection",
                                   f"Please select exactly {self.MAX_PICKS} stocks.")
            return
        if self.model is None:
            messagebox.showwarning("Model", "Please load a model first.")
            return

        self._set_status("Fetching data and building features…")
        self.update_idletasks()

        try:
            # >>>  DROP-IN 4  <<<
            # Adjust start / end dates as needed, or let the user choose them.
            end_dt   = datetime.date.today()
            start_dt = end_dt - datetime.timedelta(weeks=60)
            # Update history using last run
            history = self.build_full_history()
            feature_df, edge_index, edge_weight = build_features(
                self.selected,
                start_dt.isoformat(),
                end_dt.isoformat(),
            )

            with torch.no_grad():
                output = self.model(feature_df, edge_index, edge_weight, history=history,)  # adjust if needed

                raw = output.squeeze().cpu().numpy()

                e = np.exp(raw - np.max(raw))
                weights_arr = e / e.sum()

            weights = dict(zip(self.selected, weights_arr))
        except Exception as exc:
            messagebox.showerror("Inference Error", str(exc))
            self._set_status("Inference failed.")
            return
            
    
        self._display_weights(weights)
        
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.run_ts_var.set(f"Last run: {ts}")
        self._set_status("Inference complete.")
        return

    # ── Display weights ───────────────────────────────────────────────────────
    def _display_weights(self, weights: dict):
        for row in self.weights_tree.get_children():
            self.weights_tree.delete(row)

        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        bar_max = sorted_w[0][1] if sorted_w else 1.0
        BAR_CHARS = 20
        for ticker, w in sorted_w:
            pct     = w * 100
            bar_len = int(round((w / bar_max) * BAR_CHARS))
            bar     = "█" * bar_len + "░" * (BAR_CHARS - bar_len)
            self.weights_tree.insert(
                "", "end", values=(ticker, f"{pct:.2f}%", bar))

        self.last_weights = weights   # keep for CSV export

    # ── History helpers ───────────────────────────────────────────────────────
    def _append_history(self, weights: dict):
        ts = datetime.datetime.now().isoformat()
        tickers = list(weights.keys())
        # Convert NumPy floats to standard Python floats
        wvals = [float(round(v, 6)) for v in weights.values()]
        entry = {"timestamp": ts, "tickers": tickers, "weights": wvals}
        self.data["history"].append(entry)
        save_data(self.data)

    def _refresh_history_tree(self):
        for row in self.hist_tree.get_children():
            self.hist_tree.delete(row)
        for entry in reversed(self.data["history"]):
            ts   = entry["timestamp"].replace("T", " ")
            tkrs = "  ".join(entry["tickers"])
            wstr = [f"{w*100:.1f}%" for w in entry["weights"]]
            # pad to 10 cols
            while len(wstr) < 10:
                wstr.append("")
            self.hist_tree.insert("", "end",
                                  values=[ts, tkrs] + wstr,
                                  tags=(json.dumps(entry["tickers"]),))

    def _on_history_select(self, _event):
        sel = self.hist_tree.selection()
        if not sel:
            return
        tags = self.hist_tree.item(sel[0], "tags")
        if tags:
            tickers = json.loads(tags[0])
            if messagebox.askyesno(
                    "Restore Selection",
                    "Load this historical selection into the picker?"):
                self._restore_selection(tickers)

    def _clear_history(self):
        if messagebox.askyesno("Clear History",
                                "Delete all history entries?"):
            self.data["history"] = []
            save_data(self.data)
            self._refresh_history_tree()
    # ── Status bar ────────────────────────────────────────────────────────────
    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    # ── Export ────────────────────────────────────────────────────────────────
    def _export_weights_csv(self):
        if not hasattr(self, "last_weights") or not self.last_weights:
            messagebox.showinfo("No Data", "Run inference first.")
            return
    
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"weights_{datetime.date.today()}.csv"
        )
        if not path:
            return
    
        # Save CSV
        df = pd.DataFrame(list(self.last_weights.items()), columns=["Ticker", "Weight"])
        df.to_csv(path, index=False)
    
        # Append to JSON history using CURRENT weights
        # Ensure the list of weights matches the pick order
        w_dict = {ticker: self.last_weights.get(ticker, 0.0) for ticker in self.selected}
        self._append_history(w_dict)
    
        self._set_status(f"Weights exported to '{os.path.basename(path)}'")
            
        

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PortfolioApp()
    app.mainloop()
