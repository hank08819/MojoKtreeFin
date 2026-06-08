"""Sklearn brute-force + k-d tree benchmark for all extended datasets."""
import numpy as np, pandas as pd, time, sys, os
from sklearn.neighbors import KNeighborsClassifier

BASE = "/Users/henry_han/LHP2/MOJO/extended_benchmark/data"
N_RUNS = 10; K = 5
FEATURE_COLS = [
    "Open","High","Low","Close","Volume","money","return",
    "MACD","EMA_7","EMA_21","EMA_56","RSI","BB_mid","BB_std","BB_up","BB_low",
    "slowK","slowD","ADX","plus_DI","minus_DI","pseudo_vol","COH","vol_ratio",
]

def bench(clf, Xtr, ytr, Xte, n_runs):
    clf.fit(Xtr, ytr)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pred = clf.predict(Xte)
        times.append(time.perf_counter()-t0)
    return float(np.mean(times)), pred

def run(ticker):
    d = f"{BASE}/{ticker}"
    tr = pd.read_csv(f"{d}/train_data_d24.csv", index_col=0)
    te = pd.read_csv(f"{d}/test_data_d24.csv",  index_col=0)
    mu  = tr[FEATURE_COLS].mean()
    sig = tr[FEATURE_COLS].std().replace(0, 1.0)
    Xtr = np.nan_to_num(((tr[FEATURE_COLS]-mu)/sig).values.astype(np.float32))
    Xte = np.nan_to_num(((te[FEATURE_COLS]-mu)/sig).values.astype(np.float32))
    ytr = tr["label"].values
    yte = te["label"].values
    print(f"\n=== {ticker}  train={len(Xtr):,}  test={len(Xte):,}  d={Xtr.shape[1]} ===")

    bf_t, bf_p = bench(KNeighborsClassifier(n_neighbors=K,algorithm='brute',
                                             metric='euclidean',n_jobs=1),
                       Xtr, ytr, Xte, N_RUNS)
    kd_t, kd_p = bench(KNeighborsClassifier(n_neighbors=K,algorithm='kd_tree',
                                             metric='euclidean',n_jobs=1),
                       Xtr, ytr, Xte, N_RUNS)
    acc = float(np.mean(bf_p == yte))
    print(f"  SK Brute:  {bf_t:.4f}s")
    print(f"  SK KD:     {kd_t:.4f}s")
    print(f"  Accuracy:  {acc:.4f}")
    with open(f"{d}/sklearn_times.txt","w") as f:
        f.write(f"sk_brute={bf_t:.6f}\nsk_kd={kd_t:.6f}\n"
                f"n_train={len(Xtr)}\nn_test={len(Xte)}\nd=24\nacc={acc:.6f}\n")
    return bf_t, kd_t, acc

if __name__ == "__main__":
    tickers = sys.argv[1:] if sys.argv[1:] else ["SPY","QQQ","BTC","EURUSD"]
    results = {}
    for t in tickers:
        results[t] = run(t)
    print("\n=== Summary ===")
    print(f"{'Ticker':<10} {'SK Brute':>10} {'SK KD':>10} {'Acc':>8}")
    for t,(b,k,a) in results.items():
        print(f"{t:<10} {b:>10.4f} {k:>10.4f} {a:>8.4f}")
