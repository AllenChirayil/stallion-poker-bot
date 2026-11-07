#!/usr/bin/env python3
"""
train_preflop_model.py - Train an offline neural regression model for preflop hand equity.

Generates thousands of Short Deck 5-Card Omaha hands, simulates ground-truth
showdown equity against random opponents and boards using omaha_score,
trains a compact 2-layer MLP, and exports the weights to preflop_model.json.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
import numpy as np

from evaluator import warmup, omaha_score, NUM_CARDS

def extract_features(hole: tuple[int, ...]) -> list[float]:
    """Extract 22 numerical features for a 5-card preflop hand."""
    ranks = sorted([c >> 2 for c in hole], reverse=True)
    suits = [c & 3 for c in hole]
    
    rank_counts = Counter(ranks)
    pairs = sum(1 for c in rank_counts.values() if c == 2)
    trips = sum(1 for c in rank_counts.values() if c == 3)
    quads = sum(1 for c in rank_counts.values() if c >= 4)
    
    suit_counts = sorted(Counter(suits).values(), reverse=True)
    while len(suit_counts) < 4:
        suit_counts.append(0)
    
    suits_with_ge_2 = sum(1 for c in suit_counts if c >= 2)
    is_double_suited = 1.0 if suits_with_ge_2 >= 2 else 0.0
    
    suit_to_cards = {}
    for c in hole:
        s = c & 3
        suit_to_cards.setdefault(s, []).append(c >> 2)
    
    has_nut_flush_draw = 0.0
    has_second_nut_flush_draw = 0.0
    for s, s_ranks in suit_to_cards.items():
        if len(s_ranks) >= 2:
            if 8 in s_ranks: # Ace
                has_nut_flush_draw = 1.0
            elif 7 in s_ranks: # King
                has_second_nut_flush_draw = 1.0
                
    distinct_ranks = sorted(set(ranks))
    conn_adjacent = 0
    conn_onegap = 0
    for i in range(len(distinct_ranks) - 1):
        gap = distinct_ranks[i+1] - distinct_ranks[i]
        if gap == 1:
            conn_adjacent += 1
        elif gap == 2:
            conn_onegap += 1
    wheel_wrap = 1.0 if (8 in distinct_ranks and 0 in distinct_ranks) else 0.0
    
    aces = ranks.count(8)
    kings = ranks.count(7)
    queens = ranks.count(6)
    jacks = ranks.count(5)
    
    avg_rank = sum(ranks) / 5.0
    
    return [
        avg_rank / 8.0,
        aces / 4.0, kings / 4.0, queens / 4.0, jacks / 4.0,
        pairs / 2.0, float(trips), float(quads),
        suit_counts[0] / 5.0, suit_counts[1] / 5.0,
        is_double_suited,
        has_nut_flush_draw,
        has_second_nut_flush_draw,
        conn_adjacent / 4.0,
        conn_onegap / 4.0,
        wheel_wrap,
        len(distinct_ranks) / 5.0,
        ranks[0] / 8.0, ranks[1] / 8.0, ranks[2] / 8.0, ranks[3] / 8.0, ranks[4] / 8.0
    ]

def main():
    print("Initializing evaluator...")
    warmup()
    rng = random.Random(42)
    np.random.seed(42)

    NUM_SAMPLES = 8000
    ROLLOUTS_PER_SAMPLE = 100

    print(f"Generating {NUM_SAMPLES} hands with {ROLLOUTS_PER_SAMPLE} simulations each...")
    t0 = time.time()
    
    X = []
    y = []
    all_cards = list(range(NUM_CARDS))

    for idx in range(NUM_SAMPLES):
        hole = tuple(sorted(rng.sample(all_cards, 5)))
        feat = extract_features(hole)
        
        used = set(hole)
        rem = [c for c in range(NUM_CARDS) if c not in used]
        wins = 0
        ties = 0
        for _ in range(ROLLOUTS_PER_SAMPLE):
            s = rng.sample(rem, 10)
            b = tuple(s[:5])
            opp = tuple(s[5:])
            s_my = omaha_score(hole, b)
            s_opp = omaha_score(opp, b)
            if s_my > s_opp:
                wins += 1
            elif s_my == s_opp:
                ties += 1
        equity = (wins + ties * 0.5) / float(ROLLOUTS_PER_SAMPLE)
        X.append(feat)
        y.append(equity)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    print(f"Data generation complete in {time.time()-t0:.2f}s!")

    # Train 2-layer MLP (22 -> 32 -> 1)
    input_dim = X.shape[1]
    hidden_dim = 32
    output_dim = 1

    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
    b2 = np.array([0.5], dtype=np.float32)

    mW1 = np.zeros_like(W1)
    vW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1)
    vb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2)
    vW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2)
    vb2 = np.zeros_like(b2)

    lr = 0.003
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    batch_size = 64
    epochs = 150
    n = len(X)

    print("Training neural network...")
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx]
            yb = y[idx]

            # Forward
            z1 = xb @ W1 + b1
            h1 = np.maximum(0.0, z1)
            y_pred = h1 @ W2 + b2

            # Loss grad
            grad_out = (y_pred - yb) * (2.0 / len(xb))

            # Backprop
            dW2 = h1.T @ grad_out
            db2 = np.sum(grad_out, axis=0)

            dh1 = grad_out @ W2.T
            dh1[h1 <= 0.0] = 0.0
            dW1 = xb.T @ dh1
            db1 = np.sum(dh1, axis=0)

            # Adam updates
            mW1 = beta1 * mW1 + (1 - beta1) * dW1
            vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
            W1 -= lr * mW1 / (np.sqrt(vW1) + eps)

            mb1 = beta1 * mb1 + (1 - beta1) * db1
            vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
            b1 -= lr * mb1 / (np.sqrt(vb1) + eps)

            mW2 = beta1 * mW2 + (1 - beta1) * dW2
            vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
            W2 -= lr * mW2 / (np.sqrt(vW2) + eps)

            mb2 = beta1 * mb2 + (1 - beta1) * db2
            vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)
            b2 -= lr * mb2 / (np.sqrt(vb2) + eps)

    # Final evaluation
    z1 = X @ W1 + b1
    h1 = np.maximum(0.0, z1)
    preds = (h1 @ W2 + b2).flatten()
    actuals = y.flatten()

    corr = float(np.corrcoef(actuals, preds)[0, 1])
    mae = float(np.mean(np.abs(actuals - preds)))
    print(f"Training finished!")
    print(f"Pearson Correlation: {corr:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    # Export to JSON
    model_data = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.flatten().tolist(),
        "b2": float(b2[0]),
    }
    with open("preflop_model.json", "w") as f:
        json.dump(model_data, f)
    print("Saved preflop model weights to preflop_model.json")

if __name__ == "__main__":
    main()
