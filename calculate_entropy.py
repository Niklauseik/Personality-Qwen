import numpy as np
import pandas as pd

# --- entropy function (log2 → bits) ---
def distribution_entropy(counts, base="bits", epsilon=1e-12):
    counts = np.asarray(counts, dtype=float)
    probs = counts / counts.sum()
    probs = np.where(probs == 0, epsilon, probs)
    log_fn = np.log2 if base == "bits" else np.log
    return -float(np.sum(probs * log_fn(probs)))

# ===== current counts filled from your latest table =====
# Keep the label order noted in comments for each dataset.
datasets = {
    # order: [invalid, mixed, negative, neutral, positive]
    "imdb_sentiment": {
        "base": [1, 18, 5322, 20, 4639],
        "F":    [2, 55, 5146,  6, 4790],
        "T":    [0, 29, 5647, 87, 4237],
    },
    # order: [depression, invalid, normal]
    "mental_sentiment": {
        "base": [27527, 8, 4212],
        "F":    [29226, 8, 2513],
        "T":    [27447, 5, 4295],
    },
    # order: [bearish, bullish, invalid, mixed, neutral]
    "news_sentiment": {
        "base": [3429, 4771, 2, 0, 3729],
        "F":    [3616, 5495, 3, 0, 2817],
        "T":    [3199, 4092, 1, 0, 4641],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    "fiqasa_sentiment": {
        "base": [0, 0, 556, 384, 233],
        "F":    [1, 0, 505, 223, 444],
        "T":    [0, 0, 657, 354, 162],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    "imdb_sklearn": {
        "base": [0, 0, 5313, 0, 4687],
        "F":    [0, 0, 5197, 0, 4803],
        "T":    [0, 0, 5740, 0, 4260],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    "sst2": {
        "base": [2, 1, 5548, 525, 3924],
        "F":    [4, 17, 4992, 329, 4657],
        "T":    [1, 0, 6163, 855, 2981],
    }
}

# --- compute entropies (bits) ---
rows = []
for name, data in datasets.items():
    base_ent = distribution_entropy(data["base"], base="bits")
    f_ent = distribution_entropy(data["F"], base="bits")
    t_ent = distribution_entropy(data["T"], base="bits")
    rows.append({
        "数据集": name,
        "Base分布熵 (bits)": round(base_ent, 6),
        "F分布熵 (bits)": round(f_ent, 6),
        "T分布熵 (bits)": round(t_ent, 6),
    })

df_entropy = pd.DataFrame(rows).sort_values("数据集").reset_index(drop=True)
print(df_entropy.to_string(index=False))
