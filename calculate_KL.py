import numpy as np
import pandas as pd

# KL divergence in bits (log2)
def kl_divergence_bits(q_counts, p_counts, epsilon=1e-12):
    p_probs = np.array(p_counts, dtype=float)
    q_probs = np.array(q_counts, dtype=float)
    p_probs = p_probs / p_probs.sum()
    q_probs = q_probs / q_probs.sum()
    p_probs = np.where(p_probs == 0, epsilon, p_probs)
    q_probs = np.where(q_probs == 0, epsilon, q_probs)
    return np.sum(q_probs * np.log2(q_probs / p_probs))

# ===== current counts filled from your table =====
# Each dataset uses the label order shown in the comment.
datasets = {
    # order: [invalid, mixed, negative, neutral, positive]
    'imdb_sentiment': {
        'base': [1, 18, 5322, 20, 4639],
        'F':    [2, 55, 5146,  6, 4790],
        'T':    [0, 29, 5647, 87, 4237],
    },
    # order: [depression, invalid, normal]
    'mental_sentiment': {
        'base': [27527, 8, 4212],
        'F':    [29226, 8, 2513],
        'T':    [27447, 5, 4295],
    },
    # order: [bearish, bullish, invalid, mixed, neutral]
    'news_sentiment': {
        'base': [3429, 4771, 2, 0, 3729],
        'F':    [3616, 5495, 3, 0, 2817],
        'T':    [3199, 4092, 1, 0, 4641],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    'fiqasa_sentiment': {
        'base': [0, 0, 556, 384, 233],
        'F':    [1, 0, 505, 223, 444],
        'T':    [0, 0, 657, 354, 162],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    'imdb_sklearn': {
        'base': [0, 0, 5313, 0, 4687],
        'F':    [0, 0, 5197, 0, 4803],
        'T':    [0, 0, 5740, 0, 4260],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    'sst2': {
        'base': [2, 1, 5548, 525, 3924],
        'F':    [4, 17, 4992, 329, 4657],
        'T':    [1, 0, 6163, 855, 2981],
    }
}

# Compute KL(F||Base) and KL(T||Base)
rows = []
for name, d in datasets.items():
    f_kl = kl_divergence_bits(d['F'], d['base'])
    t_kl = kl_divergence_bits(d['T'], d['base'])
    rows.append({
        "数据集": name,
        "F模型 KL散度 (bits)": round(f_kl, 6),
        "T模型 KL散度 (bits)": round(t_kl, 6),
    })

df = pd.DataFrame(rows).sort_values("数据集").reset_index(drop=True)
print(df.to_string(index=False, float_format='%.6f'))
