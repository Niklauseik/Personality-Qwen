import matplotlib.pyplot as plt
import os

# === 0. 原始结果（替换为 qwen2.5-3B 的最新数字） =================
normalized_results_with_invalid = {
    "imdb_sentiment": {
        "base": {"invalid": 0, "mixed": 0, "negative": 5309, "neutral": 0, "positive": 4691},
        "f":    {"invalid": 0, "mixed": 0, "negative": 5149, "neutral": 0, "positive": 4851},
        "t":    {"invalid": 0, "mixed": 0, "negative": 5435, "neutral": 0, "positive": 4565},
    },
    "mental_sentiment": {
        "base": {"depression": 15621, "invalid": 0, "normal": 16126},
        "f":    {"depression": 17428, "invalid": 0, "normal": 14319},
        "t":    {"depression": 14451, "invalid": 0, "normal": 17296},
    },
    "news_sentiment": {  # <- 原来叫 financial_sentiment，这里改名为 news_sentiment
        "base": {"bearish": 1943, "bullish": 3116, "invalid": 0, "mixed": 0, "neutral": 6872},
        "f":    {"bearish": 1546, "bullish": 3991, "invalid": 0, "mixed": 0, "neutral": 6394},
        "t":    {"bearish": 1985, "bullish": 2617, "invalid": 0, "mixed": 0, "neutral": 7329},
    },
    "fiqasa_sentiment": {
        "base": {"invalid": 0, "mixed": 0, "negative": 467, "neutral": 149, "positive": 557},
        "f":    {"invalid": 0, "mixed": 0, "negative": 447, "neutral": 86,  "positive": 640},
        "t":    {"invalid": 0, "mixed": 0, "negative": 448, "neutral": 244, "positive": 481},
    },
    "imdb_sklearn": {
        "base": {"invalid": 1, "mixed": 0, "negative": 5582, "neutral": 0, "positive": 4417},
        "f":    {"invalid": 1, "mixed": 0, "negative": 5481, "neutral": 0, "positive": 4518},
        "t":    {"invalid": 8, "mixed": 0, "negative": 5689, "neutral": 0, "positive": 4303},
    },
    "sst2": {
        "base": {"invalid": 9,  "mixed": 0, "negative": 6813, "neutral": 0, "positive": 3178},
        "f":    {"invalid": 2,  "mixed": 0, "negative": 6251, "neutral": 0, "positive": 3747},
        "t":    {"invalid": 19, "mixed": 0, "negative": 7339, "neutral": 0, "positive": 2642},
    }
}

# === 1. 允许字段配置（与数据集名称一致） ==========================
allowed_labels = {
    "imdb_sentiment":   {"negative", "positive"},
    "mental_sentiment": {"depression", "normal"},
    "news_sentiment":   {"bearish", "bullish", "neutral"},
    "fiqasa_sentiment": {"negative", "positive", "neutral"},
    "imdb_sklearn":     {"negative", "positive"},
    "sst2":             {"negative", "positive"},
}

# === 2. 统一归并到 invalid =======================================
for dataset, model_data in normalized_results_with_invalid.items():
    allow = allowed_labels.get(dataset, set())
    for model_name, counts in model_data.items():
        extra_sum = 0
        for label in list(counts):
            if label not in allow and label != "invalid":
                extra_sum += counts.pop(label)
        counts["invalid"] = counts.get("invalid", 0) + extra_sum

# === 3. 绘图与保存 ===============================================
os.makedirs("plots", exist_ok=True)

for dataset, model_data in normalized_results_with_invalid.items():
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle(f"{dataset} Prediction Distribution", fontsize=14)

    for idx, model in enumerate(['base', 'f', 't']):
        data   = model_data.get(model, {})
        labels = list(data.keys())
        sizes  = list(data.values())
        axs[idx].pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140
        )
        axs[idx].axis('equal')
        axs[idx].set_title(f"{model.upper()}")

    fig.savefig(f"plots/{dataset}_prediction_distribution.png")

plt.close('all')
