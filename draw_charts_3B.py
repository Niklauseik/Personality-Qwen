import matplotlib.pyplot as plt
import os

# === 0. 原始结果（替换为 qwen2.5-3B 的最新数字） =================
normalized_results_with_invalid = {
    "imdb_sentiment": {
        "base": {"invalid": 0, "mixed": 18, "negative": 5117, "neutral": 13, "positive": 4852},
        "f":    {"invalid": 0, "mixed": 32, "negative": 5021, "neutral": 10, "positive": 4937},
        "t":    {"invalid": 0, "mixed": 8,  "negative": 5251, "neutral": 18, "positive": 4723},
    },
    "mental_sentiment": {
        "base": {"depression": 19547, "invalid": 1, "mixed": 3, "neutral": 0, "normal": 12196},
        "f":    {"depression": 20681, "invalid": 1, "mixed": 5, "neutral": 0, "normal": 11060},
        "t":    {"depression": 18459, "invalid": 2, "mixed": 4, "neutral": 0, "normal": 13282},
    },
    "news_sentiment": {
        "base": {"bearish": 1340, "bullish": 3628, "invalid": 1,  "mixed": 0, "neutral": 6962},
        "f":    {"bearish": 1399, "bullish": 4156, "invalid": 44, "mixed": 0, "neutral": 6332},
        "t":    {"bearish": 1159, "bullish": 3417, "invalid": 0,  "mixed": 0, "neutral": 7355},
    },
    "fiqasa_sentiment": {
        "base": {"invalid": 0, "mixed": 1, "negative": 314, "neutral": 370, "positive": 488},
        "f":    {"invalid": 0, "mixed": 1, "negative": 317, "neutral": 320, "positive": 535},
        "t":    {"invalid": 0, "mixed": 0, "negative": 304, "neutral": 433, "positive": 436},
    },
    "imdb_sklearn": {
        "base": {"invalid": 0, "mixed": 18, "negative": 5142, "neutral": 7,  "positive": 4833},
        "f":    {"invalid": 0, "mixed": 45, "negative": 5037, "neutral": 6,  "positive": 4912},
        "t":    {"invalid": 0, "mixed": 15, "negative": 5276, "neutral": 15, "positive": 4694},
    },
    "sst2": {
        "base": {"invalid": 0, "mixed": 6,  "negative": 4950, "neutral": 496, "positive": 4548},
        "f":    {"invalid": 1, "mixed": 10, "negative": 4771, "neutral": 345, "positive": 4873},
        "t":    {"invalid": 0, "mixed": 7,  "negative": 5142, "neutral": 687, "positive": 4164},
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
