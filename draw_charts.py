import matplotlib.pyplot as plt
import os

# === 0. 原始结果（直接填入最新数字） =========================
normalized_results_with_invalid = {
    "imdb_sentiment": {
        "base": {"invalid": 1, "mixed": 18, "negative": 5322, "neutral": 20, "positive": 4639},
        "f":    {"invalid": 2, "mixed": 55, "negative": 5146, "neutral": 6,  "positive": 4790},
        "t":    {"invalid": 0, "mixed": 29, "negative": 5647, "neutral": 87, "positive": 4237},
    },
    "mental_sentiment": {
        "base": {"depression": 27527, "invalid": 8, "normal": 4212},
        "f":    {"depression": 29226, "invalid": 8, "normal": 2513},
        "t":    {"depression": 27447, "invalid": 5, "normal": 4295},
    },
    "financial_sentiment": {
        "base": {"bearish": 3429, "bullish": 4771, "invalid": 2, "mixed": 0, "neutral": 3729},
        "f":    {"bearish": 3616, "bullish": 5495, "invalid": 3, "mixed": 0, "neutral": 2817},
        "t":    {"bearish": 3199, "bullish": 4092, "invalid": 1, "mixed": 0, "neutral": 4641},
    },
    "fiqasa_sentiment": {
        "base": {"invalid": 0, "mixed": 0, "negative": 556, "neutral": 384, "positive": 233},
        "f":    {"invalid": 1, "mixed": 0, "negative": 505, "neutral": 223, "positive": 444},
        "t":    {"invalid": 0, "mixed": 0, "negative": 657, "neutral": 354, "positive": 162},
    },
    "imdb_sklearn": {
        "base": {"invalid": 0, "mixed": 0, "negative": 5313, "neutral": 0, "positive": 4687},
        "f":    {"invalid": 0, "mixed": 0, "negative": 5197, "neutral": 0, "positive": 4803},
        "t":    {"invalid": 0, "mixed": 0, "negative": 5740, "neutral": 0, "positive": 4260},
    },
    "sst2": {
        "base": {"invalid": 2,  "mixed": 1,  "negative": 5548, "neutral": 525,  "positive": 3924},
        "f":    {"invalid": 4,  "mixed": 17, "negative": 4992, "neutral": 329,  "positive": 4657},
        "t":    {"invalid": 1,  "mixed": 0,  "negative": 6163, "neutral": 855,  "positive": 2981},
    }
}

# === 1. 允许字段配置（sst2 仅保留 positive & negative） ===========
allowed_labels = {
    "imdb_sentiment":      {"negative", "positive"},
    "mental_sentiment":    {"depression", "normal"},
    "financial_sentiment": {"bearish", "bullish", "neutral"},
    "fiqasa_sentiment":    {"negative", "positive", "neutral"},
    "imdb_sklearn":        {"negative", "positive"},
    "sst2":                {"negative", "positive"},   # <- 关键修改
}

# === 2. 统一归并到 invalid ========================================
for dataset, model_data in normalized_results_with_invalid.items():
    allow = allowed_labels.get(dataset, set())
    for model_name, counts in model_data.items():
        extra_sum = 0
        for label in list(counts):  # 复制键，便于安全删除
            if label not in allow and label != "invalid":
                extra_sum += counts.pop(label)
        counts["invalid"] = counts.get("invalid", 0) + extra_sum

# === 3. 绘图与保存（与截图保持一致） =============================
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
        axs[idx].axis('equal')                # 保持为圆
        axs[idx].set_title(f"{model.upper()}")

    # 保存到 plots 文件夹
    fig.savefig(f"plots/{dataset}_prediction_distribution.png")

plt.close('all')  # 释放内存
