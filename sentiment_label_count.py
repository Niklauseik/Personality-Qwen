import os
import re
import pandas as pd
from collections import Counter, defaultdict

# ========= 数据集配置 =========
datasets = [
    {
        "name":       "imdb_sentiment",
        "file":       "imdb_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/imdb",
    },
    {
        "name":       "mental_sentiment",
        "file":       "mental_sentiment_results.csv",
        "label_map":  None,
        "allowed_labels": ["normal", "depression"],
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/mental",
    },
    {
        "name":       "news_sentiment",
        "file":       "news_sentiment_results.csv",
        "label_map":  {"0": "bearish", "1": "bullish", "2": "neutral"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/news",
    },
    {
        "name":       "fiqasa_sentiment",
        "file":       "fiqasa_sentiment_results.csv",
        "label_map":  None,
        "allowed_labels": ["negative", "positive", "neutral"],
        "label_col":  "answer",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/fiqasa",
    },
    {
        "name":       "imdb_sklearn",
        "file":       "imdb_sklearn_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/imdb_sklearn",
    },
    {
        "name":       "sst2",
        "file":       "sst2_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/sst2",
    },
]

# ========= 模型文件夹名 =========
models = {
    "base": "原始基座模型",
    "f":    "F性格模型",
    "t":    "T性格模型",
}

# ========= 清洗函数 =========
def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z]", "", text.strip().lower())

# ========= 匹配函数 =========
def matches_label(text, label):
    pattern = re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE)
    return bool(pattern.search(str(text)))

# ========= 统计容器 =========
dist_all = defaultdict(lambda: defaultdict(lambda: {"true": 0, "base": 0, "f": 0, "t": 0}))

for ds in datasets:
    print(f"🔍 处理数据集：{ds['name']}")
    true_done = False

    # 获取合法标签集（原始标签 + mixed / neutral / invalid）
    allowed = set(map(clean, ds["allowed_labels"])) if ds["allowed_labels"] else set()
    if not allowed and ds["label_map"]:
        allowed = set(map(clean, ds["label_map"].values()))
    elif not allowed:
        allowed = set()

    extended = {"neutral", "mixed", "invalid"}
    all_labels = allowed.union(extended)

    for mkey, mfolder in models.items():
        file_base = os.path.join(ds["base_path"], mfolder, ds["file"])
        file_proc = file_base.replace(".csv", ".processed.csv")
        path = file_proc if os.path.exists(file_proc) else file_base

        if not os.path.exists(path):
            print(f"  ⚠️ 缺少文件：{path}")
            continue

        df = pd.read_csv(path)

        # 标签映射
        if ds["label_map"]:
            df[ds["label_col"]] = df[ds["label_col"]].astype(str).map(ds["label_map"])

        df[ds["label_col"]] = df[ds["label_col"]].astype(str).apply(clean)
        df["raw_pred"] = df[ds["pred_col"]].astype(str)

        # 统计真实标签（只统计一次）
        if not true_done:
            for lbl, cnt in Counter(df[ds["label_col"]]).items():
                if lbl in all_labels:
                    dist_all[ds["name"]][lbl]["true"] = cnt
            true_done = True

        # 统计 prediction 中的标签分布
        for lbl in all_labels:
            count = df["raw_pred"].apply(lambda x: matches_label(x, lbl)).sum()
            dist_all[ds["name"]][lbl][mkey] = count

# ========= 输出 TXT =========
outfile = "label_distribution_summary.txt"
with open(outfile, "w", encoding="utf-8") as f:
    for dname, label_dict in dist_all.items():
        f.write(f"======== {dname} ========\n")
        df_out = (
            pd.DataFrame(label_dict).T
              .fillna(0)
              .astype(int)
              .loc[:, ["true", "base", "f", "t"]]
              .rename(columns={
                  "true": "真实数量",
                  "base": "基座模型",
                  "f":    "F模型",
                  "t":    "T模型",
              })
              .sort_index()
        )
        f.write(df_out.to_string())
        f.write("\n\n")

print(f"\n📊 统计完成！结果已保存到 {outfile}")
