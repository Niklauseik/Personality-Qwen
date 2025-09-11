import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 模型路径配置（Qwen 版本） ===
model_configs = {
    "原始基座模型": "./qwen2.5-3B-Instruct",
    "F性格模型": "./dpo_outputs/model_f_3B",
    "T性格模型": "./dpo_outputs/model_t_3B"
}

# === 数据集路径配置 ===
dataset_configs = {
    "imdb": "datasets/sentiment/imdb.csv",
    "sst2": "datasets/sentiment/sst2.csv",
    "imdb_sklearn": "datasets/sentiment/imdb_sklearn.csv",
    "fiqasa": "datasets/finbench/fiqasa.csv",
    "news": "datasets/news/news_sentiment.csv",
    "mental": "datasets/medical/mental_health_sentiment.csv"
}

# === Prompt 构造器 ===
def build_prompt(dataset_name: str, text: str) -> str:
    if dataset_name == "imdb":
        return (
            "You are a movie review sentiment classifier. "
            "Classify the following review as either positive or negative. "
            "Respond with only one word: positive or negative.\n\n"
            f"Review:\n{text}\n\nSentiment:"
        )
    elif dataset_name == "sst2":
        return (
            "You are a sentence-level sentiment analysis model. "
            "Classify the sentiment of the sentence as positive or negative. "
            "Respond with only one word: positive or negative.\n\n"
            f"Sentence:\n{text}\n\nSentiment:"
        )
    elif dataset_name == "imdb_sklearn":
        return (
            "You are a sentiment classifier trained on user-written movie reviews. "
            "Please judge whether the sentiment of the following review is positive or negative. "
            "Only respond with: positive or negative.\n\n"
            f"Movie Review:\n{text}\n\nSentiment:"
        )
    elif dataset_name == "fiqasa":
        return (
            "You are a financial sentiment classifier. "
            "Respond with only one word: either 'positive', 'neutral', or 'negative'.\n\n"
            f"{text}"
        )
    elif dataset_name == "news":
        return (
            "You are analyzing financial news headlines. Each headline reflects a short financial opinion or fact. "
            "Please classify the overall sentiment into one of the following categories:\n"
            "- Bearish\n- Bullish\n- Neutral\n\n"
            "Respond with one word only.\n\n"
            "Example:\nText: $GM -- GM loses a bull\nAnswer: Bearish\n\n"
            "Now classify the following:\n"
            f"{text}\nAnswer:"
        )
    elif dataset_name == "mental":
        return (
            "You are given a short social media post that may reflect the mental state of the writer. "
            "Please classify it as either Normal or Depression based on the emotional content.\n\n"
            f"Text: {text}\n\nRespond with a single word: Normal or Depression."
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# === 推理函数（Qwen：使用 chat template；去除会被忽略的采样参数） ===
def local_generate(prompt, tokenizer, model):
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=False,                 # 用贪婪解码，避免 Qwen 忽略 temperature/top_p/top_k 的告警
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# === 主流程 ===
for dataset_name, dataset_path in dataset_configs.items():
    df = pd.read_csv(dataset_path)  # 需包含列：text, label
    print(f"\n📄 正在测试数据集：{dataset_name}，共 {len(df)} 条")

    for model_name, model_path in model_configs.items():
        print(f"\n🧪 正在测试模型：{model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        predictions = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset_name} | {model_name}"):
            try:
                prompt = build_prompt(dataset_name, row["text"])
                pred = local_generate(prompt, tokenizer, model)
            except Exception as e:
                pred = f"[Error] {e}"
            predictions.append(pred)

        df_result = df.copy()
        df_result["prediction"] = predictions

        save_dir = os.path.join("results", "sentiment", dataset_name, model_name)
        os.makedirs(save_dir, exist_ok=True)
        df_result_path = os.path.join(save_dir, f"{dataset_name}_sentiment_results.csv")
        df_result.to_csv(df_result_path, index=False, encoding="utf-8")
        print(f"✅ 保存完成：{dataset_name} → {df_result_path}")
