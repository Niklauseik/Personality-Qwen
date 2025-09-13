# llama3_2_download.py
# pip install -U huggingface_hub

import os
from huggingface_hub import snapshot_download, login

# ===== 1) 按需修改这里 =====
MODEL_ID  = os.getenv("LLAMA_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
TARGET_DIR = os.getenv("TARGET_DIR", "./models/Llama-3.2-1B-Instruct")  # 保存目录
HF_TOKEN   = os.getenv("HUGGINGFACE_TOKEN", "")  # 必须有 HF 授权才能下 gated 模型

# 可选：国内镜像/代理（按需启用）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

def main():
    if HF_TOKEN:
        login(HF_TOKEN)

    print(f"[INFO] start downloading: {MODEL_ID}")
    local_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,  # 真文件，便于打包
        resume_download=True,          # 支持断点续传
        token=HF_TOKEN or None,
        # 只要权重文件时可以过滤
        # allow_patterns=["*.json", "*.txt", "*.md", "*.safetensors", "*.bin", "*.model"],
        # ignore_patterns=["*.png", "*.jpg", "*.jpeg", "images/*", "assets/*"],
    )
    print(f"[OK] downloaded to: {local_path}")

if __name__ == "__main__":
    main()
