# qwen2_5_download.py
# pip install -U huggingface_hub

import os
from huggingface_hub import snapshot_download, login

# ===== 1) 按需修改这里 =====
MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
TARGET_DIR = os.getenv("TARGET_DIR", "./models/qwen2.5-7B-Instruct")  # 保存目录
HF_TOKEN   = os.getenv("HUGGINGFACE_TOKEN", "")  # 如需私有/大文件加速可填

# 可选：国内镜像/代理（按需启用）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 第三方镜像示例
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"         # 开启更快传输（可选）
# os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
  
def main():
    if HF_TOKEN:
        login(HF_TOKEN)

    print(f"[INFO] start downloading: {MODEL_ID}")
    local_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,  # 真实文件，便于打包/拷贝
        resume_download=True,          # 断点续传
        token=HF_TOKEN or None,
        # 仅需权重文件时可以过滤，大幅减少体积（按需解开注释）
        # allow_patterns=["*.json", "*.txt", "*.md", "*.model", "*.safetensors", "*.bin", "*.py"],
        # ignore_patterns=["*.png", "*.jpg", "*.jpeg", "images/*", "assets/*"],
    )
    print(f"[OK] downloaded to: {local_path}")

if __name__ == "__main__":
    main()
