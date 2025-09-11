import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./qwen2.5-3B-Instruct"  # 你解压后的目录名（与训练时保持一致）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 tokenizer（Qwen 需要 trust_remote_code）
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,
    trust_remote_code=True
)
# 兜底：如无 pad_token，用 eos 代替（便于批量/推理）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载模型（Qwen 需要 trust_remote_code）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

# 构建提示词
prompt = "Hello, how are you? What is 1 plus 1 equal to?"

# 构建对话模板（Qwen 支持 chat template）
messages = [{"role": "user", "content": prompt}]
full_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 编码并生成
inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

# 解码输出
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)
print("✅ 模型输出：", response)
