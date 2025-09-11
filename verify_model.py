import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./llama-3B-Instruct"  # 你解压后的目录名
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# 构建提示词
prompt = "Hello, how are you? What is 1 plus 1 equal to?"

# 构建对话模板（LLaMA 3 使用 chat 模板）
messages = [{"role": "user", "content": prompt}]
full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 编码并生成
inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

# 解码输出
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("✅ 模型输出：", response)
