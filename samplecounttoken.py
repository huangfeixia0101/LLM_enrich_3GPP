import json
import random
from transformers import AutoTokenizer
import numpy as np

# 本地 TinyLLaMA tokenizer 路径
local_path = "TinyLlama_chat"
tokenizer = AutoTokenizer.from_pretrained(local_path)

INPUT_JSON = "dataset_for_tinyllama_token2048.json"
SAMPLE_JSON = "dataset_sample.json"
SAMPLE_SIZE = 5000  # 采样条数

# 读取数据
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# 统计 token 分布
token_lens = []
for item in data:
    text = item["instruction"].strip() + "\n" + item["response"].strip()
    tokens = tokenizer(text)["input_ids"]
    token_lens.append(len(tokens))

print(f"样本总数: {len(data)}")
print(f"平均长度: {np.mean(token_lens):.2f} tokens")
print(f"最大长度: {np.max(token_lens)} tokens")
print(f"中位数长度: {np.median(token_lens)} tokens")

# 随机采样
sampled_data = random.sample(data, SAMPLE_SIZE)
with open(SAMPLE_JSON, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"已保存随机采样 {SAMPLE_SIZE} 条到 {SAMPLE_JSON}")
