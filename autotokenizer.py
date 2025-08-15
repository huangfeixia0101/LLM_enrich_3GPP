from transformers import AutoTokenizer
import json

local_path = "TinyLlama_chat"

tokenizer = AutoTokenizer.from_pretrained(local_path)

MAX_TOKENS = 2046
INPUT_JSON = "dataset_for_tinyllama.json"
OUTPUT_JSON = "dataset_for_tinyllama_token2048.json"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

new_samples = []

for item in data:
    instruction_ids = tokenizer(item["instruction"], add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(item["response"], add_special_tokens=False)["input_ids"]

    # 留出2个token给 special tokens（比如 <s>, </s>）
    max_resp_tokens = MAX_TOKENS - len(instruction_ids) - 2
    if max_resp_tokens <= 0:
        # instruction 本身就超长，直接跳过
        continue

    # 对 response 按 max_resp_tokens 切块
    for i in range(0, len(response_ids), max_resp_tokens):
        chunk_ids = response_ids[i:i+max_resp_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        new_samples.append({
            "instruction": item["instruction"],
            "response": chunk_text
        })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(new_samples, f, ensure_ascii=False, indent=2)

print(f"拆分完成，生成 {len(new_samples)} 条样本")
