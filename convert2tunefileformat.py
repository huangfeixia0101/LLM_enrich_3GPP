import os
import json
import re

INPUT_DIR = "dataset"  # JSON 文件夹根目录
OUTPUT_JSON = "dataset_for_tinyllama.json"
MAX_CHARS_PER_SAMPLE = None
KEEP_MARKDOWN = False  # True 保留Markdown, False 清理Markdown

def clean_markdown(text):
    if KEEP_MARKDOWN:
        return text.strip()
    text = re.sub(r"[*#]", "", text)  # 去掉 Markdown 标记
    text = re.sub(r"\s+\n", "\n", text)  # 去掉多余空行
    return text.strip()

def split_text(text, max_chars):
    if max_chars is None:  # 不拆分
        return [text]
    if len(text) <= max_chars:
        return [text]
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def normalize_item(item):
    """将不同字段名的结构映射成 instruction / response，并拼接 topic/domain"""
    if "instruction" in item and "output" in item:
        instruction = item["instruction"]
        response = item["output"]
    elif "prompt" in item and "answer" in item:
        instruction = item["prompt"].strip('"')  # 去掉包裹引号
        response = item["answer"]
    else:
        return None  # 不符合要求

    # 如果有 topic/domain，就拼接到 instruction 前
    prefix_parts = []
    if "topic" in item and item["topic"]:
        prefix_parts.append(f"[Topic: {item['topic']}]")
    if "domain" in item and item["domain"]:
        prefix_parts.append(f"[Domain: {item['domain']}]")
    if prefix_parts:
        instruction = " ".join(prefix_parts) + " " + instruction

    instruction = clean_markdown(instruction)
    response = clean_markdown(response)
    return instruction, response

def process_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"跳过无效 JSON 文件: {file_path}")
            return []

    samples = []
    if isinstance(data, dict):
        data = [data]  # 单个对象也能处理

    for item in data:
        norm = normalize_item(item)
        if not norm:
            continue
        instruction, response = norm

        for ins_part, res_part in zip(split_text(instruction, MAX_CHARS_PER_SAMPLE),
                                      split_text(response, MAX_CHARS_PER_SAMPLE)):
            samples.append({
                "instruction": ins_part,
                "response": res_part
            })
    return samples

all_samples = []
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower().endswith(".json"):
            file_path = os.path.join(root, file)
            all_samples.extend(process_json_file(file_path))

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_samples, f, ensure_ascii=False, indent=2)

mode = "保留 Markdown" if KEEP_MARKDOWN else "清理 Markdown"
print(f"转换完成，共生成 {len(all_samples)} 条样本（模式：{mode}）")
