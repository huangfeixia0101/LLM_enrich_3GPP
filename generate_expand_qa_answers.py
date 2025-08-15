import os
import json
import hashlib
import time
from typing import Optional               # 兼容 3.9
from tqdm import tqdm
from glob import iglob
from ericai import EricAI

ROOT_DIR   = "QA_Data_expanded_From_ericai"
CACHE_DIR  = "QA_Data_expanded_From_ericai_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

client = EricAI()

# ---------- 缓存 ----------
def cache_path(prompt: str) -> str:
    h = hashlib.sha256(prompt.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def load_cache(prompt: str) -> Optional[str]:
    p = cache_path(prompt)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("answer")
    return None

def save_cache(prompt: str, answer: str) -> None:
    p = cache_path(prompt)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "answer": answer}, f, ensure_ascii=False, indent=2)

def call_ericai_with_retry(prompt: str, max_retries: int = 3) -> str:
    cached = load_cache(prompt)
    if cached is not None:
        return cached
    for attempt in range(max_retries):
        try:
            rsp = client.chat.completions.create(
                # model="Qwen/Qwen2.5-14B-Instruct-1M",
                model="Qwen/Qwen3-32B",
                messages=[
                    {"role": "system", "content": "You are an excellent communications expert who is familiar with 3GPP protocols. Please answer the user's questions in details as possible as you can."},
                    {"role": "user", "content": prompt}
                ]
            )
            ans = rsp.choices[0].message.content.strip()
            save_cache(prompt, ans)
            return ans
        except Exception as e:
            print(f"[retry {attempt+1}] {e}")
            # time.sleep(1)
    return ""

# ---------- 扁平收集 ----------
def collect_flat(nodes: list) -> list:
    flat = []
    def _walk(node: dict):
        if "instruction" in node:
            flat.append({
                "instruction": node["instruction"].strip(),
                "output": call_ericai_with_retry(node["instruction"]),
                "topic": node.get("topic", ""),
                "domain": node.get("domain", "")
            })
        for child in node.get("children", []):
            _walk(child)
    for n in nodes:
        _walk(n)
    return flat

# ---------- 主流程 ----------
json_files = list(iglob(os.path.join(ROOT_DIR, "**", "*.json"), recursive=True))
print(f"发现 {len(json_files)} 个 JSON 文件，开始处理...")

for fp in tqdm(json_files, desc="Processing"):
    try:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)          # list[dict]

        answered = collect_flat(data)

        base_dir, base_name = os.path.split(fp)
        new_name = os.path.splitext(base_name)[0] + ".answered.json"
        out_path = os.path.join(base_dir, new_name)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(answered, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"❌ Failed on {fp} -> {e}")

print("🎉 All done!")