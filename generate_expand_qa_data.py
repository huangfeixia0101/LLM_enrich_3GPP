import os
import json
import time
import re
import ast
from ericai import EricAI

client = EricAI()

INPUT_DIR = "QA_Data_output_From_ericai"
OUTPUT_DIR = "QA_Data_expanded_From_ericai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Logging ==========
def log(msg):
    print(f"[LOG] {msg}")

def err(msg):
    print(f"[ERROR] {msg}")

# ========== Safe JSON parse for multi-layer escaped JSON strings ==========
def safe_parse_response(response_text: str) -> list[str]:
    """
    超鲁棒提取字符串数组，兼容任何 LLM 的转义/套娃格式
    """
    s = response_text.strip()

    # 1. 先尝试标准 JSON（万一哪天模型突然守规矩）
    import json
    try:
        obj = s
        for _ in range(3):
            if isinstance(obj, str):
                obj = json.loads(obj)
            else:
                break
        if isinstance(obj, list) and all(isinstance(i, str) for i in obj):
            return obj
    except Exception:
        pass

    # 2. 正则暴力提取最里层字符串数组
    #    匹配 ["..."] 或 ["...","..."] 形式的字符串
    m = re.search(r'\[\s*(?:"(?:[^"\\]|\\.)*"\s*(?:,\s*"[^"\\]*(?:\\.[^"\\]*)*"\s*)*)\]', s)
    if m:
        inner = m.group(0)
    else:
        # 兜底：去掉首尾的 [" 和 "]
        inner = re.sub(r'^\s*\[\s*"\s*|\s*"\s*\]\s*$', '', s, flags=re.DOTALL)

    # 3. 把里面所有转义/双双引号统一清理
    inner = re.sub(r'\\"', '"', inner)   # 去掉 \"
    inner = re.sub(r'""', '"', inner)    # 去掉 Excel 双双引号

    # 4. 再尝试一次 JSON
    try:
        return json.loads(inner, strict=False)
    except Exception:
        pass

    # 5. 终极兜底：用正则直接抓引号里的内容
    quoted = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', inner)
    if quoted:
        return [q.replace('\\"', '"') for q in quoted]

    print("[ERROR] safe_parse_response: cannot extract list")
    return []

# ========== Generate Follow-up Questions ==========
def generate_followups(question, topic, layer=1, max_layer=2, retries=3):
    if layer > max_layer:
        return []

    prompt = (
        f"You are a professional 5G Core Network trainer.\n"
        f"Generate 5 fine-grained technical sub-questions to better understand the following question:\n"
        f"\"{question}\"\n\n"
        f"Each sub-question should be technical, focused, and concise.\n"
        f"Return result as a JSON array of strings only."
    )

    for attempt in range(retries):
        try:
            log(f"[Layer {layer}] Sending prompt for: {question}")
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-14B-Instruct-1M",
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content.strip()
            log(f"[Layer {layer}] Raw response: {content[:200]}...")
            # DEBUG打印原始内容
            print(f"[DEBUG] Raw response content:\n{content}\n")

            followups = safe_parse_response(content)
            if not followups:
                raise ValueError("No valid list returned")

            children = []
            for q in followups:
                children.append({
                    "instruction": q,
                    "topic": topic,
                    "layer": layer,
                    "children": generate_followups(q, topic, layer + 1, max_layer, retries)
                })
            return children

        except Exception as e:
            err(f"[Layer {layer}] Failed attempt {attempt + 1} for: {question}")
            err(str(e))
            # time.sleep(2)

    return []

# ========== Process a Single File ==========
def process_file(file_path):
    log(f"Processing file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        err(f"Failed to load JSON file: {file_path}")
        err(str(e))
        return

    expanded = []
    for item in data:
        try:
            question = item.get("instruction", "").strip("* \n")
            topic = item.get("topic", "Unknown")
            domain = item.get("domain", "5G Core Network")
            output = item.get("output", "")

            log(f"Generating follow-ups for: {question}")
            item_struct = {
                "instruction": question,
                "output": output,
                "topic": topic,
                "domain": domain,
                "layer": 0,
                "children": generate_followups(question, topic, layer=1)
            }
            expanded.append(item_struct)

        except Exception as e:
            err(f"Error processing item: {item}")
            err(str(e))

    # Save result
    base_name = os.path.basename(file_path)
    save_path = os.path.join(OUTPUT_DIR, base_name)
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(expanded, f, indent=2, ensure_ascii=False)
        log(f"✅ Saved expanded file to: {save_path}")
    except Exception as e:
        err(f"Failed to save file: {save_path}")
        err(str(e))

# ========== Batch Process ==========
def process_all_files():
    log(f"Scanning folder: {INPUT_DIR}")
    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                process_file(file_path)

# ========== Main ==========
if __name__ == "__main__":
    process_all_files()
