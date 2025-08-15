import json
import re
import time
import os
from ericai import EricAI

client = EricAI()
OUTPUT_DIR = "QA_Data_output_From_ericai"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- 1. 解析问答格式 ----------
def parse_qa_text_to_list(text):
    pattern = re.compile(r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=Question:|$)", re.DOTALL | re.IGNORECASE)
    return [
        {"instruction": q.strip(), "output": a.strip()}
        for q, a in pattern.findall(text)
        if q.strip() and a.strip()
    ]


# ---------- 2. 按主题生成 ----------
def generate_qa_for_topic(topic: str, domain: str, num_qa=30, max_retries=3):
    prompt = (
        f"You are a mobile communication expert. Please generate {num_qa} technical Q&A pairs "
        f"focused on the topic: \"{topic}\" in the context of {domain}. "
        "Each Q&A should be formatted clearly as:\n"
        "Question: <question>\nAnswer: <answer>\n"
        "Avoid repetition. Questions should be technical, and answers should be informative and concise."
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-14B-Instruct-1M",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            qa_pairs = parse_qa_text_to_list(text)
            if len(qa_pairs) < num_qa // 2:
                print(f"[{topic}] Parsed too few QA pairs ({len(qa_pairs)}). Retrying...")
                time.sleep(2)
                continue
            return qa_pairs
        except Exception as e:
            print(f"[{topic}] Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return []


# ---------- 3. 去重 + 存储 ----------
def load_existing_questions(path):
    if not os.path.exists(path):
        return set(), []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        existing_qs = set(item["instruction"] for item in data)
        return existing_qs, data


def save_qa(topic, domain, new_qa):
    safe_topic = re.sub(r'[\\/:"*?<>|]', '_', topic.lower().replace(' ', '_'))
    filename = os.path.join(OUTPUT_DIR, f"qa_{safe_topic}.json")
    existing_qs, existing_data = load_existing_questions(filename)
    unique_new = [qa for qa in new_qa if qa["instruction"] not in existing_qs]
    for qa in unique_new:
        qa["topic"] = topic
        qa["domain"] = domain
    all_data = existing_data + unique_new
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"✅ [{topic}] Added {len(unique_new)} new QAs. Total now: {len(all_data)}")


# ---------- 4. 多主题调度 ----------
def batch_generate(topics_with_domain, qa_per_topic=30):
    for topic, domain in topics_with_domain:
        print(f"\n🚀 Generating QAs for: {topic} [{domain}]")
        qa_pairs = generate_qa_for_topic(topic, domain, num_qa=qa_per_topic)
        if qa_pairs:
            save_qa(topic, domain, qa_pairs)
        else:
            print(f"❌ Failed to generate for topic: {topic}")


# ---------- 5. 主函数 ----------
def main():
    topics_file = "all_3gpp_topics.json"  # 你保存的主题JSON文件路径
    if not os.path.exists(topics_file):
        print(f"⚠️ Topics file not found: {topics_file}")
        return

    with open(topics_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data格式是列表，每项含 "topic" 和 "domain"
    topics = [(item["topic"], item["domain"]) for item in data]

    batch_generate(topics, qa_per_topic=30)


if __name__ == "__main__":
    main()
