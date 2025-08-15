import os
import json
import time
import hashlib
from docx import Document
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from ericai import EricAI
from concurrent.futures import ThreadPoolExecutor, as_completed

nltk.download('punkt')

client = EricAI()

CACHE_DIR = "ericai_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

def chunk_text(text, max_length=600, min_length=200):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_length:
            current += s + " "
        else:
            if len(current) < min_length and chunks:
                chunks[-1] += " " + current.strip()
            else:
                chunks.append(current.strip())
            current = s + " "
    if current.strip():
        if len(current) < min_length and chunks:
            chunks[-1] += " " + current.strip()
        else:
            chunks.append(current.strip())
    return chunks

def augment_questions(text_segment):
    return [
        f"What is described in the following section?\n{text_segment}",
        f"Explain the content below in simple terms.\n{text_segment}",
        f"Summarize the following product documentation.\n{text_segment}",
        f"Provide a QA summary of the text:\n{text_segment}",
        f"What are the key takeaways from the text below?\n{text_segment}"
    ]

def cache_path(prompt):
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def load_cache(prompt):
    path = cache_path(prompt)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("answer")
    return None

def save_cache(prompt, answer):
    path = cache_path(prompt)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "answer": answer}, f, ensure_ascii=False, indent=2)

def call_ericai_with_retry(prompt, max_retries=3):
    cached = load_cache(prompt)
    if cached:
        return cached

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                # model="Qwen/Qwen2.5-14B-Instruct-1M",
                model="Qwen/Qwen3-32B",
                messages=[
                    {"role": "system", "content": "You are an excellent communications expert who is familiar with 3GPP protocols and AAT products. Please answer the user's questions."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = completion.choices[0].message.content.strip()
            save_cache(prompt, answer)
            return answer
        except Exception as e:
            print(f"Attempt {attempt+1} failed for prompt: {e}")
            # time.sleep(1)
    return ""

def process_qa_pair(q_and_context):
    question, context = q_and_context
    answer = call_ericai_with_retry(question)
    if not answer:
        answer = context
    return {"instruction": question, "output": answer}

def generate_qa_dataset_from_text(raw_text, max_workers=4):
    chunks = chunk_text(raw_text)
    all_qa_pairs = []
    for chunk in chunks:
        questions = augment_questions(chunk)
        for q in questions:
            all_qa_pairs.append((q, chunk))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_qa_pair, qa) for qa in all_qa_pairs]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    return results

def process_single_file(file_path, output_folder, max_workers=4):
    print(f"Processing {file_path} ...")
    if file_path.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        raw_text = extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return

    qa_dataset = generate_qa_dataset_from_text(raw_text, max_workers=max_workers)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_folder, base_name + "_qa.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved QA dataset for {file_path} -> {output_path}")

def batch_process_folder(input_folder, output_folder, max_workers=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_exts = [".pdf", ".docx"]
    files = []

    # 递归遍历 input_folder 下所有文件
    for root, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in supported_exts:
                full_path = os.path.join(root, filename)
                files.append(full_path)

    for file_path in files:
        rel_path = os.path.relpath(file_path, input_folder)
        rel_base = os.path.splitext(rel_path)[0]
        safe_base = rel_base.replace(os.sep, "_")  # 防止多级目录影响文件名
        output_file_path = os.path.join(output_folder, safe_base + "_qa.json")

        # 跳过已处理的文件（可选）
        if os.path.exists(output_file_path):
            print(f"⏩ Skipping existing: {output_file_path}")
            continue

        try:
            process_single_file(file_path, output_folder, max_workers=max_workers)
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")



def main():
    input_folder = "./QA_Data"       # 这里改成你要处理的文件夹路径
    output_folder = "./QA_Data_output" # 这里改成结果保存的文件夹路径
    max_workers = 4                    # 线程数，建议4或根据你机器调整

    batch_process_folder(input_folder, output_folder, max_workers=max_workers)
    print("🎉 All files processed!")

if __name__ == "__main__":
    main()
