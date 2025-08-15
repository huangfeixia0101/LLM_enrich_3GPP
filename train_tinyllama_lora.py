# train_tinyllama_lora.py
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ========== 配置 ==========
MODEL_PATH = "TinyLlama_chat"  # TinyLLaMA 本地路径
DATA_PATH = "dataset_for_tinyllama_token2048.json"
OUTPUT_DIR = "./tinyllama_qa_lora"

MAX_TOKENS = 2048
BATCH_SIZE = 2
EPOCHS = 3
LR = 3e-4

# ========================

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto"
)

# 准备 LoRA 微调
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 加载数据
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

# Tokenize 函数
def tokenize_fn(ex):
    text = ex["instruction"].strip() + "\n" + ex["response"].strip()
    enc = tokenizer(text, truncation=True, max_length=MAX_TOKENS)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_checkpointing=True,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=True,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    report_to="none",  # 不上传日志
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# 开始训练
trainer.train()

# 保存 LoRA 权重
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"训练完成，LoRA 权重已保存到 {OUTPUT_DIR}")
