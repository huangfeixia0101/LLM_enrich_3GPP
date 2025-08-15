# LLM_enrich_3GPP
LLM  enrich 3GPP knowledge

Target:
Generate a small LLM with the knowledge of 3GPP and AAT producte, that can running on CPU.

Way:
1. Using TinyLlama as base LLM. https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/main 
2. Prepare QA data of 3GPP(Distilled Qwen3 model) and AAT producte documents.
3. Fine-tune TinyLlama with QLoRA on an RTX 4060 GPU and set up the training environment.
4. Trainer and TrainingArguments.
5. Deploy new LLM on CPU and test it.
