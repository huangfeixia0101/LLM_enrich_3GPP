# LLM_enrich_3GPP
LLM  enrich 3GPP knowledge

Target:
Generate a small LLM with the knowledge of 3GPP that can running on CPU.

Way:
1. Using TinyLlama as base LLM. https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/main 
2. Prepare QA data of 3GPP(Distilled Qwen3 model).
3. Fine-tune TinyLlama with QLoRA on an RTX 4060 GPU and set up the training environment.
4. Trainer and TrainingArguments.
5. Deploy new LLM on CPU and test it.

model files https://huggingface.co/huanfeixia/LLM_enrich_3GPP/upload/main

![final_result_example1](LLM_example(1).png)
![final_result_example1](LLM_example(2).png)
![final_result_example1](LLM_example(3).png)
