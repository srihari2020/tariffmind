from unsloth import FastLanguageModel

ADAPTER_DIR = "granite-3.3-hs-lora"
GGUF_DIR = "granite-3.3-hs-gguf"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=2048,
    load_in_4bit=True,
)

model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q4_k_m",
)
print(f"Exported Q4_K_M GGUF to ./{GGUF_DIR}")
