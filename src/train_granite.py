import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="unsloth/granite-3.3-2b-instruct",
        help="Base model (e.g. unsloth/granite-3.3-2b-instruct or unsloth/granite-3.3-8b-instruct)",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output-dir", default="granite-3.3-hs-lora")
    parser.add_argument("--export-gguf", action="store_true")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_dataset("json", data_files="data/train_gri.jsonl", split="train")

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=args.max_seq_len,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            warmup_steps=10,
            learning_rate=2e-4,
            optim="adamw_8bit",
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            output_dir="outputs_granite_hs",
            report_to="none",
        ),
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved LoRA adapter to {args.output_dir}")

    if args.export_gguf:
        gguf_dir = f"{args.output_dir}-gguf"
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
        print(f"Exported Q4_K_M GGUF to {gguf_dir}")


if __name__ == "__main__":
    main()
