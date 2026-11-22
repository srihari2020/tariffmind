import argparse
import gc
import json
import re
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from validator import HSValidator


def extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def evaluate_model(model_name: str, samples: list[dict], validator: HSValidator) -> dict:
    print(f"\n>>> Benchmarking: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    stats = {
        "total": len(samples),
        "valid_json": 0,
        "classifiable_total": 0,
        "exact_2": 0,
        "exact_4": 0,
        "exact_6": 0,
        "hallucinated": 0,
        "abstain_total": 0,
        "abstain_correct": 0,
        "total_tokens": 0,
        "total_time": 0.0,
    }

    for sample in tqdm(samples, desc=model_name):
        messages = sample["messages"][:2]
        gold = json.loads(sample["messages"][2]["content"])

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        stats["total_tokens"] += len(gen_tokens)
        stats["total_time"] += elapsed

        raw_pred = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred = extract_json(raw_pred)

        if pred is not None:
            stats["valid_json"] += 1

        gold_code = gold.get("hs_code_6digit")
        pred_code = pred.get("hs_code_6digit") if pred else None

        if gold_code is None:
            stats["abstain_total"] += 1
            if pred and pred.get("needs_human_review") is True and pred_code is None:
                stats["abstain_correct"] += 1
        else:
            stats["classifiable_total"] += 1
            if pred_code:
                val = validator.validate(pred_code)
                norm = val["hs_code_6digit"]
                if not val["is_valid"]:
                    stats["hallucinated"] += 1
                if norm[:2] == gold_code[:2]:
                    stats["exact_2"] += 1
                if norm[:4] == gold_code[:4]:
                    stats["exact_4"] += 1
                if norm == gold_code:
                    stats["exact_6"] += 1

    # Free VRAM before loading next model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    c_tot = max(1, stats["classifiable_total"])
    a_tot = max(1, stats["abstain_total"])

    return {
        "json_validity_pct": round(100 * stats["valid_json"] / stats["total"], 2),
        "acc_2digit_pct": round(100 * stats["exact_2"] / c_tot, 2),
        "acc_4digit_pct": round(100 * stats["exact_4"] / c_tot, 2),
        "acc_6digit_pct": round(100 * stats["exact_6"] / c_tot, 2),
        "abstention_acc_pct": round(100 * stats["abstain_correct"] / a_tot, 2),
        "hallucination_rate_pct": round(100 * stats["hallucinated"] / c_tot, 2),
        "avg_latency_sec": round(stats["total_time"] / stats["total"], 2),
        "tokens_per_sec": round(stats["total_tokens"] / max(1e-6, stats["total_time"]), 2),
        "avg_output_tokens": round(stats["total_tokens"] / stats["total"], 1),
    }


def plot_charts(base: dict, ft: dict):
    plt.style.use("dark_background")

    # 1. Accuracy & Reliability Chart
    metrics = [
        ("JSON Validity", "json_validity_pct"),
        ("2-Digit\n(Chapter)", "acc_2digit_pct"),
        ("4-Digit\n(Heading)", "acc_4digit_pct"),
        ("6-Digit\n(Exact)", "acc_6digit_pct"),
        ("Abstention\nAccuracy", "abstention_acc_pct"),
    ]
    labels = [m[0] for m in metrics]
    base_vals = [base[m[1]] for m in metrics]
    ft_vals = [ft[m[1]] for m in metrics]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    b1 = ax.bar(x - width / 2, base_vals, width, label="Base Granite 3.3 2B", color="#64748b")
    b2 = ax.bar(x + width / 2, ft_vals, width, label="TariffMind (Fine-Tuned)", color="#22c55e")

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("TariffMind vs. Base Granite 3.3 2B — Classification Accuracy", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, fontsize=10)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("assets/benchmark_accuracy.png")
    plt.close()

    # 2. Speed & Token Efficiency Chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=180)
    panels = [
        ("Avg Latency (sec/item)\n(Lower is better)", "avg_latency_sec", "s"),
        ("Throughput (tokens/sec)\n(Higher is better)", "tokens_per_sec", " tok/s"),
        ("Avg Output Tokens\n(Tighter JSON = faster)", "avg_output_tokens", " tok"),
    ]

    for ax, (title, key, unit) in zip(axes, panels):
        vals = [base[key], ft[key]]
        bars = ax.bar(["Base 2B", "TariffMind"], vals, color=["#64748b", "#3b82f6"], width=0.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
        ax.set_ylim(0, max(vals) * 1.25 + 1)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h}{unit}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5)

    fig.suptitle("TariffMind vs. Base Granite 3.3 2B — Speed & Token Efficiency", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/benchmark_speed_tokens.png")
    plt.close()
    print("\nSaved charts to assets/benchmark_accuracy.png and assets/benchmark_speed_tokens.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="unsloth/granite-3.3-2b-instruct")
    parser.add_argument("--adapter", default="granite-3.3-hs-lora")
    parser.add_argument("--eval-file", default="data/eval_gri.jsonl")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples (0 = all 138)")
    args = parser.parse_args()

    with open(args.eval_file) as f:
        samples = [json.loads(line) for line in f]
    if args.limit > 0:
        samples = samples[: args.limit]

    validator = HSValidator()
    base_res = evaluate_model(args.base, samples, validator)
    ft_res = evaluate_model(args.adapter, samples, validator)

    results = {"num_samples": len(samples), "base": base_res, "finetuned": ft_res}
    with open("data/benchmark_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Base vs. TariffMind Comparison ===")
    print(json.dumps(results, indent=2))
    plot_charts(base_res, ft_res)


if __name__ == "__main__":
    main()
