import argparse
import json
import re
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="granite-3.3-hs-lora")
    parser.add_argument("--eval-file", default="data/eval_gri.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = run all eval samples")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    validator = HSValidator()

    with open(args.eval_file) as f:
        samples = [json.loads(line) for line in f]
    if args.limit > 0:
        samples = samples[: args.limit]

    stats = {
        "total": len(samples),
        "valid_json": 0,
        "classifiable_total": 0,
        "exact_2": 0,
        "exact_4": 0,
        "exact_6": 0,
        "hallucinated_codes": 0,
        "abstain_total": 0,
        "abstain_correct": 0,
    }
    detailed = []

    for sample in tqdm(samples, desc="Evaluating"):
        messages = sample["messages"][:2]  # system + user
        gold = json.loads(sample["messages"][2]["content"])

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw_pred = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
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
                norm_pred = val["hs_code_6digit"]
                if not val["is_valid"]:
                    stats["hallucinated_codes"] += 1
                if norm_pred[:2] == gold_code[:2]:
                    stats["exact_2"] += 1
                if norm_pred[:4] == gold_code[:4]:
                    stats["exact_4"] += 1
                if norm_pred == gold_code:
                    stats["exact_6"] += 1

        detailed.append({
            "gold_code": gold_code,
            "pred_code": pred_code,
            "gold_review": gold.get("needs_human_review"),
            "pred_review": pred.get("needs_human_review") if pred else None,
            "raw_pred": raw_pred,
        })

    c_tot = max(1, stats["classifiable_total"])
    a_tot = max(1, stats["abstain_total"])

    summary = {
        "total_samples": stats["total"],
        "json_validity_pct": round(100 * stats["valid_json"] / stats["total"], 2),
        "hallucination_rate_pct": round(100 * stats["hallucinated_codes"] / c_tot, 2),
        "acc_2digit_pct": round(100 * stats["exact_2"] / c_tot, 2),
        "acc_4digit_pct": round(100 * stats["exact_4"] / c_tot, 2),
        "acc_6digit_pct": round(100 * stats["exact_6"] / c_tot, 2),
        "abstention_acc_pct": round(100 * stats["abstain_correct"] / a_tot, 2),
    }

    with open("data/eval_results.json", "w") as f:
        json.dump({"summary": summary, "details": detailed}, f, indent=2)

    print("\n=== TariffMind Evaluation Summary ===")
    for k, v in summary.items():
        print(f"  {k:<24}: {v}")


if __name__ == "__main__":
    main()
