import argparse
import json
import random
import re
import pandas as pd
from build_rag import retrieve_candidates

SYSTEM_PROMPT = (
    "You are TariffMind, a customs classification engine. Given a commercial product "
    "description and candidate HS 2022 subheadings, apply the General Rules of "
    "Interpretation (GRI) and output ONLY a valid JSON object with keys: "
    "product_summary, gri_applied, hs_code_6digit, missing_attributes, needs_human_review. "
    "If critical classification details are missing, set hs_code_6digit to null, "
    "needs_human_review to true, and list the missing_attributes."
)

INVOICE_TEMPLATES = [
    "Commercial shipment: {desc}, packed in export cartons, Qty 50 pcs",
    "ITEM: {desc} | Origin: CN | New condition, retail packaging",
    "{desc} - industrial grade, 1 lot, fob shenzhen",
    "Supply of {desc}, standard specs, boxed",
    "{desc} (bulk order, commercial invoice ref #INV-{num})",
]

UNDERSPECIFIED_CASES = [
    (
        "Electronic module, black plastic casing, 100 units",
        ["primary function/application", "voltage/power rating", "whether programmable or passive"],
    ),
    (
        "Metal machine part, custom CNC machined, 20 pcs",
        ["specific machine it is solely/principally used with", "exact metal alloy"],
    ),
    (
        "Electric motor assembly, 50 units",
        ["AC vs DC motor", "output power in watts (W) or kW", "single-phase vs multi-phase"],
    ),
    (
        "Insulated cables with connectors, 500m",
        ["voltage rating (e.g., <=80V, 80V-1000V, >1000V)", "conductor material (copper/optical)"],
    ),
    (
        "Display screen panel, 15 inch, 30 pcs",
        ["panel technology (LCD/OLED)", "whether it includes touch/driver electronics", "end-use device"],
    ),
    (
        "Battery pack for portable device, 200 pcs",
        ["battery chemistry (Li-ion, NiMH, Lead-acid)", "primary vs rechargeable"],
    ),
]


def clean_desc(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(";.")


def make_messy_invoice(hd_desc: str, sub_desc: str) -> str:
    # Combine heading + subheading cleanly, avoiding redundant "Other"
    if sub_desc.lower().startswith("other") or len(sub_desc) < 25:
        base = f"{clean_desc(hd_desc)} - {clean_desc(sub_desc)}"
    else:
        base = clean_desc(sub_desc)
    tpl = random.choice(INVOICE_TEMPLATES)
    return tpl.format(desc=base, num=random.randint(1000, 9999))


def format_candidates(candidates: list[dict]) -> str:
    return "\n".join(f"- {c['text']}" for c in candidates)


def build_dataset(chapters_filter: list[str] | None, max_samples: int, seed: int = 42):
    random.seed(seed)
    df = pd.read_csv("data/hs_2022.csv", dtype=str).fillna("")
    df["level"] = df["level"].astype(int)
    df = df[df["section"] != "TOTAL"]

    headings = df[df["level"] == 4].set_index("hscode")["description"].to_dict()
    subs = df[df["level"] == 6].copy()

    if chapters_filter:
        subs = subs[subs["hscode"].str[:2].isin(chapters_filter)]

    rows = subs.to_dict("records")
    random.shuffle(rows)
    rows = rows[:max_samples]

    records = []
    for row in rows:
        code = row["hscode"].strip()
        hd_desc = headings.get(code[:4], "")
        sub_desc = row["description"].strip()
        invoice_text = make_messy_invoice(hd_desc, sub_desc)

        candidates = retrieve_candidates(invoice_text, top_k=10)
        # Ensure ground truth is present in candidates during training
        if not any(c["hscode"] == code for c in candidates):
            candidates[-1] = {
                "hscode": code,
                "text": f"HS {code} | Chapter {code[:2]} | Heading {code[:4]}: {hd_desc} | Subheading: {sub_desc}",
            }
            random.shuffle(candidates)

        target = {
            "product_summary": clean_desc(sub_desc),
            "gri_applied": f"GRI 1 & GRI 6 - Classified under Heading {code[:4]} and Subheading {code}",
            "hs_code_6digit": code,
            "missing_attributes": [],
            "needs_human_review": False,
        }

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Product Description:\n{invoice_text}\n\nCandidate HS Codes:\n{format_candidates(candidates)}",
                },
                {"role": "assistant", "content": json.dumps(target)},
            ]
        })

    # Add ~15% underspecified "abstain" examples
    num_abstain = max(10, int(len(records) * 0.15))
    for i in range(num_abstain):
        desc, missing = UNDERSPECIFIED_CASES[i % len(UNDERSPECIFIED_CASES)]
        noisy_desc = f"{desc} (ref #{random.randint(100, 999)})"
        candidates = retrieve_candidates(noisy_desc, top_k=10)
        target = {
            "product_summary": desc,
            "gri_applied": "GRI 1 - Insufficient technical specifications to determine exact 6-digit subheading",
            "hs_code_6digit": None,
            "missing_attributes": missing,
            "needs_human_review": True,
        }
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Product Description:\n{noisy_desc}\n\nCandidate HS Codes:\n{format_candidates(candidates)}",
                },
                {"role": "assistant", "content": json.dumps(target)},
            ]
        })

    random.shuffle(records)
    split_idx = int(len(records) * 0.85)
    train_data, eval_data = records[:split_idx], records[split_idx:]

    with open("data/train_gri.jsonl", "w") as f:
        for r in train_data:
            f.write(json.dumps(r) + "\n")

    with open("data/eval_gri.jsonl", "w") as f:
        for r in eval_data:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(train_data)} train and {len(eval_data)} eval samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", default="84,85", help="Comma-separated 2-digit chapters or 'all'")
    parser.add_argument("--max-samples", type=int, default=800)
    args = parser.parse_args()

    ch_filter = None if args.chapters.lower() == "all" else [c.strip() for c in args.chapters.split(",")]
    build_dataset(ch_filter, args.max_samples)
