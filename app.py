import json
import re
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).parent / "src"))
from build_rag import retrieve_candidates
from validator import HSValidator

SYSTEM_PROMPT = (
    "You are TariffMind, a customs classification engine. Given a commercial product "
    "description and candidate HS 2022 subheadings, apply the General Rules of "
    "Interpretation (GRI) and output ONLY a valid JSON object with keys: "
    "product_summary, gri_applied, hs_code_6digit, missing_attributes, needs_human_review. "
    "If critical classification details are missing, set hs_code_6digit to null, "
    "needs_human_review to true, and list the missing_attributes."
)

st.set_page_config(page_title="TariffMind | HS 2022 Classifier", layout="wide")


@st.cache_resource
def load_engine(adapter_path: str = "granite-3.3-hs-lora"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    validator = HSValidator()
    return model, tokenizer, validator


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


def classify_item(description: str, model, tokenizer, validator, top_k: int = 10) -> dict:
    candidates = retrieve_candidates(description, top_k=top_k)
    cand_block = "\n".join(f"- {c['text']}" for c in candidates)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Product Description:\n{description}\n\nCandidate HS Codes:\n{cand_block}",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    raw_output = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    parsed = extract_json(raw_output) or {
        "product_summary": description,
        "gri_applied": "Parse error",
        "hs_code_6digit": None,
        "missing_attributes": ["Valid JSON output"],
        "needs_human_review": True,
    }

    validation = None
    if parsed.get("hs_code_6digit"):
        validation = validator.validate(parsed["hs_code_6digit"])
        if not validation["is_valid"]:
            parsed["needs_human_review"] = True
            parsed["missing_attributes"] = parsed.get("missing_attributes", []) + [
                validation["error"]
            ]

    return {
        "prediction": parsed,
        "validation": validation,
        "candidates": candidates,
    }


st.title("TariffMind — Zero-Hallucination HS 2022 Classifier")
st.caption("IBM Granite 3.3 (QLoRA) + ChromaDB RAG + Deterministic HS Tree Guardrail")

tab1, tab2, tab3 = st.tabs(["Classify Single Item", "Batch Invoice CSV", "Eval Benchmark"])

with tab1:
    desc = st.text_area(
        "Commercial Invoice / Product Description",
        value="Commercial shipment: Lithium-ion rechargeable battery packs for laptops, 11.4V 4500mAh, 200 pcs",
        height=110,
    )
    if st.button("Classify Product", type="primary"):
        with st.spinner("Retrieving HS candidates and running Granite 3.3..."):
            model, tokenizer, validator = load_engine()
            res = classify_item(desc, model, tokenizer, validator)

        pred = res["prediction"]
        val = res["validation"]

        if pred.get("needs_human_review"):
            st.warning("Flagged for Human Review — Missing Technical Specifications")
            st.write("**Missing Attributes:**", pred.get("missing_attributes", []))
            st.write("**GRI Reasoning:**", pred.get("gri_applied"))
        else:
            code = val["hs_code_6digit"]
            st.success(f"Verified 6-Digit HS Code: **{code[:4]}.{code[4:]}** (`{code}`)")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Official HS Hierarchy")
                st.write(f"**Section:** {val['hierarchy']['section']}")
                st.write(f"**Chapter:** {val['hierarchy']['chapter_2']}")
                st.write(f"**Heading:** {val['hierarchy']['heading_4']}")
                st.write(f"**Subheading:** {val['hierarchy']['subheading_6']}")
            with c2:
                st.subheader("Model Output (Strict JSON)")
                st.json(pred)

        with st.expander("Top-10 ChromaDB RAG Candidates"):
            st.dataframe(pd.DataFrame(res["candidates"]), use_container_width=True)

with tab2:
    st.write("Upload a CSV containing a `description` column to classify multiple invoice line items.")
    uploaded = st.file_uploader("Upload Invoice CSV", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        if "description" not in df_in.columns:
            st.error("CSV must contain a 'description' column.")
        else:
            st.dataframe(df_in.head(), use_container_width=True)
            if st.button("Run Batch Classification", type="primary"):
                model, tokenizer, validator = load_engine()
                rows = []
                progress = st.progress(0)
                for idx, item in enumerate(df_in["description"].astype(str)):
                    res = classify_item(item, model, tokenizer, validator)
                    p = res["prediction"]
                    v = res["validation"]
                    rows.append({
                        "description": item,
                        "hs_code_6digit": v["hs_code_6digit"] if v and v["is_valid"] else None,
                        "valid_code": v["is_valid"] if v else False,
                        "needs_human_review": p.get("needs_human_review"),
                        "gri_applied": p.get("gri_applied"),
                        "missing_attributes": ", ".join(p.get("missing_attributes", [])),
                    })
                    progress.progress((idx + 1) / len(df_in))

                df_out = pd.DataFrame(rows)
                st.dataframe(df_out, use_container_width=True)
                st.download_button(
                    "Download Classified CSV",
                    df_out.to_csv(index=False).encode("utf-8"),
                    file_name="tariffmind_classified.csv",
                    mime="text/csv",
                )

with tab3:
    eval_path = Path("data/eval_results.json")
    if eval_path.exists():
        data = json.loads(eval_path.read_text())
        s = data["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("6-Digit Exact Match", f"{s['acc_6digit_pct']}%")
        m2.metric("4-Digit Heading Acc", f"{s['acc_4digit_pct']}%")
        m3.metric("Hallucination Rate", f"{s['hallucination_rate_pct']}%")
        m4.metric("Abstention Accuracy", f"{s['abstention_acc_pct']}%")

        st.subheader("Full Evaluation Summary")
        st.json(s)

        df_det = pd.DataFrame(data["details"])
        misses = df_det[df_det["gold_code"] != df_det["pred_code"]]
        st.subheader(f"Misclassified Samples ({len(misses)})")
        st.dataframe(misses, use_container_width=True)
    else:
        st.info("Run `python src/eval.py` first to populate benchmark metrics.")
