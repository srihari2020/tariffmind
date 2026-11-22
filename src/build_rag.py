import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

CSV_PATH = "data/hs_2022.csv"
CHROMA_PATH = "./chroma_hs"
COLLECTION_NAME = "hs_2022"

# Map everyday commercial invoice terms -> official WCO HS nomenclature terms
CUSTOMS_SYNONYMS = {
    r"\brechargeable\b": "electric accumulator rechargeable",
    r"\blithium[- ]?ion\b": "lithium-ion electric accumulator",
    r"\bpower ?bank\b": "electric accumulator lithium-ion portable charger",
    r"\blaptop\b": "portable automatic data processing machine laptop",
    r"\bsmartphone\b": "telephone for cellular networks smartphone",
    r"\bpcb\b": "printed circuit board",
    r"\bled\b": "light-emitting diode semiconductor",
    r"\bcharger\b": "static converter power supply charger",
    r"\bssd\b": "solid-state non-volatile storage device",
    r"\bhdmi|usb\b": "insulated electric conductor fitted with connectors",
}


def expand_customs_query(query: str) -> str:
    additions = []
    q_lower = query.lower()
    for pattern, synonym in CUSTOMS_SYNONYMS.items():
        if re.search(pattern, q_lower):
            additions.append(synonym)
    if additions:
        return f"{query} | Customs terms: {', '.join(additions)}"
    return query


def build_index():
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    df["level"] = df["level"].astype(int)
    df = df[df["section"] != "TOTAL"]

    headings = df[df["level"] == 4].set_index("hscode")["description"].to_dict()
    subheadings = df[df["level"] == 6].copy()

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    docs, metadatas, ids = [], [], []
    for row in subheadings.itertuples():
        code = row.hscode.strip()
        hd_code = code[:4]
        hd_desc = headings.get(hd_code, "").strip()
        sub_desc = row.description.strip()

        full_text = f"HS {code} | {hd_desc} — {sub_desc}"
        docs.append(full_text)
        ids.append(code)
        metadatas.append({
            "section": row.section,
            "chapter": code[:2],
            "heading": hd_code,
            "subheading": code,
        })

    batch_size = 500
    for i in range(0, len(docs), batch_size):
        collection.upsert(
            documents=docs[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )

    print(f"Re-indexed {len(docs)} clean HS 2022 subheadings into {CHROMA_PATH}")


def retrieve_candidates(query: str, top_k: int = 15) -> list[dict]:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    expanded = expand_customs_query(query)
    results = collection.query(query_texts=[expanded], n_results=top_k)
    return [
        {"hscode": code, "text": doc, "distance": round(dist, 4)}
        for code, doc, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )
    ]


if __name__ == "__main__":
    q = "Lithium-ion rechargeable battery packs for laptops, 11.4V 4500mAh"
    print(f"Expanded query: {expand_customs_query(q)}\n")
    for c in retrieve_candidates(q, top_k=5):
        print(f"  [{c['hscode']}] (dist={c['distance']}) -> {c['text'][:110]}...")
