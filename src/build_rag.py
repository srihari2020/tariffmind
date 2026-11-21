import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

CSV_PATH = "data/hs_2022.csv"
CHROMA_PATH = "./chroma_hs"
COLLECTION_NAME = "hs_2022"


def build_index():
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    df["level"] = df["level"].astype(int)

    # Lookup tables for parent context
    chapters = df[df["level"] == 2].set_index("hscode")["description"].to_dict()
    headings = df[df["level"] == 4].set_index("hscode")["description"].to_dict()
    subheadings = df[df["level"] == 6].copy()

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    docs, metadatas, ids = [], [], []
    for row in subheadings.itertuples():
        code = row.hscode.strip()
        ch_code = code[:2]
        hd_code = code[:4]

        ch_desc = chapters.get(ch_code, "")
        hd_desc = headings.get(hd_code, "")
        sub_desc = row.description.strip()

        full_text = (
            f"HS {code} | Section {row.section} | "
            f"Chapter {ch_code}: {ch_desc} | "
            f"Heading {hd_code}: {hd_desc} | "
            f"Subheading: {sub_desc}"
        )

        docs.append(full_text)
        ids.append(code)
        metadatas.append({
            "section": row.section,
            "chapter": ch_code,
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

    print(f"Indexed {len(docs)} HS 2022 subheadings into {CHROMA_PATH}")


def retrieve_candidates(query: str, top_k: int = 10) -> list[dict]:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    results = collection.query(query_texts=[query], n_results=top_k)
    candidates = []
    for code, doc, dist in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        candidates.append({"hscode": code, "text": doc, "distance": round(dist, 4)})
    return candidates


if __name__ == "__main__":
    build_index()
    sample = "Men's woven denim jacket, 80% cotton 20% polyester"
    print(f"\nTest query: {sample}")
    for c in retrieve_candidates(sample, top_k=3):
        print(f"  [{c['hscode']}] (dist={c['distance']}) -> {c['text'][:120]}...")
