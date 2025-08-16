import argparse, json, os, sys, time
from pathlib import Path
from tqdm import tqdm

from modules.parse_pdf import parse_pdf_to_pages
from modules.normalize_content import normalize_pages
from modules.structure_detect import detect_headings
from modules.chunking import chunk_documents
from modules.qc_checks import build_report

# Optional components (guard imports)
def _import_embeddings():
    try:
        from modules.embeddings import embed_texts, save_faiss_index
        return embed_texts, save_faiss_index
    except Exception as e:
        return None, None

def _import_bm25():
    try:
        from modules.bm25_index import build_bm25, build_tfidf, save_sparse_index
        return build_bm25, build_tfidf, save_sparse_index
    except Exception as e:
        return None, None, None

def save_dataset(chunks, work_dir):
    import pandas as pd
    import json
    out_jsonl = Path(work_dir) / "chunks.jsonl"
    out_parquet = Path(work_dir) / "chunks.parquet"
    # write jsonl
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # parquet if available
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
        df = pd.DataFrame(chunks)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, out_parquet)
    except Exception:
        pass
    return str(out_jsonl)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config JSON")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    paths = cfg["paths"]
    work_dir = Path(paths["work_dir"]); work_dir.mkdir(parents=True, exist_ok=True)
    indices_dir = Path(paths["indices_dir"]); indices_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(paths["reports_dir"]); reports_dir.mkdir(parents=True, exist_ok=True)

    # === LLM assist (optional) ===
    llm_cfg = cfg.get("llm", {"enabled": False})
    llm_enabled = llm_cfg.get("enabled", False)
    llm = None
    cache = None
    if llm_enabled:
        from modules.llm_client import LLMClient
        from modules.cache import DiskCache
        llm = LLMClient(base_url=llm_cfg.get("base_url"), model=llm_cfg.get("model"))
        cache = DiskCache(root=str(Path(paths["work_dir"]) / "llm_cache"))

    # 1) Parse
    pages = parse_pdf_to_pages(paths["input_pdf"], ocr_if_needed=cfg["parse"]["ocr_if_needed"])

    # 2) Normalize
    pages_norm = normalize_pages(pages)

    # 3) Structure
    docs = None
    if llm_enabled and llm_cfg.get("sectionize", True):
        # Build page paragraphs
        import regex as re
        page_paras = []
        for p in pages_norm:
            for para in re.split(r"\n{2,}", p["text"]):
                t = para.strip()
                if t:
                    page_paras.append({"page": p["page_num"], "text": t})
        try:
            from modules.structure_llm import llm_sections
            blocks = llm_sections(llm, cache, page_paras)
            docs = blocks
        except Exception as e:
            print("[warn] LLM sectionize failed, falling back:", e)
    if docs is None:
        docs = detect_headings(pages_norm)

    # 4) Chunk
    chunks = chunk_documents(docs, target_chars=cfg["chunking"]["target_chars"], overlap=cfg["chunking"]["overlap"])

    # 5) Save dataset
    dataset_path = save_dataset(chunks, work_dir)
    # 5.1) Optional: LLM extractive QA per chunk
    if llm_enabled and llm_cfg.get("qa_pairs", True):
        from modules.extractive_qa import make_qa
        import json
        qa_path = Path(work_dir) / "qa.jsonl"
        with open(qa_path, "w", encoding="utf-8") as fqa:
            for ch in chunks:
                try:
                    qa = make_qa(llm, cache, ch)
                    fqa.write(json.dumps({"chunk_id": ch["chunk_id"], **qa}, ensure_ascii=False) + "\\n")
                except Exception as e:
                    print("[warn] QA failed for chunk", ch["chunk_id"], e)


    # 6) Optional: BM25/TFIDF
    if cfg["bm25"]["enabled"]:
        build_bm25, build_tfidf, save_sparse_index = _import_bm25()
        if build_bm25 is not None:
            try:
                bm25_obj = build_bm25([c["text"] for c in chunks])
                save_sparse_index(bm25_obj, indices_dir / "bm25.pkl")
            except Exception as e:
                print("[warn] BM25 build failed:", e, file=sys.stderr)
        if build_tfidf is not None:
            try:
                tfidf = build_tfidf([c["text"] for c in chunks])
                save_sparse_index(tfidf, indices_dir / "tfidf.pkl")
            except Exception as e:
                print("[warn] TFIDF build failed:", e, file=sys.stderr)

    # 7) Optional: Embeddings + FAISS
    if cfg["embeddings"]["enabled"]:
        embed_texts, save_faiss_index = _import_embeddings()
        if embed_texts is not None:
            try:
                vecs = embed_texts([c["text"] for c in chunks], model_name=cfg["embeddings"].get("model_name", None), device=cfg["embeddings"].get("device", "cpu"))
                # Save basic .npy for embeddings
                import numpy as np
                import json
                import pathlib
                np.save(pathlib.Path(work_dir) / "embeddings.npy", vecs)
                # Optional: push to Pinecone if configured
                try:
                    vcfg = cfg.get("vectordb", {})
                    if vcfg and vcfg.get("provider") == "pinecone":
                        from modules.vectordb_pinecone import push_to_pinecone
                        push_to_pinecone(vecs, chunks, cfg)
                        print("Pinecone upsert complete.")
                except Exception as e:
                    print("[warn] Pinecone push failed:", e, file=sys.stderr)

                if cfg.get("faiss", {}).get("enabled", False) and save_faiss_index is not None:
                    metric = cfg["faiss"].get("metric", "ip")
                    ids = list(range(len(chunks)))
                    save_faiss_index(vecs, ids, indices_dir / "faiss.index", metric=metric)
            except Exception as e:
                print("[warn] Embeddings/FAISS failed:", e, file=sys.stderr)

    # 8) QC report
    report = build_report(pages, chunks)
    with open(Path(reports_dir) / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("DONE")
    print("Dataset:", dataset_path)
    print("Indices dir:", indices_dir)
    print("Report:", Path(reports_dir) / "report.json")

if __name__ == "__main__":
    main()
