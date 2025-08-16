def build_report(pages, chunks):
    total_chars_source = sum(len(p["text"]) for p in pages)
    total_chars_chunks = sum(len(c["text"]) for c in chunks)
    coverage = (total_chars_chunks / total_chars_source) if total_chars_source else 0.0
    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "source_chars": total_chars_source,
        "chunk_chars": total_chars_chunks,
        "coverage_ratio": round(coverage, 3)
    }
