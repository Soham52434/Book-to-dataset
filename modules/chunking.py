from typing import List, Dict

def chunk_documents(blocks: List[Dict], target_chars: int = 1200, overlap: int = 120) -> List[Dict]:
    """
    Greedy fixed-size character chunking with overlap across contiguous blocks.
    Emits rows: {chunk_id, section, page_start, page_end, text}
    """
    chunks = []
    buf = []
    buf_len = 0
    page_start = None
    page_end = None
    chunk_id = 0

    def flush():
        nonlocal buf, buf_len, page_start, page_end, chunk_id
        if buf_len == 0: 
            return
        text = "\n\n".join(buf).strip()
        chunks.append({
            "chunk_id": chunk_id,
            "section": current_section,
            "page_start": page_start,
            "page_end": page_end,
            "text": text
        })
        chunk_id += 1
        # overlap
        if overlap > 0 and len(text) > overlap:
            tail = text[-overlap:]
            buf = [tail]
            buf_len = len(tail)
        else:
            buf, buf_len = [], 0
        page_start = None
        page_end = None

    current_section = None
    for b in blocks:
        if current_section != b["section_path"] and buf_len:
            flush()
        current_section = b["section_path"]
        if page_start is None:
            page_start = b["page"]
        page_end = b["page"]
        t = b["text"]
        if buf_len + len(t) + 2 > target_chars:
            flush()
            current_section = b["section_path"]
            page_start = b["page"]
            page_end = b["page"]
        buf.append(t)
        buf_len += len(t) + 2
    flush()
    return chunks
