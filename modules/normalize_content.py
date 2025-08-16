import regex as re

def _strip_headers_footers(pages, threshold=0.6):
    """
    Heuristic: find repeated first/last lines across pages and remove them.
    """
    first_lines, last_lines = [], []
    for p in pages:
        lines = [ln.strip() for ln in p["text"].splitlines() if ln.strip()]
        if not lines: 
            first_lines.append("")
            last_lines.append("")
            continue
        first_lines.append(lines[0])
        last_lines.append(lines[-1])
    def common(items):
        from collections import Counter
        cnt = Counter(items)
        if not cnt: return set()
        max_freq = max(cnt.values())
        return {it for it, c in cnt.items() if c >= max_freq * threshold and it}
    common_first = common(first_lines)
    common_last = common(last_lines)

    new_pages = []
    for p in pages:
        lines = [ln for ln in p["text"].splitlines()]
        pruned = []
        for idx, ln in enumerate(lines):
            if idx == 0 and ln.strip() in common_first: 
                continue
            if idx == len(lines)-1 and ln.strip() in common_last:
                continue
            pruned.append(ln)
        txt = "\n".join(pruned)
        new_pages.append({**p, "text": txt})
    return new_pages

def _fix_hyphenation(text: str) -> str:
    # join hyphenated line-breaks: e.g., "informa-\ntion" -> "information"
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

def normalize_pages(pages):
    out = []
    pages = _strip_headers_footers(pages)
    for p in pages:
        txt = p["text"]
        txt = txt.replace("\r", "\n")
        txt = _fix_hyphenation(txt)
        # collapse extra spaces
        txt = re.sub(r"[ \t]+", " ", txt)
        # collapse excessive newlines
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        out.append({**p, "text": txt.strip()})
    return out
