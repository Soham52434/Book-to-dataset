import regex as re
from typing import List, Dict

HEADING_RE = re.compile(r"^(?:[A-Z][A-Z0-9 ,;:'\"()/-]{3,}|[0-9]+(?:\.[0-9]+)*[^\S\r\n].{3,})$")

def detect_headings(pages: List[Dict]):
    """
    Splits pages into sections with simple heading heuristics.
    Output: list of blocks: {section_path, page, text}
    """
    blocks = []
    current_section = ["Root"]
    for p in pages:
        for para in re.split(r"\n{2,}", p["text"]):
            line0 = para.strip().split("\n", 1)[0] if para.strip() else ""
            if line0 and HEADING_RE.match(line0):
                # new section
                current_section = ["Root", line0.strip()[:120]]
                continue
            if para.strip():
                blocks.append({
                    "section_path": " / ".join(current_section),
                    "page": p["page_num"],
                    "text": para.strip()
                })
    return blocks
