import json, regex as re

SECTION_SCHEMA = {
  "type":"object",
  "properties":{
    "blocks":{"type":"array","items":{
      "type":"object",
      "properties":{
        "section_path":{"type":"string"},
        "page":{"type":"integer"},
        "text":{"type":"string"}
      },
      "required":["section_path","page","text"]
    }},
    "dropped_count":{"type":"integer"}
  },
  "required":["blocks"]
}

SYSTEM = """You segment book paragraphs into a hierarchy.
Rules:
- Use only the given paragraphs; do not invent or drop text.
- Keep section_path compact like 'Root / Chapter 1 / 1.1 Overview'.
- Preserve paragraph order. Return strictly JSON in the provided schema."""

PROMPT_TMPL = """You receive page-numbered paragraphs.
Goal: assign each paragraph to a section_path like "Root / Chapter 1 / 1.1 Overview".
Do not invent text. Do not drop paragraphs.

Input (JSON lines):
{payload}
"""

def llm_sections(llm, cache, page_paras):
    out_blocks = []
    # batch by 40 paragraphs to keep prompts small
    for i in range(0, len(page_paras), 40):
        batch = page_paras[i:i+40]
        payload = "\n".join(json.dumps(x, ensure_ascii=False) for x in batch)
        prompt = PROMPT_TMPL.format(payload=payload)
        key = {"task":"sections","batch": batch}
        cached = cache.get(key)
        if cached:
            out = cached
        else:
            out = llm.chat_json(SYSTEM, prompt, temperature=0.0, top_p=0.1, seed=7, schema=SECTION_SCHEMA)
            cache.put(key, out)
        out_blocks.extend(out["blocks"])
    return out_blocks
