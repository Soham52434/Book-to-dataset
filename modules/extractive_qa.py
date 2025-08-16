import json

QA_SCHEMA = {
  "type":"object",
  "properties":{
    "items":{"type":"array","items":{
      "type":"object",
      "properties":{
        "question":{"type":"string"},
        "answer":{"type":"string"},
        "quotes":{"type":"array","items":{
          "type":"object",
          "properties":{
            "page_start":{"type":"integer"},
            "page_end":{"type":"integer"},
            "text":{"type":"string"}
          },
          "required":["page_start","page_end","text"]
        }}
      },
      "required":["question","answer","quotes"]
    }},
    "count":{"type":"integer"}
  },
  "required":["items"]
}

SYSTEM = """You are an extractive assistant. Use only the provided context.
Return JSON conforming to schema. Provide 3-7 Q/A pairs, each supported by 1-3 verbatim quotes with page spans."""

PROMPT = """Create 3-7 useful Q/A pairs for study/retrieval from the context below.
Rules:
- Extractive only: answers should be paraphrases strictly supported by quotes.
- Provide 1-3 quotes per item with (page_start, page_end, text).
Context:
[pages {p0}-{p1}]
{text}
"""

def make_qa(llm, cache, chunk):
    p0, p1 = chunk.get("page_start"), chunk.get("page_end")
    ctx = chunk["text"]
    prompt = PROMPT.format(p0=p0, p1=p1, text=ctx[:8000])
    key = {"task":"qa","chunk_id":chunk["chunk_id"]}
    cached = cache.get(key)
    if cached: return cached
    out = llm.chat_json(SYSTEM, prompt, temperature=0.0, top_p=0.1, seed=7, schema=QA_SCHEMA)
    # minimal verifier: ensure quotes appear in ctx
    items = []
    for it in out.get("items", []):
        quotes = [q for q in it.get("quotes", []) if q.get("text","") and q["text"][:200] in ctx]
        if quotes:
            items.append({"question": it.get("question",""), "answer": it.get("answer",""), "quotes": quotes})
    result = {"items": items, "count": len(items)}
    cache.put(key, result)
    return result
