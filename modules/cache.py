import json, hashlib
from pathlib import Path

class DiskCache:
    def __init__(self, root="data/work/llm_cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, obj):
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest() + ".json"

    def get(self, prompt_obj):
        p = self.root / self._key(prompt_obj)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def put(self, prompt_obj, result):
        p = self.root / self._key(prompt_obj)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
