import os, json, requests

class LLMClient:
    """
    Minimal OpenAI-compatible client for vLLM / provider endpoints.
    Expects env:
      - LLM_BASE_URL (e.g., http://127.0.0.1:8000/v1)
      - LLM_API_KEY
      - LLM_MODEL
    """
    def __init__(self, base_url=None, api_key=None, model=None, timeout=120):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "local")
        self.model = model or os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.timeout = timeout

    def chat_json(self, system, user, temperature=0.0, top_p=0.1, seed=7, schema=None):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature, "top_p": top_p
        }
        # Many providers ignore seed, but include it for determinism where supported
        try:
            body["seed"] = seed
        except Exception:
            pass
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        else:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
