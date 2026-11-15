import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"  # change to the name from `ollama list`

def ollama_llm(messages, tools=None):
    payload = {
        "model": "qwen3:0.6b",
        "messages": messages,
        "stream": False,
    }
    if tools:
        # Ollama expects tools in the OpenAI-style "function" format
        payload["tools"] = [
            {"type": "function",
             "function": {"name": t["name"],
                          "description": t.get("description", ""),
                          "parameters": t["schema"]}}
            for t in tools
        ]

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    msg = data["message"]

    return {
        "content": msg.get("content", ""),
        "tool_calls": [
            {"id": str(i), "name": tc["function"]["name"],
             "arguments": tc["function"]["arguments"]}
            for i, tc in enumerate(msg.get("tool_calls", []) or [])
        ],
        "tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    }