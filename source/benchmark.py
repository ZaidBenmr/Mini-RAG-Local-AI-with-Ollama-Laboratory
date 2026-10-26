"""Benchmark local Ollama models on the same prompt."""
import time, requests, psutil, pandas as pd

OLLAMA = "http://localhost:11434"
PROMPT = "Write a 500-word explanation of reinforcement learning."


def same_model(name, model):
    return name == model or name == f"{model}:latest"


def unload(model):
    requests.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": 0})
    time.sleep(2)


def loaded_info(model):
    for m in requests.get(f"{OLLAMA}/api/ps").json()["models"]:
        if same_model(m["name"], model):
            return m["size"] / 1e9, m.get("size_vram", 0) / 1e9
    return float("nan"), float("nan")


def disk_size(model):
    for m in requests.get(f"{OLLAMA}/api/tags").json()["models"]:
        if same_model(m["name"], model):
            return m["size"] / 1e9
    return float("nan")


def run(model, prompt):
    payload = {"model": model, "prompt": prompt, "stream": False, "think": False}
    r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=900)
    if r.status_code == 400 and "think" in r.text.lower():      # model without 'thinking'
        payload.pop("think")
        r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=900)
    r.raise_for_status()
    return r.json()


def bench(model, prompt=PROMPT, runs=1):
    rows = []
    for _ in range(runs):
        unload(model)                       # cold start, so the load time is visible
        ram_before = psutil.virtual_memory().used / 1e9
        t0 = time.perf_counter()
        r = run(model, prompt)
        wall = time.perf_counter() - t0
        total_gb, vram_gb = loaded_info(model)
        eval_s = r.get("eval_duration", 0) / 1e9
        rows.append({
            "Model": model,
            "Disk size (GB)": disk_size(model),
            "Loaded size (GB)": total_gb,
            "VRAM (GB)": vram_gb,
            "Load time (s)": r.get("load_duration", 0) / 1e9,
            "Response time (s)": wall,
            "Tokens": r.get("eval_count"),
            "Tokens/s": r.get("eval_count", 0) / eval_s if eval_s else float("nan"),
            "System RAM used (%)": psutil.virtual_memory().percent,
            "RAM increase (GB)": psutil.virtual_memory().used / 1e9 - ram_before,
            "response": r.get("response", ""),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    df = pd.concat([bench(m) for m in sys.argv[1:]]).drop(columns="response")
    print(df.round(2).to_string(index=False))
