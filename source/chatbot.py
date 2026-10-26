"""Console chatbot with conversation memory (uses /api/chat)."""
import os
import requests

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MODEL", "qwen3.5:4b")


def call(messages):
    payload = {"model": MODEL, "messages": messages, "stream": False, "think": False}
    r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=900)
    if r.status_code == 400 and "think" in r.text.lower():      # model without 'thinking'
        payload.pop("think")
        r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=900)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    messages = [{"role": "system", "content": "You are a helpful, concise assistant."}]
    print("Local AI Assistant — type 'exit' to stop.")
    while True:
        q = input("\nYou: ")
        if q.lower() == "exit":
            break
        messages.append({"role": "user", "content": q})
        reply = call(messages)
        messages.append({"role": "assistant", "content": reply})
        print("\nAI:", reply)


if __name__ == "__main__":
    main()
