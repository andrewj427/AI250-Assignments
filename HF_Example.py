import os
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()

paragraph = (
    "Artificial intelligence and cloud computing work together to power modern digital systems by enabling scalable data processing, rapid model training, and global deployment of intelligent applications. "
    "AI relies on the clouds vast storage, flexible compute resources, and managed services to analyze large datasets, automate tasks, and support realtime decision making across industries such as healthcare, finance, retail, and transportation. "
    "Cloud platforms make AI accessible to organizations of all sizes by offering prebuilt models, APIs, and tools that reduce complexity. "
    "As both technologies evolve, their integration continues to accelerate innovation, improve efficiency, and create new opportunities for smarter, datadriven solutions."
)


writing_style = "explaining to a baby"


prompt = f"Rewrite the following paragraph in {writing_style}. Only return the rewritten paragraph, nothing else.\n\nParagraph:\n{paragraph}"

formatted_prompt = prompt


def timed(label, fn):
    start = time.perf_counter()
    try:
        result = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"provider": label, "ok": True, "latency": round(elapsed_ms, 1), "result": result}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"provider": label, "ok": False, "latency": round(elapsed_ms, 1), "error": str(e)}


def rewrite_HF():
    token = os.getenv("HF_Token")
    model = "google/flan-t5-large"

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "return_full_text": False  # Only return the generated text, not the input prompt
        }
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    return data


def main():
    token = os.getenv("HF_Token")

    if not token:
        print("ERROR: HF_Token not found in .env file")
        return

    print("Token loaded: OK")
    print(f"\nWriting style: {writing_style}")
    print(f"\nInput paragraph:\n{paragraph}\n")

    result = timed("Hugging Face (Mistral-7B)", rewrite_HF)

    print(f"== {result['provider']} ==")
    print(f"OK: {result['ok']}  Latency_ms: {result['latency']}")
    if result["ok"]:
        print("\nRewritten Output:")
        print(result["result"] if isinstance(result["result"], str) else json.dumps(result["result"], indent=2))
    else:
        print("Error:", result["error"])


if __name__ == "__main__":
    main()