import os
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()

paragraph = ("Artificial intelligence and cloud computing work together to power modern digital systems by enabling scalable data processing, rapid model training, and global deployment of intelligent applications. AI relies on the clouds vast storage, flexible compute resources, and managed services to analyze large datasets, automate tasks, and support realtime decision making across industries such as healthcare, finance, retail, and transportation. Cloud platforms make AI accessible to organizations of all sizes by offering prebuilt models, APIs, and tools that reduce complexity. As both technologies evolve, their integration continues to accelerate innovation, improve efficiency, and create new opportunities for smarter, datadriven solutions.")

prompt = f"Analyze the emotion / tone of this paragraph: \n\n {paragraph}"

def timed(label, fn):
    start = time.perf_counter()

    try:
        result = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return{"provider":label, "ok":True, "latency":round(elapsed_ms, 1), "result":result}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return{"provider":label, "ok":False, "latency":round(elapsed_ms, 1), "error": str(e)}
    

#summarize
def summarize_HF():
    token = os.getenv("HF_Token")
    model = os.getenv("HF_Model", "facebook/bart-large-cnn")

    #HF uses a router endpoint
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": paragraph}

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    #summarize the return
    if isinstance(data, list) and data and "summary_text" in data[0]:
        return data[0]["summary_text"]
    return data


def main():
    tests = []
    if os.getenv("HF_Token"):
        tests.append(("Hugging Face Summarization", summarize_HF))
    else:
        print("Skipping Hugging Face")

    print("\nInput paragraph: \n", paragraph, "\n")

    results = [timed(label, fn) for label, fn in tests]

    for r in results:
        print(f" == {r['provider']} == ")
        print(f"OK: {r['ok']} Latency_ms: {r['latency']}")
        if r["ok"]:
            print("Summary/Output:")
            print(r['result'] if isinstance(r['result'], str) else json.dumps(r['result'], indent=2))
        else:
            print("Error: ", r['error'])
        
        print()
if __name__ == "__main__":
    main()