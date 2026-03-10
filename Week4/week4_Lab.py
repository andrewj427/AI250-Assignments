import requests
import json
import sys

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

def query_ollama(prompt: str):
    """Send a prompt to the local Ollama model and return the response."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("\n Error: Could not connect to Ollama.")
        print("Make sure Ollama is running: run 'ollama serve' in a separate terminal.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("\n Error: Request timed out. The model may be loading, try again.")
        sys.exit(1)


def get_code_input():
    """Prompt the user to paste code, ended by a sentinel line."""
    print("\n Paste your Python code below.")
    print("When you're done, type 'END' on a new line and press Enter:\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def explain_code(code: str):
    prompt = f"""You are an expert Python developer and teacher.

Given the following Python code, provide:
1. A plain English explanation of what the code does overall (2-4 sentences)
2. A step-by-step breakdown of the key parts

Be clear and concise. Assume the reader is a beginner.

Code:
```python
{code}
```"""
    return query_ollama(prompt)


def comment_code(code: str):
    prompt = f"""You are an expert Python developer.

Add helpful inline comments to the following Python code. 
- Add a comment above or beside each important line or block
- Keep comments brief and clear
- Return ONLY the commented code, no extra explanation

Code:
```python
{code}
```"""
    return query_ollama(prompt)


def print_section(title: str, content: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(content)


def main():
    print("=" * 60)
    print("  Python Code Explainer — Powered by Ollama (llama3)")
    print("=" * 60)

    code = get_code_input()

    if not code.strip():
        print(" No code entered. Exiting.")
        sys.exit(1)

    print("\n Analyzing your code with llama3... (this may take a moment)\n")

    explanation = explain_code(code)
    print_section(" Plain English Explanation", explanation)

    print("\n Adding inline comments...\n")
    commented = comment_code(code)
    print_section(" Code With Inline Comments", commented)

    print("\n" + "=" * 60)
    print("   Done!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()