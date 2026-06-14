import ollama

response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "What is artificial intelligence?"
        }
    ]
)

print(response["message"]["content"])