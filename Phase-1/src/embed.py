import ollama

response = ollama.embeddings(
    model="nomic-embed-text",
    prompt="Artificial Intelligence is transforming research."
)

embedding = response["embedding"]

print(f"Vector Length: {len(embedding)}")
print(embedding[:10])