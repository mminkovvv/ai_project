from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

text = "Employees recieve 20 days of paid annual leave per year."

embedding = model.encode(text)

print(embedding)
print(len(embedding))