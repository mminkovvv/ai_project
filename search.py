import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as file:
                content = file.read()
                documents.append(content)

    return documents


def create_chunks(text):
    chunks = text.split("\n\n")
    return chunks


def search(question, chunks, embeddings, model):
    question_embedding = model.encode([question])

    scores = cosine_similarity(question_embedding, embeddings)[0]

    top_indices = scores.argsort()[-3:][::-1]

    top_chunks = [chunks[i] for i in top_indices]

    return top_chunks


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = load_documents("data")

chunks = []

for document in documents:
    chunks.extend(create_chunks(document))

embeddings = model.encode(chunks)

question = input("Ask a question: ")

results = search(question, chunks, embeddings, model)

context = "\n\n".join(results)

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6-luna",
    input=f"""
You are an assistant that answers questions using company documents.

Rules:
1. Answer the user's question using ONLY the information provided in the context.
2. Do not use outside knowledge.
3. Do not make up or assume information.
4. If the context does not contain enough information to answer the question, say:
   "I don't have enough information in the provided documents to answer this question."

Context:
{context}

Question:
{question}
"""
)

print("\nAnswer:")
print(response.output_text)