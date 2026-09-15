from sentence_transformers import SentenceTransformer

def load_document(file_path):
    with open(file_path, "r") as file:
        return file.read()

def create_chunks(text):
    return  text.split("\n\n")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

text = load_document("data/vacation_policy.txt")

chunks = create_chunks(text)

embeddings = model.encode(chunks)

for i, embedding in enumerate(embeddings):
    print(f"Chunk {i + 1}: ")
    print(chunks[i])
    print("Embedding size:",len(embedding))
    print()


