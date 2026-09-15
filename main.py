def load_document(file_path):
    with open(file_path, "r") as file:
        return file.read()


def create_chunks(text):
    chunks = text.split("\n\n")
    return chunks

text = load_document("data/vacation_policy.txt")
chunks = create_chunks(text)

for chunk in chunks:
    print("CHUNK:")
    print(chunk)
    print("------------")