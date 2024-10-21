from sympy.physics.units import power


def get_persistent_client(database_folder):
    import chromadb
    return chromadb.PersistentClient(path=database_folder)


def get_http_client(host='localhost', port=8000):
    import chromadb
    return chromadb.HttpClient(host=host, port=port)


def load_documents(source):
    docs = []
    import os
    for filename in os.listdir(source):
        with open(
                os.path.join(source, filename), "r", encoding="utf-8"
        ) as file:
            docs.append({"id": filename, "text": file.read()})
    return docs


def split_on_chunks(text, size=1000, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def split_documents(documents, size=1000, overlap=20):
    chunked_documents = []
    for doc in documents:
        chunks = split_on_chunks(doc["text"], size, overlap)

        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i + 1}", "text": chunk})

    return chunked_documents


def add_open_ai_embeddings(chunked_documents, openai_api_key):
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

    for chunked_doc in chunked_documents:
        response = client.embeddings.create(input=chunked_doc["text"], model="text-embedding-3-small")
        print(f"OpenAI-->{response}")
        chunked_doc["embedding"] = response.data[0].embedding

    return chunked_documents
