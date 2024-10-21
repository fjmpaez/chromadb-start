import os
import pathlib

import dotenv
from chromadb.utils import embedding_functions

from chroma_functions import add_open_ai_embeddings, get_persistent_client

from chroma_functions import load_documents, split_documents
import argparse

if __name__ == '__main__':
    base_dir = pathlib.Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('source', type=str, help='Use OpenAI embeddings')
    parser.add_argument('--openai_embedding', action='store_true', help='Use OpenAI embeddings')
    args = parser.parse_args()

    dotenv.load_dotenv()

    documents = load_documents(args.source)
    chunked_documents = split_documents(documents, size=2000, overlap=20)

    if args.openai_embedding:
        chunked_documents = add_open_ai_embeddings(chunked_documents, openai_api_key=os.getenv("OPENAI_API_KEY"))

    chroma_client = get_persistent_client(database_folder=f"{base_dir}/db/recipes")

    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection_name = "recipes_default" if not args.openai_embedding else "recipes_openai"
    collection = chroma_client.get_or_create_collection(collection_name, embedding_function=default_ef)

    for doc in chunked_documents:
        embeddings = [doc["embedding"]] if args.openai_embedding else None
        print("*" * 10)
        print(f"Inserting document {doc["id"]} \ntext: {doc["text"]}  \nembeddings: {embeddings}")
        print("*" * 10)
        collection.upsert(ids=[doc["id"]], documents=[doc["text"]], embeddings=embeddings)
