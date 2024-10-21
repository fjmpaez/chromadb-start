import argparse
import pathlib

import chromadb
import dotenv
import os

from chromadb.utils import embedding_functions


def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--openai_embedding', action='store_true', help='Use OpenAI embeddings')
    args = parser.parse_args()

    base_dir = pathlib.Path(__file__).parent.resolve()
    dotenv.load_dotenv(dotenv_path=f"{base_dir}/.env")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    collection_name = "recipes_default" if not args.openai_embedding else "recipes_openai"

    embedding_function = embedding_functions.DefaultEmbeddingFunction() if not args.openai_embedding else embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key, model_name="text-embedding-3-small"
    )
    collection = chroma_client.get_collection(collection_name, embedding_function=embedding_function)

    while True:
        user_input = input(f"({collection_name})-> Cliente: ")
        if user_input.lower() == 'exit':
            break

        query_result = collection.query(query_texts=[user_input], n_results=3)

        if len(query_result["documents"][0]) == 0:
            print("No similar documents found. Try Again!")
            continue

        print(
            f"Found documents:"
        )

        for ids, document in enumerate(query_result["documents"][0]):
            doc_id = query_result["ids"][0][ids]
            distance = query_result["distances"][0][ids]

            print(
                f"- ID: {doc_id}, Distance: {distance}, Text: {document[:150]}..."
            )


if __name__ == '__main__':
    main()
