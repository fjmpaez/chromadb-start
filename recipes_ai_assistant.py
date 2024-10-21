import argparse

import chromadb
import dotenv
import os

from chromadb.utils import embedding_functions
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

system_prompt = """
You are a helpful recipes assistant designed to help clients to find recipes.
You must always provide recipes from context. If you can't find a proper recipe, just answer you don't have the recipe.
The result must be one only recipe.
"""


def main():
    parser = argparse.ArgumentParser(description="AI recipes assistant.")
    parser.add_argument('--openai_embedding', action='store_true', help='Use OpenAI embeddings')
    args = parser.parse_args()

    dotenv.load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    collection_name = "recipes_default" if not args.openai_embedding else "recipes_openai"
    embedding_function = embedding_functions.DefaultEmbeddingFunction() if not args.openai_embedding else OpenAIEmbeddings(
        api_key=openai_api_key, model="text-embedding-3-small"
    )

    db = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_function
    )

    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="output"
    )

    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")
    tools = [
        create_retriever_tool(
            db.as_retriever(),
            "recipes",
            "Searches and returns recipes."
        )
    ]

    agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True,
                                                  system_message=SystemMessage(content=system_prompt))

    while True:
        user_input = input(f"({collection_name})-> Cliente: ")
        if user_input.lower() == 'exit':
            break

        response = agent.invoke({"input": user_input})
        print(f"AI: {response['output']}")


if __name__ == '__main__':
    main()
