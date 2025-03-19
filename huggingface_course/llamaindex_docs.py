import argparse
from llama_index.embeddings.ollama import OllamaEmbedding
import dotenv
import pandas as pd
from rich import print
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
import chromadb

# add current directory to working directory
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
dotenv.load_dotenv()


def setup_chroma():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("user_data")
    store = ChromaVectorStore(chroma_collection=chroma_collection)
    return store


def setup_docs():
    docs: pd.DataFrame = pd.read_csv("huggingface_course/data/fakeuser.csv")
    docs = [
        Document(text=row.to_string(), metadata=row.to_dict())
        for _, row in docs.iterrows()
    ]
    return docs


def setup_embed_model():
    embed_model = OllamaEmbedding(
        model_name="bge-m3:latest",
        api_base="http://localhost:11434",
        api_key="ollama",
    )
    return embed_model


def run_pipeline(docs, vector_store, embed_model):
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=200, chunk_overlap=0),
            embed_model,
        ],
        vector_store=vector_store,
    )

    nodes = pipeline.run(documents=docs)
    print("embedding length: ", len(nodes[0].embedding))
    print("nodes sample: ", nodes[0])


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true")
    args = parser.parse_args()

    vector_store = setup_chroma()
    embed_model = setup_embed_model()
    if args.setup:
        print("setting up vector store")
        docs = setup_docs()
        print("sample doc: ", docs[0])
        run_pipeline(docs, vector_store, embed_model)
    else:
        print("Using existing vector store")
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        llm = OpenAI(model="gpt-4o-mini")
        query_engine = index.as_query_engine(
            llm=llm,
            response_mode="tree_summarize",
        )

        response = query_engine.query(
            " User has to have a same name. If you dont have info. return dont have info.\
            Give all info about of Isabella"
        )
        print(response.response)
