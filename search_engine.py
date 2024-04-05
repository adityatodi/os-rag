from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from llama_index.core import (
    VectorStoreIndex, StorageContext
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import ServiceContext
from model import get_embed_model, get_llm

def get_query_engine():
    # initialize connection to pinecone
    api_key = os.environ["PINECONE_API_KEY"]
    index_name = "os-helper"
    pc = Pinecone(api_key=api_key)

    # connect to the index
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # create a StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # setup a ServiceContext
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        chunk_overlap=64,
        llm=get_llm(),
        embed_model=get_embed_model()
    )

    # Create a VectorStoreIndex
    index = VectorStoreIndex([], storage_context=storage_context, service_context=service_context)
    query_engine = index.as_query_engine()
    return query_engine