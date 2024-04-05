from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
import os
import logging
import sys
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    LangchainEmbedding, 
    ServiceContext
)
import torch
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_embeddings_model():
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    )
    return embed_model

def get_llm():
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q8_0.gguf',
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 16},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm


def main():
    load_dotenv()
    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.getenv("PINECONE_INDEX_KEY")
    pc = Pinecone(api_key=api_key)
    pinecone_index = pc.Index(index_name=index_name)
    
    # load documents
    documents = SimpleDirectoryReader("./data").load_data()
    llm = get_llm()

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    loader = PyPDFDirectoryLoader('./data')
    pdf_docs = loader.load()
    
    #get the text chunks
    chunks = get_text_chunks(pdf_docs)
    print(chunks)

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )
    
    Pinecone.from_text(chunks, index_name=index_name)

    
    


if __name__ == "__main__":
    main()