import os
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rich import print
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from sentence_transformers import CrossEncoder
from reranking_models import reranking_cohere, reranking_colbert, reranking_gpt, reranking_german

import fitz  # PyMuPDF
from langchain.docstore.document import Document

import docx

import openpyxl
from langchain_milvus import Milvus
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from pathlib import Path
from openai import AzureOpenAI
from pymilvus import connections, Collection, utility
from langchain.chains import HypotheticalDocumentEmbedder
from tqdm import tqdm

from langchain_community.storage.mongodb import MongoDBStore
from pymongo import MongoClient

# Al principio del archivo, después de las importaciones
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME")

def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if not file.startswith('~$'):
            file_path = os.path.join(folder_path, file)
            if file.lower().endswith('.pdf'):
                documents.extend(load_pdf(file_path))
            elif file.lower().endswith('.docx'):
                documents.extend(load_docx(file_path))
            elif file.lower().endswith('.xlsx'):
                documents.extend(load_xlsx(file_path))
    return documents


def load_xlsx(file_path):
    """
    Loads text from an XLSX file, including sheet names in metadata.
    """
    wb = openpyxl.load_workbook(file_path)
    documents = []
    for sheet_num, sheet_name in enumerate(wb.sheetnames):
        ws = wb[sheet_name]
        text = ""
        for row in ws.iter_rows(values_only=True):
            text += " ".join([str(cell) for cell in row if cell is not None]) + "\n"
        text = clean_extra_whitespace(text)
        text = group_broken_paragraphs(text)
        if text:  # Only add if the text is not empty
            document = Document(
                page_content=text,
                metadata={"source": file_path, "sheet": sheet_name, "page": sheet_num + 1}
            )
            documents.append(document)

    # Agregado
    # print ("------------------------------------------------------------------------------>>>\nXlsx documents:")
    # print (documents)

    return documents


import fitz
import re
from typing import List
from langchain.schema import Document

def load_pdf(file_path: str) -> List[Document]:
    """
    Carga documentos de un archivo PDF usando PyMuPDF, divide en páginas y crea objetos Document con overlap.
    """
    try:
        doc = fitz.open(file_path)
        pages = []
        documents = []
        overlap_words = 40
        
        # Extraer y limpiar el texto de cada página
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            text = clean_extra_whitespace(text)
            text = group_broken_paragraphs(text)
            pages.append(text)
        
        # Crear objetos Document con overlap
        for i, page_content in enumerate(pages):
            content_parts = []
            
            # Agregar overlap antes
            if i > 0:
                prev_page_words = pages[i-1].split()
                content_parts.append(" ".join(prev_page_words[-overlap_words:]))
            
            # Agregar contenido de la página actual
            content_parts.append(page_content)
            
            # Agregar overlap después
            if i < len(pages) - 1:
                next_page_words = pages[i+1].split()
                content_parts.append(" ".join(next_page_words[:overlap_words]))
            
            # Crear objeto Document
            doc = Document(
                page_content=" ".join(content_parts),
                metadata={"source": file_path, "page": i + 1}
            )
            documents.append(doc)
        
        # print("------------------------------------------------------------------------------>>>\nPDF documents:")
        # print(documents)
        
        return documents
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        raise


import docx2txt
import re
from typing import List
from langchain.schema import Document

def split_into_pages(text: str) -> List[str]:
    """
    Divide el texto en páginas basándose en saltos de página.
    """
    return text.split('\f')

def load_docx(file_path: str) -> List[Document]:
    """
    Carga texto de un archivo DOCX, lo divide en páginas y crea objetos Document con overlap.
    """
    # Extraer texto del archivo DOCX
    text = docx2txt.process(file_path)
    
    # Limpiar y agrupar párrafos
    text = clean_extra_whitespace(text)
    text = group_broken_paragraphs(text)
    
    # Dividir en páginas
    pages = split_into_pages(text)
    
    documents = []
    overlap_words = 40

    for i, page_content in enumerate(pages):
        # Preparar el contenido con overlap
        content_parts = []
        
        # Agregar overlap antes
        if i > 0:
            prev_page_words = pages[i-1].split()
            content_parts.append(" ".join(prev_page_words[-overlap_words:]))
        
        # Agregar contenido de la página actual
        content_parts.append(page_content)
        
        # Agregar overlap después
        if i < len(pages) - 1:
            next_page_words = pages[i+1].split()
            content_parts.append(" ".join(next_page_words[:overlap_words]))
        
        # Crear objeto Document
        doc = Document(
            page_content=" ".join(content_parts),
            metadata={"source": file_path, "page": i + 1}
        )
        documents.append(doc)
    
    # print("------------------------------------------------------------------------------>>>\nDocx documents:")
    # print(documents)
    
    return documents


def split_documents(
    chunk_size: int,
    knowledge_base,
    tokenizer_name= EMBEDDING_MODEL_NAME,
):
    """
    Splits documents into chunks of maximum size `chunk_size` tokens, using a specified tokenizer.
    
    Parameters:
    - chunk_size: The maximum number of tokens for each chunk.
    - knowledge_base: A list of LangchainDocument objects to be split.
    - tokenizer_name: (Optional) The name of the tokenizer to use. Defaults to `EMBEDDING_MODEL_NAME`.
    
    Returns:
    - A list of LangchainDocument objects, each representing a chunk. Duplicates are removed based on `page_content`.
    
    Raises:
    - ImportError: If necessary modules for tokenization are not available.
    """

    if tokenizer_name=="openai":
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=["\n\n\n", "\n\n", "\n", ".", ""],
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            model_name="gpt-4",
            is_separator_regex=False,
            add_start_index=True,
            strip_whitespace=True,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
        )

    docs_processed = (text_splitter.split_documents([doc]) for doc in knowledge_base)

    print(f">>> Numero de chunks: {len(docs_processed)}")

    # Flatten list and remove duplicates more efficiently
    unique_texts = set()
    docs_processed_unique = []
    for doc_chunk in docs_processed:
        for doc in doc_chunk:
            if doc.page_content not in unique_texts:
                unique_texts.add(doc.page_content)
                docs_processed_unique.append(doc)

    return docs_processed_unique


def load_embedding_model(
    model_name = EMBEDDING_MODEL_NAME, 
):
    """
    Loads an embedding model from the Hugging Face repository with specified configurations.

    Parameters:
    - model_name: The name of the model to load. Defaults to "BAAI/bge-large-en-v1.5".
    - device: The device to run the model on (e.g., 'cpu', 'cuda', 'mps'). Defaults to 'mps'.

    Returns:
    - An instance of HuggingFaceBgeEmbeddings configured with the specified model and device.

    Raises:
    - ValueError: If an unsupported device is specified.
    - OSError: If the model cannot be loaded from the Hugging Face repository.
    """

    try:
        if model_name=="openai":
             embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            model_name=EMBEDDING_MODEL_NAME 
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            model_kwargs = {"device": device}
            encode_kwargs = {"normalize_embeddings": True}  # For cosine similarity computation

            embedding_model = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        return embedding_model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise


def retrieve_context(query, retriever):
    """
    Retrieves and reranks documents relevant to a given query.

    Parameters:
    - query: The search query as a string.
    - retriever: An instance of a Retriever class used to fetch initial documents.

    Returns:
    - A list of reranked documents deemed relevant to the query.

    """

    retrieved_docs = retriever.invoke(input=query)

    return retrieved_docs


def get_fusion_retriever(docs, embedding_model, collection_name="test", top_k=3):
    """
    Initializes a retriever object to fetch the top_k most relevant documents based on cosine similarity.

    Parameters:
    - docs: A list of documents to be indexed and retrieved.
    - embedding_model: The embedding model to use for generating document embeddings.
    - top_k: The number of top relevant documents to retrieve. Defaults to 3.

    Returns:
    - A retriever object configured to retrieve the top_k relevant documents.

    Raises:
    - ValueError: If any input parameter is invalid.
    """

    # Hybrid search
    # Example of parameter validation (optional)
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    try:
        vector_store = Milvus.from_documents(
            documents=docs, 
            embedding=embedding_model,
            collection_name=collection_name,
        )

        retriever = vector_store.as_retriever(search_kwargs={"k":top_k})
        # retriever.k = top_k

        # add keyword search 
        keyword_retriever = BM25Retriever.from_documents(docs)
        keyword_retriever.k =  3

        ensemble_retriever = EnsembleRetriever(retrievers=[retriever,
                                                    keyword_retriever],
                                        weights=[0.5, 0.5])

        return ensemble_retriever
    except Exception as e:
        print(f"An error occurred while initializing the retriever: {e}")
        raise


def get_or_create_milvus_collection(docs, embedding_model, collection_name):
    """
    Obtiene una colección existente de Milvus o crea una nueva si no existe.
    Incluye una barra de progreso para mostrar el avance de la inserción de documentos.

    Parameters:
    - docs: Documentos a insertar si se crea una nueva colección.
    - embedding_model: Modelo de embedding a utilizar.
    - collection_name: Nombre de la colección.

    Returns:
    - Una instancia de Milvus (vectorstore).
    """
    # Establecer conexión con Milvus
    connections.connect()

    # Verificar si la colección ya existe
    if utility.has_collection(collection_name):
        print(f"Laden der bestehenden Kollektion in Milvus:'{collection_name}'")
        vectorstore = Milvus(
            collection_name=collection_name,
            embedding_function=embedding_model,
        )
    else:
        print(f"Die Kollektion '{collection_name}' existiert nicht in Milvus. Erstellen und Hinzufügen von Dokumenten...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            auto_id=True  # Agregar esta línea
        )
        all_splits = text_splitter.split_documents(docs)
        docs_processed = []

        # Iterar sobre los documentos y mostrar el progreso
        for doc in tqdm(all_splits, desc="Verarbeitung von Dokumenten"):
            docs_processed.append(doc)

        # Crear el vectorstore con los documentos procesados en lotes
        vectorstore = Milvus.from_documents(
            documents=docs_processed,
            embedding=embedding_model,
            collection_name=collection_name,
        )

    return vectorstore

def get_milvus_collection(embedding_model, collection_name):
    """
    Obtiene una colección existente de Milvus.
    Incluye una barra de progreso para mostrar el avance de la inserción de documentos.

    Parameters:
    - embedding_model: Modelo de embedding a utilizar.
    - collection_name: Nombre de la colección.

    Returns:
    - Una instancia de Milvus (vectorstore).
    """
    # Establecer conexión con Milvus
    connections.connect()

    print(f"Laden der bestehenden Kollektion in Milvus:'{collection_name}'")
    vectorstore = Milvus(
        collection_name=collection_name,
        embedding_function=embedding_model,
        auto_id=True  # Agregar esta línea
    )
    
    return vectorstore


def create_milvus_collection(docs, embedding_model, collection_name):
    """
    Obtiene una colección existente de Milvus o crea una nueva si no existe.
    Incluye una barra de progreso para mostrar el avance de la inserción de documentos.

    Parameters:
    - docs: Documentos a insertar si se crea una nueva colección.
    - embedding_model: Modelo de embedding a utilizar.
    - collection_name: Nombre de la colección.

    Returns:
    - Una instancia de Milvus (vectorstore).
    """
    # Establecer conexión con Milvus
    connections.connect()

    print(f"Die Kollektion '{collection_name}' existiert nicht in Milvus. Erstellen und Hinzufügen von Dokumenten...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        add_start_index=True,
        strip_whitespace=True,
    )
    all_splits = text_splitter.split_documents(docs)
    docs_processed = []

    # Iterar sobre los documentos y mostrar el progreso
    for doc in tqdm(all_splits, desc="Verarbeitung von Dokumenten"):
        docs_processed.append(doc)

    # Crear el vectorstore con los documentos procesados en lotes
    vectorstore = Milvus.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        collection_name=collection_name,
    )

    return vectorstore


def get_ensemble_retriever(docs, embedding_model, llm, collection_name="test", top_k=3):
    """
    Initializes a retriever object to fetch the top_k most relevant documents based on cosine similarity.
    Now includes a HyDE retriever in the ensemble, using Milvus as the vector store.
    Checks for existing collections before creating new ones.

    Parameters:
    - docs: A list of documents to be indexed and retrieved.
    - embedding_model: The embedding model to use for generating document embeddings.
    - llm: Language model to use for HyDE.
    - collection_name: The name of the collection.
    - top_k: The number of top relevant documents to retrieve. Defaults to 3.

    Returns:
    - An ensemble retriever object configured to retrieve the top_k relevant documents.

    Raises:
    - ValueError: If any input parameter is invalid.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    try:
        # Crear o cargar el vectorstore base
        base_vectorstore = get_or_create_milvus_collection(docs, embedding_model, collection_name)
        retriever = base_vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        # Crear el retriever de palabras clave
        keyword_retriever = BM25Retriever.from_documents(docs)
        keyword_retriever.k = top_k

        # Crear el retriever de consultas múltiples
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm
        )

        # Crear o cargar el vectorstore HyDE
        hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
            llm,
            embedding_model,
            "web_search"
        )
        hyde_collection_name = f"{collection_name}_hyde"
        hyde_vectorstore = get_or_create_milvus_collection(docs, hyde_embeddings, hyde_collection_name)
        hyde_retriever = hyde_vectorstore.as_retriever(search_kwargs={"k": top_k})

        # Crear el Parent Document Retriever
        parent_retriever = create_parent_retriever(docs, embedding_model, collection_name, top_k)

        # Crear el ensemble retriever con los cinco retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, keyword_retriever, multi_query_retriever, hyde_retriever, parent_retriever],
            # weights=[0.2, 0.2, 0.2, 0.2, 0.2]  # Pesos iguales para todos los retrievers
            weights=[0.1, 0.1, 0.3, 0.2, 0.3]
        )

        return ensemble_retriever
    except Exception as e:
        print(f"An error occurred while initializing the retriever: {e}")
        raise


def get_ensemble_retriever_check(folder_path, embedding_model, llm, collection_name="test", top_k=3):
    """
    Initializes a retriever object to fetch the top_k most relevant documents based on cosine similarity.
    Now includes a HyDE retriever in the ensemble, using Milvus as the vector store.
    Checks for existing collections before creating new ones.

    Parameters:
    - docs: A list of documents to be indexed and retrieved.
    - embedding_model: The embedding model to use for generating document embeddings.
    - llm: Language model to use for HyDE.
    - collection_name: The name of the collection.
    - top_k: The number of top relevant documents to retrieve. Defaults to 3.

    Returns:
    - An ensemble retriever object configured to retrieve the top_k relevant documents.

    Raises:
    - ValueError: If any input parameter is invalid.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    try:
        # Establecer conexión con Milvus
        connections.connect()

        if utility.has_collection(collection_name):
            # Cargar el vectorstore base
            base_vectorstore = get_milvus_collection(embedding_model, collection_name)

            # Cargar el Keyword Retrieval
            # keyword_retriever = load_bm25(f'{collection_name}_keywords')

            # Cargar el vectorstore HyDE
            hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
                llm,
                embedding_model,
                "web_search"
            )
            hyde_vectorstore = get_milvus_collection(hyde_embeddings, f"{collection_name}_hyde")

            # Crear el Parent Document Retriever
            parent_vectorstore = get_milvus_collection(embedding_model, f"{collection_name}_children")
            parent_retriever = create_parent_retriever(parent_vectorstore, collection_name, top_k)

        else:
            # Cargar los documentos
            docs = load_documents(folder_path=folder_path)
            
            # Crear el vectorstore base
            base_vectorstore = create_milvus_collection(docs, embedding_model, collection_name)

            # Crear el Keyword Retrieval
            # keyword_retriever = create_and_save_bm25(docs, f'{collection_name}_keywords')

            # Crear el vectorstore HyDE
            hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
                llm,
                embedding_model,
                "web_search"
            )
            hyde_vectorstore = create_milvus_collection(docs, hyde_embeddings, f"{collection_name}_hyde")

            # Crear el Parent Document Retriever
            parent_vectorstore = create_milvus_collection(docs, embedding_model, f"{collection_name}_children")
            parent_retriever = create_parent_retriever_from_docs(docs, parent_vectorstore, collection_name, top_k)

        
        # Crear o cargar el vectorstore base
        retriever = base_vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        # Crear el retriever de palabras clave
        # keyword_retriever = BM25Retriever.from_documents(docs)
        keyword_retriever = load_bm25(f'{collection_name}_parents')
        keyword_retriever.k = top_k

        '''
        # Cargar el vectorstore HyDE
        hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
            llm,
            embedding_model,
            "web_search"
        )
        hyde_vectorstore = get_milvus_collection(hyde_embeddings, f"{collection_name}")
        '''

        # Crear el retriever de consultas múltiples
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm
        )

        # Crear o cargar el vectorstore HyDE
        hyde_retriever = hyde_vectorstore.as_retriever(search_kwargs={"k": top_k})

        # Crear el ensemble retriever con los cinco retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, keyword_retriever, multi_query_retriever, hyde_retriever, parent_retriever],
            # weights=[0.2, 0.2, 0.2, 0.2, 0.2]  # Pesos iguales para todos los retrievers
            # weights=[0.1, 0.1, 0.3, 0.2, 0.3]
            weights=[0.2, 0.1, 0.3, 0.1, 0.3]
        )

        return ensemble_retriever
    except Exception as e:
        print(f"An error occurred while initializing the retriever: {e}")
        raise


import pickle
from pymongo import MongoClient
from bson.binary import Binary
from langchain_community.retrievers import BM25Retriever

def get_mongo_collection(collection_name):
    client = MongoClient(MONGODB_CONNECTION_STRING)
    db = client[MONGODB_DATABASE_NAME]
    return db[collection_name]

'''
# Función para crear y guardar el BM25Retriever en MongoDB
def create_and_save_bm25(docs, collection_name):
    print(f"Die Kollektion '{collection_name}' existiert nicht in MongoDB. Erstellen und Hinzufügen von Dokumenten...")
    keyword_retriever = BM25Retriever.from_documents(docs)
    serialized_retriever = Binary(pickle.dumps(keyword_retriever))
    
    keyword_collection = get_mongo_collection(collection_name)
    keyword_collection.update_one(
        {"name": "BM25Retriever"},
        {"$set": {"keyword_retriever": serialized_retriever}},
        upsert=True
    )
    print(f"BM25Retriever guardado en MongoDB.")
    return keyword_retriever
'''
# Función para cargar el BM25Retriever desde MongoDB
def load_bm25(collection_name):
    
    print(f"Laden der bestehenden Kollektion in MongoDB:'{collection_name}' (für Keywords)")
    docstore = MongoDBStore(
        connection_string=MONGODB_CONNECTION_STRING,
        db_name=MONGODB_DATABASE_NAME,
        collection_name=collection_name,
    )
    keys = [key for key in docstore.yield_keys()]
    docs_processed = docstore.mget(keys)

    retriever = BM25Retriever.from_documents(docs_processed)
    # retriever.k = top_k
    
    return retriever
    
    '''
    keyword_collection = get_mongo_collection(collection_name)
    result = keyword_collection.find_one({"name": "BM25Retriever"})
    
    if result and "keyword_retriever" in result:
        keyword_retriever = pickle.loads(result["keyword_retriever"])
        print(f"Laden der bestehenden Kollektion in MongoDB:'{collection_name}'")
        return keyword_retriever
    else:
        print(f"Die Kollektion '{collection_name}' existiert nicht in MongoDB.")
        return None
    '''


def create_multi_query_retriever(base_retriever, llm):
    """
    Create a multi-query retriever based on the base retriever and LLM. 

    Parameters: 
    - base_retriever: base retriever 
    - llm: LLM to generate variations of queries.

    Returns: 
    - A retriever that is able to generate multiple queries. 
    """

    multiquery_retriever = MultiQueryRetriever.from_llm(base_retriever, llm)

    return multiquery_retriever


from langchain_milvus import Milvus
from pymilvus import connections, Collection, utility

def create_parent_retriever(
    vectorstore,
    collection_name, 
    top_k=5,
):
    """
    Initializes a Parent Document Retriever object to fetch the top_k most relevant documents based on cosine similarity.
    Uses MongoDB for persistent storage of parent documents.

    Parameters:
    - docs: A list of documents to be indexed and retrieved.
    - embedding_model: The embedding model to use for generating document embeddings.
    - collection_name: The name of the collection
    - top_k: The number of top relevant documents to retrieve. Defaults to 5.

    Returns:
    - A Parent Document Retriever object configured to retrieve the top_k relevant documents.

    Raises:
    - ValueError: If any input parameter is invalid.
    """

    # Establecer conexión con Milvus
    # connections.connect()

    # Verificar si la colección ya existe en Milvus
    # vectorstore = get_or_create_milvus_collection(docs, embedding_model, f"{collection_name}_children")

    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n\n", "\n\n", "\n", ".", ""],
        chunk_size=512,
        chunk_overlap=0,
        model_name="gpt-4",
        is_separator_regex=False,
    )

    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n\n", "\n\n", "\n", ".", ""],
        chunk_size=256,
        chunk_overlap=0,
        model_name="gpt-4",
        is_separator_regex=False,
    )

    # Crear conexión a MongoDB
    # client = MongoClient(MONGODB_CONNECTION_STRING)
    # db = client[MONGODB_DATABASE_NAME]
    mongo_collection = get_mongo_collection(f'{collection_name}_parents')

    # Verificar si la colección existe en MongoDB
    if mongo_collection.count_documents({}) > 0:
        print(f"Laden der bestehenden Kollektion in MongoDB:'{collection_name}_parents'")
        store = MongoDBStore(
            connection_string=MONGODB_CONNECTION_STRING,
            db_name=MONGODB_DATABASE_NAME,
            collection_name=f'{collection_name}_parents'
        )
    else:
        print(f"Die Kollektion '{collection_name}_parents' existiert nicht in MongoDB. Erstellen und Hinzufügen von Dokumenten...")
        store = MongoDBStore(
            connection_string=MONGODB_CONNECTION_STRING,
            db_name=MONGODB_DATABASE_NAME,
            collection_name=f'{collection_name}_parents'
        )
        # Crear el retriever y agregar documentos
        temp_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            k=top_k,
        )
        
        # Añadir barra de progreso para la carga de documentos en MongoDB
        print("Verarbeitung und Laden von Dokumenten in MongoDB...")
        for i, doc in tqdm(enumerate(docs), total=len(docs), desc="Laden von Dokumenten in MongoDB"):
            # Asegurarse de que los metadatos existan
            if doc.metadata is None:
                doc.metadata = {}
            
            # Agregar 'start_index' y 'doc_id' a los metadatos
            doc.metadata['start_index'] = i
            doc.metadata['doc_id'] = str(i)
            
            # Verificar que el contenido del documento no esté vacío
            if doc.page_content.strip():
                try:
                    # Añadir el documento al retriever
                    temp_retriever.add_documents([doc])
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von Dokument {i}: {str(e)}")
                    print(f"Inhalt des Dokuments: {doc.page_content[:100]}...")  # Imprimir los primeros 100 caracteres
            else:
                print(f"Das Dokument {i} ist leer und wird übersprungen.")

    # Crear el retriever final
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        k=top_k,
    )

    return retriever


def create_parent_retriever_from_docs(
    docs,
    vectorstore,
    collection_name, 
    top_k=5,
):
    """
    Initializes a Parent Document Retriever object to fetch the top_k most relevant documents based on cosine similarity.
    Uses MongoDB for persistent storage of parent documents.

    Parameters:
    - docs: A list of documents to be indexed and retrieved.
    - embedding_model: The embedding model to use for generating document embeddings.
    - collection_name: The name of the collection
    - top_k: The number of top relevant documents to retrieve. Defaults to 5.

    Returns:
    - A Parent Document Retriever object configured to retrieve the top_k relevant documents.

    Raises:
    - ValueError: If any input parameter is invalid.
    """

    # Establecer conexión con Milvus
    # connections.connect()

    # Verificar si la colección ya existe en Milvus
    # vectorstore = get_or_create_milvus_collection(docs, embedding_model, f"{collection_name}_children")

    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n\n", "\n\n", "\n", ".", ""],
        chunk_size=512,
        chunk_overlap=0,
        model_name="gpt-4",
        is_separator_regex=False,
    )

    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n\n", "\n\n", "\n", ".", ""],
        chunk_size=256,
        chunk_overlap=0,
        model_name="gpt-4",
        is_separator_regex=False,
    )

    # Crear conexión a MongoDB
    # client = MongoClient(MONGODB_CONNECTION_STRING)
    # db = client[MONGODB_DATABASE_NAME]
    mongo_collection = get_mongo_collection(f'{collection_name}_parents')

    # Verificar si la colección existe en MongoDB
    if mongo_collection.count_documents({}) > 0:
        print(f"Laden der bestehenden Kollektion in MongoDB:'{collection_name}_parents'")
        store = MongoDBStore(
            connection_string=MONGODB_CONNECTION_STRING,
            db_name=MONGODB_DATABASE_NAME,
            collection_name=f'{collection_name}_parents'
        )
    else:
        print(f"Die Kollektion '{collection_name}_parents' existiert nicht in MongoDB. Erstellen und Hinzufügen von Dokumenten...")
        store = MongoDBStore(
            connection_string=MONGODB_CONNECTION_STRING,
            db_name=MONGODB_DATABASE_NAME,
            collection_name=f'{collection_name}_parents'
        )
        # Crear el retriever y agregar documentos
        temp_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            k=top_k,
        )
        
        # Añadir barra de progreso para la carga de documentos en MongoDB
        print("Procesando y cargando documentos en MongoDB...")
        for i, doc in tqdm(enumerate(docs), total=len(docs), desc="Cargando documentos en MongoDB"):
            # Asegurarse de que los metadatos existan
            if doc.metadata is None:
                doc.metadata = {}
            
            # Agregar 'start_index' y 'doc_id' a los metadatos
            doc.metadata['start_index'] = i
            doc.metadata['doc_id'] = str(i)
            
            # Verificar que el contenido del documento no esté vacío
            if doc.page_content.strip():
                try:
                    # Añadir el documento al retriever
                    temp_retriever.add_documents([doc])
                except Exception as e:
                    print(f"Error al procesar el documento {i}: {str(e)}")
                    print(f"Contenido del documento: {doc.page_content[:100]}...")  # Imprimir los primeros 100 caracteres
            else:
                print(f"El documento {i} está vacío y será omitido.")

    # Crear el retriever final
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        k=top_k,
    )

    return retriever


def rerank_docs(query, retrieved_docs, reranker_model):
    """
    Rerank the provided context chunks

    Parameters:
    - reranker_model: the model selection.
    - query: user query - string
    - retrieved_docs: chunks that needs to be ranked. 

    
    Returns:
    - Sorted list of chunks based on their relevance to the query. 

    """

    if reranker_model=="gpt":
        ranked_docs = reranking_gpt(retrieved_docs, query)
    elif reranker_model=="german":
        ranked_docs = reranking_german(retrieved_docs, query)
    elif reranker_model=="cohere":
        ranked_docs = reranking_cohere(retrieved_docs, query)
    elif reranker_model=="colbert":
        ranked_docs = reranking_colbert(retrieved_docs, query)
    else: # just return the original order
        ranked_docs = [(query, r.page_content) for r in retrieved_docs]

    return ranked_docs


def retrieve_context_reranked(query, retriever, reranker_model):
    """
    Retrieve the context and rerank them based on the selected re-ranking model.

    Parameters:
    - query: user query - string
    - retrieved_docs: chunks that needs to be ranked. 
    - reranker_model: the model selection.

    
    Returns:
    - Sorted list of chunks based on their relevance to the query. 

    """

    retrieved_docs = retriever.invoke(input=query)

    #print(type(retrieved_docs), type(retrieved_docs[0]) if retrieved_docs else None)

    if len(retrieved_docs) == 0:
        print(
            f"Es konnte kein relevantes Dokument mit der Abfrage `{query}` gefunden werden. Versuche, deine Frage zu ändern!"
        )
    reranked_docs = rerank_docs(
        query=query, retrieved_docs=retrieved_docs, reranker_model=reranker_model
    )

    if len(reranked_docs) == 0:
        print(
            f"Die re-rankteten Dokumente sind 0."
        )
    return reranked_docs

def azure_openai_call(prompt):
    # Si el prompt es un objeto HumanMessage, extraemos su contenido
    if isinstance(prompt, HumanMessage):
        prompt_content = prompt.content
    else:
        prompt_content = str(prompt)
    
    response = load_llm_client().chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
            {"role": "user", "content": prompt_content}
        ]
    )
    return response.choices[0].message.content

def load_llm_client():
    dotenv_path = Path(ENV_VAR_PATH)
    load_dotenv(dotenv_path=dotenv_path)

    # Configurar el cliente de Azure OpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2023-05-15",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return client