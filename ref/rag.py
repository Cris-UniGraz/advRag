import os
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rich import print
import torch
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ParentDocumentRetriever
from reranking_models import reranking_cohere, reranking_colbert, reranking_gpt, reranking_german

# import fitz  # PyMuPDF
from langchain.docstore.document import Document

# import docx
# import openpyxl
from langchain_milvus import Milvus
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from pathlib import Path
from openai import AzureOpenAI
from pymilvus import connections, Collection, utility
from langchain.chains import HypotheticalDocumentEmbedder
from tqdm import tqdm

from langchain_community.storage.mongodb import MongoDBStore
from pymongo import MongoClient

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rag2.loaders import load_pdf, load_docx, load_xlsx

# Al principio del archivo, después de las importaciones
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
AZURE_OPENAI_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL")

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 16))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", 4096))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", 0))
PAGE_OVERLAP = int(os.getenv("PAGE_OVERLAP", 256))


def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if not file.startswith('~$'):
            file_path = os.path.join(folder_path, file)
            if file.lower().endswith('.pdf'):
                documents.extend(load_pdf(file_path, file, PAGE_OVERLAP))
            elif file.lower().endswith('.docx'):
                documents.extend(load_docx(file_path, file))
            elif file.lower().endswith('.xlsx'):
                documents.extend(load_xlsx(file_path, file))
    return documents


def split_documents(documents, split_size, split_overlap):
    # Dividir documentos en tamaño DOC_SIZE
    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=split_size,
        chunk_overlap=split_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    split_docs = []
    for doc in documents:
        splits = doc_splitter.split_text(doc.page_content)
        for i, split in enumerate(splits):
            metadata = doc.metadata.copy()
            metadata['doc_chunk'] = i
            split_docs.append(Document(page_content=split, metadata=metadata))
    
    return split_docs


def process_documents(folder_path):
    # Cargar documentos
    documents = load_documents(folder_path)
    
    # Dividir en documentos de tamaño DOC_SIZE
    split_docs = split_documents(documents, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)
    
    # Dividir en chunks de tamaño CHUNK_SIZE
    split_chunks = split_documents(split_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    
    return split_chunks

   
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


def create_milvus_collection(docs, embedding_model, chunk_size, chunk_overlap, collection_name):
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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
    )
    all_splits = text_splitter.split_documents(docs)
    docs_processed = []

    # Iterar sobre los documentos y mostrar el progreso
    for doc in tqdm(all_splits, desc="Verarbeitung von Dokumenten"):
        docs_processed.append(doc)

    # Supongamos que Milvus.from_documents permite la inserción en lotes
    batch_size = 1  # Tamaño del lote
    num_batches = len(docs_processed) // batch_size + (1 if len(docs_processed) % batch_size != 0 else 0)

    # Crear el vectorstore con los documentos procesados en lotes
    for i in tqdm(range(num_batches), desc="Einfügen von Dokumenten in Milvus"):
        batch = docs_processed[i * batch_size:(i + 1) * batch_size]
        Milvus.from_documents(
            documents=batch, 
            embedding=embedding_model, 
            collection_name=collection_name,
        )
    
    vectorstore = Milvus(collection_name=collection_name, embedding_function=embedding_model)
    
    '''
    # Crear el vectorstore con los documentos procesados en lotes
    Milvus.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        collection_name=collection_name,
    )
    '''

    return vectorstore


def get_ensemble_retriever(folder_path, embedding_model, llm, collection_name="test", top_k=3, language="german"):
    """
    Initializes a retriever object with chat history awareness to fetch the top_k most relevant documents.
    Now includes chat history awareness in all retrievers in the ensemble.

    Parameters:
    - folder_path: Path to the documents folder
    - embedding_model: The embedding model to use for generating document embeddings
    - llm: Language model to use for query reformulation and HyDE
    - collection_name: The name of the collection
    - top_k: The number of top relevant documents to retrieve
    - language: The language for prompts and responses

    Returns:
    - An ensemble retriever object that considers chat history when retrieving documents
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    try:
        # Establecer conexión con Milvus
        connections.connect()

        if utility.has_collection(collection_name):
            # Cargar el vectorstore base
            base_vectorstore = get_milvus_collection(embedding_model, collection_name)

            # Crear el Parent Document Retriever
            children_vectorstore = get_milvus_collection(embedding_model, collection_name)
            parent_retriever = create_parent_retriever(children_vectorstore, collection_name, top_k)
        else:
            # Cargar los documentos
            # docs = load_documents(folder_path=folder_path)
            docs = split_documents(load_documents(folder_path=folder_path), PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)
            
            # Crear el vectorstore base
            base_vectorstore = create_milvus_collection(docs, embedding_model, CHUNK_SIZE, CHUNK_OVERLAP, collection_name)

            # Crear el Parent Document Retriever
            children_vectorstore = get_milvus_collection(embedding_model, collection_name)
            parent_retriever = create_parent_retriever(children_vectorstore, collection_name, top_k, docs=docs)

        # Crear o cargar el vectorstore base with history awareness
        base_retriever = base_vectorstore.as_retriever(search_kwargs={"k": top_k})
        contextualize_q_system_prompt = (
            f"Given a chat history and the latest user question in {language} "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_base_retriever = create_history_aware_retriever(
            llm, base_retriever, contextualize_q_prompt
        )

        # Crear el retriever de palabras clave with history awareness
        # keyword_retriever = BM25Retriever.from_documents(docs)
        keyword_retriever = load_bm25(f'{collection_name}_parents')
        if keyword_retriever is None:
            keyword_retriever = history_aware_base_retriever
        else:
            keyword_retriever.k = top_k
            history_aware_keyword_retriever = create_history_aware_retriever(
                llm, keyword_retriever, contextualize_q_prompt
            )

        # Crear el retriever de consultas múltiples with history awareness
        try:
            multi_query_retriever = get_multi_query_retriever(base_retriever, llm, language)
        except Exception as e:
            print(f"Warning: Could not create MultiQueryRetriever: {e}")
            multi_query_retriever = base_retriever  # Usar el retriever base como fallback

        # Crear el HyDE retriever with history awareness
        hyde_retriever = get_hyde_retriever(embedding_model, llm, collection_name, language, top_k)
        history_aware_hyde_retriever = create_history_aware_retriever(
            llm, hyde_retriever, contextualize_q_prompt
        )

        # Create parent retriever with history awareness
        history_aware_parent_retriever = create_history_aware_retriever(
            llm, parent_retriever, contextualize_q_prompt
        )

        # Crear el ensemble retriever con los cinco retrievers with all history-aware retrievers
        working_retrievers = []
        weights = []
        
        # Lista de tuplas con (nombre, retriever, peso)
        retrievers_config = [
            ("history_aware_base_retriever", history_aware_base_retriever, 0.1),
            ("history_aware_keyword_retriever", history_aware_keyword_retriever if keyword_retriever is not None else history_aware_base_retriever, 0.1),
            ("multi_query_retriever", multi_query_retriever, 0.4),
            ("history_aware_hyde_retriever", history_aware_hyde_retriever, 0.1),
            ("history_aware_parent_retriever", history_aware_parent_retriever, 0.3)
        ]
        
        for name, retriever, weight in retrievers_config:
            if retriever is not None:
                working_retrievers.append(retriever)
                weights.append(weight)
                # print(f">> Retriever incluido en el ensemble_retriever: {name} .")
        
        # Normalizar los pesos
        if weights:
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
        
        # print(f">> Pesos del ensemble_retriever: {weights} .")
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=working_retrievers,
            weights=weights
        )
        
        '''
        # Pesos finales
        weights=[0.1, 0.1, 0.4, 0.1, 0.3]
        )
        '''

        return ensemble_retriever
        

    except Exception as e:
        print(f"An error occurred while initializing the retriever: {e}")
        raise

def retrieve_context(query, retriever, chat_history=[]):
    """
    Retrieves documents relevant to a given query, considering chat history.

    Parameters:
    - query: The search query as a string
    - retriever: An instance of a Retriever class used to fetch documents
    - chat_history: List of previous chat messages as (human_message, ai_message) tuples

    Returns:
    - A list of documents deemed relevant to the query
    """
    # Convert chat history to the format expected by the retriever
    formatted_history = []
    for human_msg, ai_msg in chat_history:
        formatted_history.extend([
            HumanMessage(content=human_msg),
            AIMessage(content=ai_msg)
        ])

    # Retrieve documents using the chat history
    retrieved_docs = retriever.invoke({
        "input": query,
        "chat_history": formatted_history
    })

    return retrieved_docs


# import pickle
from pymongo import MongoClient
from bson.binary import Binary
from langchain_community.retrievers import BM25Retriever

def get_mongo_collection(collection_name):
    client = MongoClient(MONGODB_CONNECTION_STRING)
    db = client[MONGODB_DATABASE_NAME]
    return db[collection_name]

# Función para cargar el BM25Retriever desde MongoDB
def load_bm25(collection_name):
    print(f"Laden der bestehenden Kollektion in MongoDB:'{collection_name}' (für Keywords)")
    docstore = MongoDBStore(
        connection_string=MONGODB_CONNECTION_STRING,
        db_name=MONGODB_DATABASE_NAME,
        collection_name=collection_name,
    )
    keys = list(docstore.yield_keys())
    
    if not keys:
        print(f"No se encontraron documentos en la colección '{collection_name}'")
        return None
    
    docs_processed = docstore.mget(keys)
    
    if not docs_processed:
        print(f"No se pudieron recuperar documentos de la colección '{collection_name}'")
        return None
    
    # Filtrar documentos válidos
    valid_docs = [doc for doc in docs_processed if doc and hasattr(doc, 'page_content') and hasattr(doc, 'metadata')]
    
    if not valid_docs:
        print("No se encontraron documentos válidos para crear el BM25Retriever")
        return None
    
    try:
        retriever = BM25Retriever.from_documents(valid_docs)
        # print(f"BM25Retriever creado con éxito con {len(valid_docs)} documentos")
        return retriever
    except Exception as e:
        print(f"Error al crear BM25Retriever: {str(e)}")
        return None
    

from typing import List

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


from langchain_milvus import Milvus
from pymilvus import connections, Collection, utility

def create_parent_retriever(
    vectorstore,
    collection_name, 
    top_k=5,
    docs=None  # Hacer que docs sea opcional
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
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        model_name="gpt-4",
        is_separator_regex=False,
    )

    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n\n", "\n\n", "\n", ".", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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
        if docs is None:
            raise ValueError("Documents are required when creating a new collection")
        
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
        
        # Solo procesar documentos si estamos creando una nueva colección
        if docs:
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


def get_multi_query_retriever(base_retriever, llm, language):
    """
    Create a multi-query retriever based on the base retriever and LLM.
    
    Parameters:
    - base_retriever: base retriever
    - llm: LLM to generate variations of queries.
    - language: Language for generating queries
    
    Returns:
    - A retriever that is able to generate multiple queries.
    """
    # Si el base_retriever es un RunnableBinding, creamos un nuevo retriever básico
    if hasattr(base_retriever, 'retriever'):
        vectorstore = base_retriever.retriever.vectorstore
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": base_retriever.retriever.search_kwargs.get("k", 4)})

    output_parser = LineListOutputParser()
    
    prompt = PromptTemplate(
        input_variables=["question"],
        template=f"""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question in {language} to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {{question}}""",
    )

    # Modificar la cadena para asegurar que la salida sea un string
    def format_output(output):
        if hasattr(output, 'text'):
            return output.text
        return str(output)
    
    # Crear la cadena LLM con el formato correcto
    llm_chain = (
        {"question": lambda x: x}
        | prompt
        | llm
        | format_output
        | output_parser
    )
    
    # Crear el MultiQueryRetriever
    try:
        retriever = MultiQueryRetriever(
            retriever=base_retriever,
            llm_chain=llm_chain,
            parser_key="lines"  # Este es el nombre de la clave que contendrá las queries generadas
        )
        return retriever
    except Exception as e:
        print(f"Error creating MultiQueryRetriever: {e}")
        # En caso de error, devolver el retriever original
        return base_retriever


def get_hyde_retriever(embedding_model, llm, collection_name, language, top_k=3):
    """
    Initializes a HyDE retriever that generates multiple query variations based on the provided LLM.

    Parameters:
    - embedding_model: The embedding model used to generate document embeddings.
    - llm: LLM to generate variations of queries.
    - collection_name: The name of the collection in the vector store.
    - top_k: The number of top relevant documents to retrieve. Defaults to 3.

    Returns:
    - An instance of a HyDE (Hypothetical Document Embedder)
    """
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=f"""Please write a passage in {language} to answer the question.
    Question: {{question}}""",
    )

    llm_chain = QUERY_PROMPT | llm | StrOutputParser()

    hyde_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain,
        base_embeddings=embedding_model,
    )
    hyde_vectorstore = get_milvus_collection(hyde_embeddings, collection_name)
    '''
    vectorstore = Milvus(
        hyde_embeddings,
        connection_args={"uri": VECTOR_DATABASE_URI},
        collection_name=collection_name,
    )
    '''
    hyde_retriever = hyde_vectorstore.as_retriever(search_kwargs={"k": top_k})

    return hyde_retriever


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


def retrieve_context_reranked(query, retriever, reranker_model, chat_history=[]):
    """
    Retrieve the context and rerank them based on the selected re-ranking model,
    considering chat history.

    Parameters:
    - query: user query - string
    - retriever: retriever instance
    - reranker_model: the model selection
    - chat_history: List of previous chat messages as (human_message, ai_message) tuples

    Returns:
    - Sorted list of chunks based on their relevance to the query
    """
    retrieved_docs = retrieve_context(query, retriever, chat_history)

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
        model=AZURE_OPENAI_LLM_MODEL,
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
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AAZURE_OPENAI_API_LLM_DEPLOYMENT_ID")
    )
    return client