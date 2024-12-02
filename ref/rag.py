import asyncio
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain.chains import HypotheticalDocumentEmbedder, create_history_aware_retriever
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from reranking_models import reranking_cohere, reranking_colbert, reranking_gpt, reranking_german
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.storage.mongodb import MongoDBStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_milvus import Milvus
from langchain_openai import AzureOpenAIEmbeddings
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AzureOpenAI
from pathlib import Path
from pymilvus import connections, utility
from pymongo import MongoClient
from rich import print
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from rag2.loaders import load_documents

from glossary import find_glossary_terms, find_glossary_terms_with_explanation, get_glossary  # Importa el método get_glossary


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

# LangSmith configuration
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

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

   
def load_embedding_model(
    model_name = EMBEDDING_MODEL_NAME, 
):
    """
    Loads an embedding model from either Azure OpenAI or Hugging Face repository.

    Parameters:
    - model_name: The name of the model to load. Use "openai" for Azure OpenAI embeddings
                 or a Hugging Face model name (defaults to EMBEDDING_MODEL_NAME).

    Returns:
    - An instance of AzureOpenAIEmbeddings or HuggingFaceBgeEmbeddings

    Raises:
    - ValueError: If an unsupported device is specified.
    - OSError: If the model cannot be loaded.
    """

    try:
        if model_name == "azure_openai":
            embedding_model = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
            )
            # aoaimodel=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
            # print(f"Se ha cargado el embedding model de Azure OpenAI: {aoaimodel}")
        else:
            model_name = EMBEDDING_MODEL_NAME 
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


async def get_ensemble_retriever(folder_path, embedding_model, llm, collection_name="test", top_k=3, language="german", max_concurrency=5):
    """
    Initializes an async ensemble retriever with parallel search capabilities.
    
    Parameters:
    - folder_path: Path to the documents folder
    - embedding_model: The embedding model to use
    - llm: Language model for query reformulation
    - collection_name: The name of the collection
    - top_k: Number of top relevant documents to retrieve
    - language: The language for prompts and responses
    - max_concurrency: Maximum number of concurrent operations
    
    Returns:
    - An async ensemble retriever configured for parallel search
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    try:
        # Establecer conexión con Milvus
        connections.connect()

        if utility.has_collection(collection_name):
            base_vectorstore = get_milvus_collection(embedding_model, collection_name)
            children_vectorstore = get_milvus_collection(embedding_model, collection_name)
            parent_retriever = create_parent_retriever(children_vectorstore, collection_name, top_k)
        else:
            docs = split_documents(load_documents(folder_path=folder_path), PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)
            base_vectorstore = create_milvus_collection(docs, embedding_model, CHUNK_SIZE, CHUNK_OVERLAP, collection_name)
            children_vectorstore = get_milvus_collection(embedding_model, collection_name)
            parent_retriever = create_parent_retriever(children_vectorstore, collection_name, top_k, docs=docs)

        # Configurar retrievers individuales
        base_retriever = base_vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        keyword_retriever = load_bm25(f'{collection_name}_parents')
        if keyword_retriever is None:
            keyword_retriever = base_retriever
        else:
            keyword_retriever.k = top_k

        multi_query_retriever = get_multi_query_retriever(parent_retriever, llm, language)
        hyde_retriever = get_hyde_retriever(embedding_model, llm, collection_name, language, top_k)

        # Crear el ensemble retriever con configuración para búsqueda paralela
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                base_retriever,
                keyword_retriever,
                multi_query_retriever,
                hyde_retriever,
                parent_retriever
            ],
            # weights=[0.2, 0.2, 0.2, 0.2, 0.2]  # Pesos iguales para todos los retrievers
            # weights=[0.0, 0.0, 1.0, 0.0, 0.0]
            # weights=[0.1, 0.1, 0.4, 0.1, 0.3]

            # Pesos finales
            weights=[0.1, 0.1, 0.4, 0.1, 0.3],
            c=60,
            batch_config={
                "max_concurrency": max_concurrency
            }
        )

        # Configurar el prompt para contextualizar preguntas con el historial
        contextualize_q_system_prompt = (
            f"Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. Give the question in {language}."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Crear el retriever con awareness del historial
        history_aware_retriever = create_history_aware_retriever(
            llm,
            ensemble_retriever,
            contextualize_q_prompt
        )

        return history_aware_retriever

    except Exception as e:
        print(f"An error occurred while initializing the retriever: {e}")
        raise


async def retrieve_context_async(query: str, retriever, chat_history=[], language="german"):
    """
    Versión asíncrona de retrieve_context que permite ejecución paralela.
    """
    # Convert chat history to the format expected by the retriever
    formatted_history = []
    for human_msg, ai_msg in chat_history:
        formatted_history.extend([
            HumanMessage(content=human_msg),
            AIMessage(content=ai_msg)
        ])

    # Usar ThreadPoolExecutor para ejecutar la operación de recuperación en un thread separado
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        retrieved_docs = await loop.run_in_executor(
            executor,
            lambda: retriever.invoke({
                "input": query,
                "chat_history": formatted_history,
                "language": language
            })
        )

    return retrieved_docs


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
 

# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


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
    Create a glossary-aware multi-query retriever
    """
    output_parser = LineListOutputParser()
    
    def create_multi_query_chain(query):
        matching_terms = find_glossary_terms_with_explanation(query, language)
        
        if not matching_terms:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an AI language model assistant. Your task is to generate five different versions of the given user question in {language} to retrieve relevant documents. By generating multiple perspectives on the user question, your goal is to help overcome some limitations of distance-based similarity search. Provide these alternative questions separated by newlines."""),
                ("human", "{question}")
            ])
        else:
            relevant_glossary = "\n".join([f"{term}: {explanation}" 
                                         for term, explanation in matching_terms])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an AI language model assistant. Your task is to generate five different versions of the given user question in {language} to retrieve relevant documents. The following terms from the question have specific meanings: "
                "{relevant_glossary}."
                "Generate questions that incorporate these specific meanings. Provide these alternative questions separated by newlines."""),
                ("human", "{question}")
            ])

        chain = prompt | llm | output_parser
        return chain

    class GlossaryAwareMultiQueryRetriever(MultiQueryRetriever):
        async def _aget_relevant_documents(self, query: str, *, run_manager=None):
            self.llm_chain = create_multi_query_chain(query)
            return await super()._aget_relevant_documents(query, run_manager=run_manager)

    retriever = GlossaryAwareMultiQueryRetriever(
        retriever=base_retriever,
        llm_chain=create_multi_query_chain("")
    )
    
    return retriever



def get_hyde_retriever(embedding_model, llm, collection_name, language, top_k=3):
    """
    Initializes a HyDE retriever with glossary-aware query generation
    """
    def create_hyde_chain(query):
        matching_terms = find_glossary_terms_with_explanation(query, language)
        
        if not matching_terms:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"Please write a passage in {language} to answer the question."),
                ("human", "{question}")
            ])
        else:
            relevant_glossary = "\n".join([f"{term}: {explanation}" 
                                         for term, explanation in matching_terms])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""Please write a passage in {language} to answer the question. The following terms from the question have specific meanings: "
                "{relevant_glossary}"""),
                ("human", "{question}")
            ])

        chain = prompt | llm | StrOutputParser()
        return chain

    class GlossaryAwareHyDEEmbedder(HypotheticalDocumentEmbedder):
        def embed_query(self, query: str, *args, **kwargs):
            self.llm_chain = create_hyde_chain(query)
            return super().embed_query(query, *args, **kwargs)

    hyde_embeddings = GlossaryAwareHyDEEmbedder(
        llm_chain=create_hyde_chain(""),  # Placeholder chain
        base_embeddings=embedding_model
    )
    
    hyde_vectorstore = get_milvus_collection(hyde_embeddings, collection_name)
    return hyde_vectorstore.as_retriever(search_kwargs={"k": top_k})


def rerank_docs(query, retrieved_docs, reranker_type, language: str):
    """
    Rerank the provided context chunks

    Parameters:
    - reranker_type: the model selection.
    - query: user query - string
    - retrieved_docs: chunks that needs to be ranked. 

    
    Returns:
    - Sorted list of chunks based on their relevance to the query. 

    """

    if reranker_type=="gpt":
        ranked_docs = reranking_gpt(retrieved_docs, query)
    elif reranker_type=="german":
        ranked_docs = reranking_german(retrieved_docs, query)
    elif reranker_type=="cohere":
        if (language.lower() == "english"):
            model = os.getenv("ENGLISH_COHERE_RERANKING_MODEL")
        else:
            model = os.getenv("GERMAN_COHERE_RERANKING_MODEL")
        ranked_docs = reranking_cohere(retrieved_docs, query, model)
    elif reranker_type=="colbert":
        ranked_docs = reranking_colbert(retrieved_docs, query)
    else: # just return the original order
        ranked_docs = [(query, r.page_content) for r in retrieved_docs]

    return ranked_docs


async def rerank_docs_async(query: str, retrieved_docs: List, reranker_type: str, language: str):
    """
    Versión asíncrona de rerank_docs que permite ejecución paralela.
    """
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        ranked_docs = await loop.run_in_executor(
            executor,
            lambda: rerank_docs(query, retrieved_docs, reranker_type, language)
        )
    return ranked_docs

async def getStepBackQuery(
    query: str,
    llm: Any,
    language: str = "german"
) -> str:
    """
    Generate a more generic step-back query from the original query, considering glossary terms.
    
    Args:
        query: Original user query
        llm: Azure OpenAI LLM instance
        language: Target language for the response
    Returns:
        str: Generated step-back query
    """
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel's was born in what country?",
            "output": "what is Jan Sindel's personal history?",
        },
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # Buscar términos del glosario en la query
    matching_terms = find_glossary_terms_with_explanation(query, language)

    if not matching_terms:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Please note that the question has been asked in the context of the University of Graz. Give the generic step-back question in {language}. Here are a few examples:""",
            ),
            few_shot_prompt,
            ("user", "{question}"),
        ])
    else:
        relevant_glossary = "\n".join([f"{term}: {explanation}" 
                                     for term, explanation in matching_terms])
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. The following terms from the question have specific meanings:
                {relevant_glossary}
                Please consider these specific meanings when generating the step-back question. Please note that the question has been asked in the context of the University of Graz. Give the generic step-back question in {language}. Here are a few examples:""",
            ),
            few_shot_prompt,
            ("user", "{question}"),
        ])

    chain = prompt | llm | StrOutputParser()
    step_back_query = await chain.ainvoke({"language": language, "question": query})
    
    return step_back_query



async def translate_query(query: str, language: str, target_language: str, llm: Any) -> str:
    from langchain_core.output_parsers import StrOutputParser
    
    matching_terms = find_glossary_terms(query, language)
        
    if not matching_terms:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Translate the following text to {target_language}. Only provide the translation, no explanations."),
            ("human", "{query}")
        ])
    else:        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Translate the following text to {target_language}. Only provide the translation, no explanations. If these terms {matching_terms} appear in the text to be translated, do not translate them but use them as they are written."),
            ("human", "{query}")
        ])
      
    chain = prompt | llm | StrOutputParser()
    translated_query = await chain.ainvoke({"query": query})
    return translated_query


@traceable # Auto-trace this function
async def process_queries_and_combine_results(
    query: str,
    llm: Any,
    retriever_de: Any,
    retriever_en: Any,
    reranker_type_de: str,
    reranker_type_en: str,
    chat_history: List[Tuple[str, str]] = [],
    language: str = "german"
) -> List[Document]:
    """
    Process both original and step-back queries in parallel and combine their results.
    
    Args:
        query: Original user query
        llm: Azure OpenAI LLM instance
        retriever_de: First retriever instance, in german
        retriever_en: Second retriever instance, in english
        reranker_type_de: Type of first reranker, in german
        reranker_type_en: Type of second reranker, in english
        chat_history: List of chat history tuples
        language: Query language
    
    Returns:
        List[Document]: Combined and sorted list of retrieved documents
    """
    # Determinar queries según el idioma
    if (language.lower() == "german"):
        query_de = query
        query_en = await translate_query(query, language, "english", llm)
    else:
        query_de = await translate_query(query, language, "german", llm)
        query_en = query
    
    # Get step-back query
    step_back_query_de = await getStepBackQuery(query_de, llm, "german")
    step_back_query_en = await getStepBackQuery(query_en, llm, "english")
    
    # print(f"query_de: {query_de}")
    # print(f"step_back_query_de: {step_back_query_de}")
    # print(f"query_en: {query_en}")
    # print(f"step_back_query: {step_back_query_en}")

    # Process both queries in parallel
    retrieval_tasks = [
        retrieve_context_reranked(
            query_de,
            retriever_de,
            reranker_type_de,
            chat_history,
            "german"
        ),
        retrieve_context_reranked(
            step_back_query_de,
            retriever_de,
            reranker_type_de,
            chat_history,
            "german"
        ),
        retrieve_context_reranked(
            query_en,
            retriever_en,
            reranker_type_en,
            chat_history,
            "english"
        ),
        retrieve_context_reranked(
            step_back_query_en,
            retriever_en,
            reranker_type_en,
            chat_history,
            "english"
        )
    ]
    
    # Wait for all retrievals to complete
    results = await asyncio.gather(*retrieval_tasks)
    reranked_docs_original_de, reranked_docs_step_back_de, reranked_docs_original_en, reranked_docs_step_back_en = results
    
    # Combine and sort results
    all_reranked_docs = reranked_docs_original_de + reranked_docs_step_back_de + reranked_docs_original_en + reranked_docs_step_back_en
        
    if not all_reranked_docs:
        print(
            f"Es konnte kein relevantes Dokument mit der Abfrage `{query}` gefunden werden. "
            "Versuche, deine Frage zu ändern!"
        )
        return []

    # Remove duplicates based on content and source
    seen = set()
    unique_docs = []
    
    for doc in all_reranked_docs:
        doc_id = (doc.page_content, doc.metadata.get('source'))
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)
    
    # Sort by reranking score
    sorted_docs = sorted(
        unique_docs,
        key=lambda x: x.metadata.get('reranking_score', 0),
        reverse=True
    )
    
    return sorted_docs


@traceable # Auto-trace this function
async def retrieve_context_reranked(
    query: str,
    retriever,
    reranker: str,
    chat_history=[],
    language: str = "german"
):
    """
    Versión asíncrona mejorada que ejecuta las recuperaciones y reranking en paralelo,
    aprovechando ainvoke para búsquedas verdaderamente asíncronas.
    """
    try:
        # Convertir el historial de chat al formato esperado
        formatted_history = []
        for human_msg, ai_msg in chat_history:
            formatted_history.extend([
                HumanMessage(content=human_msg),
                AIMessage(content=ai_msg)
            ])
        
        # 1. Preparar los argumentos para ambos retrievers
        retrieval_args = {
            "input": query,
            "chat_history": formatted_history,
            "language": language
        }
        
        # 2. Ejecutar ambas recuperaciones en paralelo usando ainvoke
        retrieved_docs = await retriever.ainvoke(retrieval_args)
        
        # 3. Ejecutar reranking en paralelo
        reranked_docs = await rerank_docs_async(
            query,
            retrieved_docs,
            reranker,
            language
        )
           
        if not reranked_docs:
            print("Die re-rankteten Dokumente sind 0.")
            
        return reranked_docs
    
    except Exception as e:
        print(f"Error en retrieve_context_reranked: {str(e)}")
        return []


@traceable # Auto-trace this function
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
        azure_deployment=os.getenv("AZURE_OPENAI_API_LLM_DEPLOYMENT_ID")
    )
    return wrap_openai(client)