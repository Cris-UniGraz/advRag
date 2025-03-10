import asyncio
import os
from dotenv import load_dotenv
from langchain.chains import HypotheticalDocumentEmbedder, create_history_aware_retriever
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from reranking_models import reranking_cohere, reranking_colbert, reranking_gpt, reranking_german
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.storage.mongodb import MongoDBStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_milvus import Milvus
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AzureOpenAI
from pathlib import Path
from pymilvus import connections, utility
from pymongo import MongoClient
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from rag_utils.loaders import load_documents

from glossary import find_glossary_terms, find_glossary_terms_with_explanation, get_glossary  # Importa el método get_glossary
from coroutine_manager import coroutine_manager
from query_optimizer import QueryOptimizer
from EmbeddingManager import EmbeddingManager


# Al principio del archivo, después de las importaciones
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

AZURE_OPENAI_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL")

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 16))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", 4096))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", 0))

GERMAN_EMBEDDING_MODEL_NAME = os.getenv("GERMAN_EMBEDDING_MODEL_NAME")
ENGLISH_EMBEDDING_MODEL_NAME = os.getenv("ENGLISH_EMBEDDING_MODEL_NAME")

MAX_CHUNKS_LLM = int(os.getenv("MAX_CHUNKS_LLM", 3))  # Convertir a entero con valor por defecto

SHOW_INTERNAL_MESSAGES = os.getenv("SHOW_INTERNAL_MESSAGES", "false").lower() == "true"

# LangSmith configuration
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

'''
GERMAN_EMBEDDING_MODEL = None
ENGLISH_EMBEDDING_MODEL = None

def initialize_embedding_models():
    global GERMAN_EMBEDDING_MODEL, ENGLISH_EMBEDDING_MODEL
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_german = executor.submit(load_embedding_model, model_name=GERMAN_EMBEDDING_MODEL_NAME)
        future_english = executor.submit(load_embedding_model, model_name=ENGLISH_EMBEDDING_MODEL_NAME)
        
        GERMAN_EMBEDDING_MODEL = future_german.result()
        ENGLISH_EMBEDDING_MODEL = future_english.result()
'''
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
    
    return vectorstore


async def get_ensemble_retriever(folder_path, embedding_manager: EmbeddingManager, llm, collection_name="test", top_k=3, language="german", max_concurrency=5):
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

        # Seleccionar el modelo apropiado según el idioma
        if isinstance(embedding_manager, EmbeddingManager):
            embedding_model = (
                embedding_manager.german_model if language == "german" 
                else embedding_manager.english_model
            )
        else:
            embedding_model = embedding_manager

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
        
        #if language == "english":
        #    print(f">> get_multi_query_retriever > query = {query} - language = {language} - matching_terms = {matching_terms}.")

        if not matching_terms:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an AI language model assistant. Your task is to generate five different versions of the given user question in {language} to retrieve relevant documents. By generating multiple perspectives on the user question, your goal is to help overcome some limitations of distance-based similarity search. Provide these alternative questions separated by newlines."""),
                ("human", "{question}")
            ])
        else:
            relevant_glossary = "\n".join([f"{term}: {explanation}" 
                                         for term, explanation in matching_terms])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an AI language model assistant. Your task is to generate five different versions of the given user question in {language} to retrieve relevant documents. The following terms from the question have specific meanings: {relevant_glossary}. Generate questions that incorporate these specific meanings. Provide these alternative questions separated by newlines."""),
                ("human", "{question}")
            ])

        chain = prompt | llm | output_parser
        return chain

    class GlossaryAwareMultiQueryRetriever(MultiQueryRetriever):
        async def _aget_relevant_documents(self, query: str, *, run_manager=None):
            self.llm_chain = create_multi_query_chain(query)
            
            if SHOW_INTERNAL_MESSAGES:
                # Imprimir el prompt y la respuesta
                print("\n=== Multi Query Retriever ===")
                print(f"Query original: {query}")
                generated_queries = await self.llm_chain.ainvoke({"question": query})
                print("=== Multi Query Retriever - Queries generadas: ===")
                for q in generated_queries:
                    print(f"- {q}")
                print("===========================\n")
            
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
                ("system", f"""Please write a passage in {language} to answer the question. The following terms from the question have specific meanings: {relevant_glossary}"""),
                ("human", "{question}")
            ])
            
        chain = prompt | llm | StrOutputParser()
        return chain

    class GlossaryAwareHyDEEmbedder(HypotheticalDocumentEmbedder):
        def embed_query(self, query: str, *args, **kwargs):
            self.llm_chain = create_hyde_chain(query)
            
            if SHOW_INTERNAL_MESSAGES:
                # Imprimir el prompt y la respuesta
                print("\n=== HyDE Retriever ===")
                print(f"Query original: {query}")
                hypothetical_doc = self.llm_chain.invoke({"question": query})
                print("=== HyDE Retriever - Documento hipotético generado: ===")
                print(hypothetical_doc)
                print("=====================\n")
            
            return super().embed_query(query, *args, **kwargs)


    hyde_embeddings = GlossaryAwareHyDEEmbedder(
        llm_chain=create_hyde_chain(""),  # Placeholder chain
        base_embeddings=embedding_model
    )
    
    hyde_vectorstore = get_milvus_collection(hyde_embeddings, collection_name)
    return hyde_vectorstore.as_retriever(search_kwargs={"k": top_k})


@coroutine_manager.coroutine_handler(timeout=30)
async def rerank_docs(query, retrieved_docs, reranker_type, language: str):
    try:
        if reranker_type=="gpt":
            ranked_docs = await reranking_gpt(retrieved_docs, query)
        elif reranker_type=="german":
            ranked_docs = await reranking_german(retrieved_docs, query)
        elif reranker_type=="cohere":
            model = os.getenv("ENGLISH_COHERE_RERANKING_MODEL") if language.lower() == "english" else os.getenv("GERMAN_COHERE_RERANKING_MODEL")
            ranked_docs = await reranking_cohere(retrieved_docs, query, model)
        elif reranker_type=="colbert":
            ranked_docs = await reranking_colbert(retrieved_docs, query)
        else:
            ranked_docs = [doc for doc in retrieved_docs]
        return ranked_docs
    except asyncio.TimeoutError:
        print("Reranking operation timed out")
        return retrieved_docs
    except Exception as e:
        print(f"Error during reranking: {e}")
        return retrieved_docs




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
    
    if SHOW_INTERNAL_MESSAGES:
        # Imprimir el prompt y la respuesta
        print("\n=== Step-Back Query ===")
        print(f"Original Query: {query}")
        print(f"Step-Back Query in {language}: {step_back_query}")
        print("===========================\n")

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

    if SHOW_INTERNAL_MESSAGES:
        # Imprimir el prompt y la respuesta
        print("\n=== Translated Query ===")
        print(f"Original Query: {query}")
        print(f"Translated in {target_language}: {translated_query}")
        print("===========================\n")
    
    return translated_query


@traceable
@coroutine_manager.coroutine_handler(timeout=120)
async def process_queries_and_combine_results(
    query: str,
    llm: Any,
    retriever_de: Any,
    retriever_en: Any,
    reranker_type_de: str,
    reranker_type_en: str,
    chat_history: List[Tuple[str, str]] = [],
    language: str = "german",
    embedding_manager: EmbeddingManager = None  # Añadir el embedding_manager como parámetro
) -> List[Document]:
    query_optimizer = QueryOptimizer()
     
    try:
        # Verificar caché primero
        cached_result = query_optimizer.get_llm_response(query, language)
        if cached_result:
            return {
                'response': cached_result['response'],  # Acceder directamente al valor
                'sources': cached_result['sources'],
                'from_cache': True,
                'documents': [Document(page_content=cached_result['response'])]
            }

        embedding_model = (
            embedding_manager.german_model if language == "german" 
            else embedding_manager.english_model
        )

        optimized_query = await query_optimizer.optimize_query(
            query,
            language,
            embedding_model
        )
        
        # Cuando se procesa el resultado del optimizador
        if optimized_query['source'] == 'cache':
            return {
                'response': optimized_query['result'],
                'sources': [],
                'from_cache': True,
                'documents': [Document(page_content=optimized_query['result'])] if isinstance(optimized_query['result'], str) else optimized_query['result']
            }
            
        # Procesar consultas en el idioma correspondiente
        if language.lower() == "german":
            query_de = query
            # Ejecutar todas las operaciones de traducción y step-back en paralelo
            all_query_tasks = [
                translate_query(query, language, "english", llm),
                getStepBackQuery(query_de, llm, language),
                getStepBackQuery(query, llm, "english")
            ]
            all_results = await coroutine_manager.gather_coroutines(*all_query_tasks)
            query_en = all_results[0]
            step_back_query_de = all_results[1]
            step_back_query_en = all_results[2]

        else:
            query_en = query
            # Ejecutar todas las operaciones de traducción y step-back en paralelo
            all_query_tasks = [
                translate_query(query, language, "german", llm),
                getStepBackQuery(query_en, llm, language),
                getStepBackQuery(query, llm, "german")
            ]
            all_results = await coroutine_manager.gather_coroutines(*all_query_tasks)
            query_de = all_results[0]
            step_back_query_en = all_results[1]
            step_back_query_de = all_results[2]
                
        # Continuar con el proceso de recuperación y reranking
        retrieval_tasks = [
            retrieve_context_reranked(
                query_de, 
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
                step_back_query_de, 
                retriever_de, 
                reranker_type_de, 
                chat_history, 
                "german"
            ),
            retrieve_context_reranked(
                step_back_query_en,
                retriever_en, 
                reranker_type_en, 
                chat_history, 
                "english"
            )
        ]
        
        results = await coroutine_manager.gather_coroutines(*retrieval_tasks)
        
        all_reranked_docs = []
        for result in results:
            if result:
                for document in result:
                    if not isinstance(document, Document):
                        continue
                    if not hasattr(document, 'metadata') or document.metadata is None:
                        document.metadata = {}
                    all_reranked_docs.append(document)
        
        if not all_reranked_docs:
            return {'response': '', 'sources': [], 'from_cache': False}

        # Preparar el contexto para el LLM
        text = ""
        sources = []
        filtered_context = []
        for document in all_reranked_docs:
            validated_doc = query_optimizer._validate_document(document)
            if len(filtered_context) <= MAX_CHUNKS_LLM:
                text += "\n" + document.page_content
                source = f"{os.path.basename(document.metadata.get('source', 'Unknown'))} (Seite {document.metadata.get('page', 'N/A')})"
                if source not in sources:
                    sources.append(source)
                filtered_context.append(validated_doc)

        # Crear el prompt template
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are an experienced virtual assistant at the University of Graz and know all the information about the University of Graz.
            Your main task is to extract information from the provided CONTEXT based on the user's QUERY.
            Think step by step and only use the information from the CONTEXT that is relevant to the user's QUERY.
            If the CONTEXT does not contain information to answer the QUESTION, try to answer the question with your knowledge, but only if the answer is appropriate.
            Give detailed answers in {language}.

            QUERY: ```{question}```\n
            CONTEXT: ```{context}```\n
            """
        )

        # Crear la cadena de procesamiento
        chain = prompt_template | llm | StrOutputParser()

        # Generar la respuesta del LLM
        response = await chain.ainvoke({
            "context": filtered_context,
            "language": language,
            "question": query
        })

        # Preparar las fuentes para caché
        sources_for_cache = []
        for doc in filtered_context:
            source_info = {
                'source': doc.metadata.get('source', 'Unknown Source'),
                'page': doc.metadata.get('page', 'N/A'),
                'sheet_name': doc.metadata.get('sheet_name', None),
                'page_number': doc.metadata.get('page_number', None)
            }
            if source_info not in sources_for_cache:  # Evitar duplicados
                sources_for_cache.append(source_info)

        # Almacenar en caché
        query_optimizer._store_llm_response(query, response, language, sources_for_cache)

        return {
            'response': response,
            'sources': sources_for_cache,
            'from_cache': False,
            'documents': all_reranked_docs
        }

    except Exception as e:
        print(f"Error processing queries: {e}")
        return {'response': '', 'sources': [], 'from_cache': False}
    finally:
        await coroutine_manager.cleanup()


@traceable # Auto-trace this function
async def retrieve_context_reranked(query, retriever, reranker, chat_history=[], language="german"):
    try:
        # Convertir el historial de chat al formato esperado
        formatted_history = []
        for human_msg, ai_msg in chat_history:
            formatted_history.extend([
                HumanMessage(content=human_msg),
                AIMessage(content=ai_msg)
            ])
        
        # Ejecutar la recuperación
        retrieved_docs = await retriever.ainvoke({
            "input": query,
            "chat_history": formatted_history,
            "language": language
        })
        
        # Ejecutar el reranking y esperar su resultado
        reranked_docs = await rerank_docs(query, retrieved_docs, reranker, language)
        
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