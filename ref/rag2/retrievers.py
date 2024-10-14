from typing import List
from langchain_milvus import Milvus
from langchain_community.storage import MongoDBStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser

from .splitters import get_text_splitter
'''
from config.constants import (
    VECTOR_DATABASE_URI,
    MONGODB_CONNECTION_STR,
    MONGODB_DATABASE_NAME,
    CHUNK_SIZE,
    PARENT_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME_SPANISH,
    EMBEDDING_MODEL_NAME_ENGLISH,
)
'''


def get_basic_retriever(embedding_model, collection_name, top_k=3):
    """
    Initializes a basic retriever object to fetch the top_k most relevant documents based on cosine similarity.

    Parameters:
    - embedding_model: The embedding model used for generating document embeddings.
    - collection_name: The name of the collection in the vector store.
    - top_k: The number of top relevant documents to retrieve. Defaults to 3.

    Returns:
    - An instance of Milvus's Basic Retriever.
    """
    vectorstore = Milvus(
        embedding_model,
        connection_args={"uri": VECTOR_DATABASE_URI},
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    return retriever


def get_keyword_search_retriever(collection_name, top_k=3):
    """
    Initializes a keyword search retriever object to fetch the top_k most relevant documents using BM25 algorithm.

    Parameters:
    - collection_name: The name of the collection in the document store.
    - top_k: The number of top relevant documents to retrieve. Defaults to 3.

    Returns:
    - An instance of the Keyword Search Retriever.
    """
    docstore = MongoDBStore(
        connection_string=MONGODB_CONNECTION_STR,
        db_name=MONGODB_DATABASE_NAME,
        collection_name=collection_name,
    )
    keys = [key for key in docstore.yield_keys()]
    docs_processed = docstore.mget(keys)

    retriever = BM25Retriever.from_documents(docs_processed)
    retriever.k = top_k

    return retriever


def get_parent_retriever(
    embedding_model, embedding_model_name, collection_name, top_k=3
):
    """
    Initializes a parent retriever object configured to fetch the top_k most relevant documents based on cosine similarity.

    Parameters:
    - embedding_model: The embedding model used to generate document embeddings.
    - collection_name: The name of the collection in the vector store and document store.
    - top_k: The number of top relevant documents to retrieve. Defaults to 3.

    Returns:
    - An instance of ParentDocumentRetriever
    """
    parent_splitter = get_text_splitter(
        embedding_model_name,
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=0,
    )
    child_splitter = get_text_splitter(
        embedding_model_name,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0,
    )

    vectorstore = Milvus(
        embedding_model,
        connection_args={"uri": VECTOR_DATABASE_URI},
        collection_name=collection_name,
        auto_id=True,
    )
    docstore = MongoDBStore(
        connection_string=MONGODB_CONNECTION_STR,
        db_name=MONGODB_DATABASE_NAME,
        collection_name=collection_name,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        k=top_k,
    )

    return retriever


# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


def get_multi_query_retriever(base_retriever, llm, language):
    """
    Create a multi-query retriever based on the base retriever and LLM.

    Parameters:
    - base_retriever: base retriever
    - llm: LLM to generate variations of queries.

    Returns:
    - A retriever that is able to generate multiple queries.
    """
    output_parser = LineListOutputParser()
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=f"""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question in {language} to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {{question}}""",
    )
    llm_chain = QUERY_PROMPT | llm | output_parser

    retriever = MultiQueryRetriever(retriever=base_retriever, llm_chain=llm_chain)

    return retriever


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
    vectorstore = Milvus(
        hyde_embeddings,
        connection_args={"uri": VECTOR_DATABASE_URI},
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    return retriever


def get_ensemble_retriever(
    embedding_model,
    llm,
    language,
    collection_name,
    parent_collection_name,
    top_k=3,
):
    """
    Initializes an ensemble retriever that combines multiple retrieval strategies
    to fetch the top_k most relevant documents based on cosine similarity.

    Parameters:
    - embedding_model: The embedding model used to generate document embeddings.
    - llm: LLM used for generating query variations and hypothetical documents.
    - collection_name: The name of the collection in the vector store for basic and HyDE retrieval.
    - parent_collection_name: The name of the collection used in parent document retrieval and keyword search retrieval.
    - top_k: The number of top relevant documents to retrieve. Defaults to 3.

    Returns:
    - An instance of EnsembleRetriever.

    Raises:
    - ValueError: If `top_k` is less than 1 or any other input parameter is invalid.
    - Exception: If an error occurs during the initialization of the retriever.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if language == "english":
        embedding_model_name = EMBEDDING_MODEL_NAME_ENGLISH
    else:
        embedding_model_name = EMBEDDING_MODEL_NAME_SPANISH

    try:
        basic_retriever = get_basic_retriever(embedding_model, collection_name, top_k)
        keyword_search_retriever = get_keyword_search_retriever(
            parent_collection_name, top_k
        )
        parent_retriever = get_parent_retriever(
            embedding_model, embedding_model_name, parent_collection_name, top_k
        )
        multi_query_retriever = get_multi_query_retriever(
            parent_retriever, llm, language
        )
        hyde_retriever = get_hyde_retriever(
            embedding_model, llm, collection_name, language, top_k
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                basic_retriever,
                keyword_search_retriever,
                parent_retriever,
                multi_query_retriever,
                hyde_retriever,
            ],
            weights=[
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
            ],
        )

        return ensemble_retriever
    except Exception as e:
        print(f"An error occurred while initializing the retriever: {e}")
        raise
