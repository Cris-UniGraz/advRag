from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank

from config.constants import RERANKER_MODEL_NAME


def rerank_docs(query, retrieved_docs, retriever, client):
    """
    Rerank the provided context chunks

    Parameters:
    - query: user query - string
    - retrieved_docs: chunks that needs to be ranked.
    - retriever: Se utiliza en Jina Rerank


    Returns:
    - Sorted list of chunks based on their relevance to the query.
    """
    if RERANKER_MODEL_NAME == "cohere":
        ranked_docs = reranking_cohere(retrieved_docs, query, client)
    elif RERANKER_MODEL_NAME == "jina":
        ranked_docs = reranking_jina(retriever, query)
    else:
        # Just return the original order without rerank.
        ranked_docs = retrieved_docs

    return ranked_docs


def reranking_cohere(similar_chunks, query, client):
    documents = [f"{doc.page_content}" for doc in similar_chunks]
    results = client.rerank(
        query=query,
        documents=documents,
        top_n=5,
        model="rerank-multilingual-v3.0",
        return_documents=True,
    )

    documents = []
    for idx, r in enumerate(results.results):
        # Solo los chunks más relevantes se pasan al LLM.
        if float(r.relevance_score) > 0.5:
            original_index = int(r.index)
            doc = similar_chunks[original_index]
            # doc.metadata["relevance_score"] = float(r.relevance_score)
            documents.append(doc)

    return documents


def reranking_jina(retriever, query):
    # TODO: Obtener los chunks con el número de página y filtrar por relevancia.
    compressor = JinaRerank(model="jina-reranker-v2-base-multilingual")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    compressed_docs = compression_retriever.invoke(query)

    documents = []
    for idx, d in enumerate(compressed_docs):
        documents.append(f"{d.page_content}")

    return documents
