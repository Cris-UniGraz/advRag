from transformers import AutoTokenizer, AutoModel
#from openai import OpenAI
from openai import AzureOpenAI
import torch
import time
import json
import cohere
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document

# Al principio del archivo, después de las importaciones
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

GERMAN_RERANKING_MODEL_NAME = os.getenv("GERMAN_RERANKING_MODEL_NAME")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")

# Códigos ANSI para colores y texto en negrita
GREEN = "\033[32m"
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"  # Para resetear el formato

# Function to compute MaxSim
def maxsim(query_embedding, document_embedding):
    # Expand dimensions for broadcasting
    # Query: [batch_size, query_length, embedding_size] -> [batch_size, query_length, 1, embedding_size]
    # Document: [batch_size, doc_length, embedding_size] -> [batch_size, 1, doc_length, embedding_size]
    expanded_query = query_embedding.unsqueeze(2)
    expanded_doc = document_embedding.unsqueeze(1)

    # Compute cosine similarity across the embedding dimension
    sim_matrix = torch.nn.functional.cosine_similarity(expanded_query, expanded_doc, dim=-1)

    # Take the maximum similarity for each query token (across all document tokens)
    # sim_matrix shape: [batch_size, query_length, doc_length]
    max_sim_scores, _ = torch.max(sim_matrix, dim=2)

    # Average these maximum scores across all query tokens
    avg_max_sim = torch.mean(max_sim_scores, dim=1)
    return avg_max_sim

def reranking_gpt(similar_chunks, query):
    start = time.time()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2023-05-15",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    response = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
        {"role": "system", 
        "content": """Du bist ein Experte für Relevanzbewertung. Anhand einer Liste von Dokumenten und einer Abfrage musst du bestimmen, wie relevant jedes Dokument für die Beantwortung der Abfrage ist. 
        Deine Ausgabe ist JSON, d.h. eine Liste von Dokumenten.  Jedes Dokument hat zwei Felder: Inhalt und Punktzahl. relevance_score liegt zwischen 0,0 und 100,0. Höhere Relevanz bedeutet höhere Punktzahl"""},
        {"role": "user", "content": f"Query: {query} Docs: {[doc.page_content for doc in similar_chunks]}"}
        ]
    )

    print("Es dauerte {:.2f} Sekunden, um Dokumente mit {} zu re-ranken.".format(time.time() - start, AZURE_OPENAI_MODEL))

    scores = json.loads(response.choices[0].message.content)["documents"]
    sorted_data = sorted(scores, key=lambda x: x['score'], reverse=True)

    reranked_documents = [
        Document(page_content=r['content'], metadata=similar_chunks[i].metadata)
        for i, r in enumerate(sorted_data)
    ]
    
    return reranked_documents


def reranking_german(similar_chunks, query):
    start = time.time()
    scores = []
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(GERMAN_RERANKING_MODEL_NAME)
    model = AutoModel.from_pretrained(GERMAN_RERANKING_MODEL_NAME)
    # Encode the query
    query_encoding = tokenizer(query, return_tensors='pt')
    query_embedding = model(**query_encoding).last_hidden_state.mean(dim=1)
    # Get score for each document
    for document in similar_chunks:
        document_encoding = tokenizer(document.page_content, return_tensors='pt', truncation=True, max_length=512)
        document_embedding = model(**document_encoding).last_hidden_state
        # Calculate MaxSim score
        score = maxsim(query_embedding.unsqueeze(0), document_embedding)
        scores.append({
            "score": score.item(),
            "document": document,
        })
    
    print(f"{BLUE}{BOLD}Es dauerte {RESET}{GREEN}{BOLD}{time.time() - start:.2f} Sekunden{RESET}{BLUE}{BOLD}, um Dokumente mit {RESET}{GREEN}{BOLD}{GERMAN_RERANKING_MODEL_NAME}{RESET}{BLUE}{BOLD} zu re-ranken.{RESET}")

    # Sort the scores by highest to lowest
    sorted_data = sorted(scores, key=lambda x: x['score'], reverse=True)
    
    # Create a list of Document objects with score included in metadata
    reranked_documents = [
        Document(
            page_content=r['document'].page_content, 
            metadata={**r['document'].metadata, "reranking_score": r['score']}
        )
        for r in sorted_data
    ]
    
    return reranked_documents

def reranking_colbert(similar_chunks, query):
    start = time.time()
    scores = []
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
    # Encode the query
    query_encoding = tokenizer(query, return_tensors='pt')
    query_embedding = model(**query_encoding).last_hidden_state.mean(dim=1)
    # Get score for each document
    for document in similar_chunks:
        document_encoding = tokenizer(document.page_content, return_tensors='pt', truncation=True, max_length=512)
        document_embedding = model(**document_encoding).last_hidden_state
        # Calculate MaxSim score
        score = maxsim(query_embedding.unsqueeze(0), document_embedding)
        scores.append({
            "score": score.item(),
            "document": document,
        })
    
    print(f"{BLUE}{BOLD}Es dauerte {RESET}{GREEN}{BOLD}{time.time() - start:.2f} Sekunden{RESET}{BLUE}{BOLD}, um Dokumente mit {RESET}{GREEN}{BOLD}{GERMAN_RERANKING_MODEL_NAME}{RESET}{BLUE}{BOLD} zu re-ranken.{RESET}")

    # Sort the scores by highest to lowest
    sorted_data = sorted(scores, key=lambda x: x['score'], reverse=True)
    
    # Create a list of Document objects with score included in metadata
    reranked_documents = [
        Document(
            page_content=r['document'].page_content, 
            metadata={**r['document'].metadata, "reranking_score": r['score']}
        )
        for r in sorted_data
    ]
    
    return reranked_documents


def reranking_cohere(similar_chunks, query):
    co = cohere.Client(os.environ["COHERE_API_KEY"])

    documents = [doc.page_content for doc in similar_chunks]
    start = time.time()

    results = co.rerank(query=query, 
                        documents=documents, 
                        model="rerank-multilingual-v3.0", 
                        return_documents=True)

    # print(f"{BLUE}{BOLD}Es dauerte {RESET}{GREEN}{BOLD}{time.time() - start:.2f} Sekunden{RESET}{BLUE}{BOLD}, um Dokumente mit {RESET}{GREEN}{BOLD}Cohere{RESET}{BLUE}{BOLD} zu re-ranken.{RESET}")

    reranked_documents = [
        Document(
            page_content=r.document.text, 
            metadata={
                **similar_chunks[r.index].metadata, 
                "reranking_score": r.relevance_score
            }
        )
        for r in results.results
    ]

    return reranked_documents


