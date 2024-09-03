from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from jsonargparse import CLI
import os
from dotenv import load_dotenv

from rag import (
    create_parent_retriever,
    load_embedding_model,
    load_documents,
    retrieve_context_reranked,
    create_multi_query_retriever,
    azure_openai_call,
    get_ensemble_retriever
)

# Al principio del archivo, después de las importaciones
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
RERANKING_TYPE = os.getenv("RERANKING_TYPE")
MAX_CHUNKS_CONSIDERED = int(os.getenv("MAX_CHUNKS_CONSIDERED", 3))  # Convertir a entero con valor por defecto

def main(
    directory: str = "C:/Pruebas/RAG Search/demo_docu", #"data/8a9ebed0-815a-469a-87eb-1767d21d8cec.pdf"
):
    llm = (lambda x: azure_openai_call(x))  # Envolver la llamada en una función lambda
    docs = load_documents(folder_path=directory)

    # print(f"\n\n>>> Documentos a procesar: {len(docs)}\n\n")

    embedding_model = load_embedding_model(model_name=EMBEDDING_MODEL_NAME)

    #Parent Document Retrieval
    #base_retriever = create_parent_retriever(docs, embedding_model, collection_name=COLLECTION_NAME)
    #retriever = create_multi_query_retriever(base_retriever, llm)

    #Fusion Retrieval
    retriever = get_ensemble_retriever(docs, embedding_model, llm, collection_name=COLLECTION_NAME, top_k=5)

    prompt_template = ChatPromptTemplate.from_template(
        (
            """
            Du bist eine erfahrene virtuelle Assistentin der Universität Graz und kennst alle Informationen über die Universität Graz. Deine Aufgabe ist es, auf der Grundlage der Benutzerfrage Informationen aus dem bereitgestellten KONTEXT zu extrahieren. 
            Denk Schritt für Schritt und verwende nur die Informationen aus dem KONTEXT, die für die Benutzerfrage relevant sind. Gib detaillierte Antworten auf Deutsch.  

            ANFRAGE: ```{question}```\n
            KONTEXT: ```{context}```\n
            """
        )
    )

    chain = prompt_template | llm | StrOutputParser()
  
    while True:
        query = input("Benutzer-Eingabe: ")

        if query == "exit":
            break

        context = retrieve_context_reranked(
            query, retriever=retriever, reranker_model=RERANKING_TYPE
        )

        text = ""
        sources = []
        for i, document in enumerate(context):
            if i < MAX_CHUNKS_CONSIDERED:
                text += "\n" + document.page_content
                source = f"{os.path.basename(document.metadata['source'])} (Seite {document.metadata.get('page', 'N/A')})"
                if source not in sources:
                    sources.append(source)
            else:
                break

        print("\n\nLLM-Antwort: ", end="")
        for e in chain.stream({"context": text, "question": query}):
            print(e, end="")
        print("\n\n\n")

        show_sources = True

        if show_sources:
            print("\n\n\n--------------------------------QUELLEN-------------------------------------")
            for source in sources:
                print(f"Dokument: {source}")


if __name__ == "__main__":

    CLI(main)
