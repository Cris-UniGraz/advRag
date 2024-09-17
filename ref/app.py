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
    get_ensemble_retriever,
    get_ensemble_retriever_check
)

# Al principio del archivo, después de las importaciones
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
RERANKING_TYPE = os.getenv("RERANKING_TYPE")
MIN_RERANKING_SCORE = float(os.getenv("MIN_RERANKING_SCORE", 0.5))  # Convertir a flotante con valor por defecto
MAX_CHUNKS_CONSIDERED = int(os.getenv("MAX_CHUNKS_CONSIDERED", 3))  # Convertir a entero con valor por defecto
DIRECTORY_PATH = os.getenv("DIRECTORY_PATH")

# Códigos ANSI para color azul y texto en negrita
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"  # Para resetear el formato

def main(
    directory: str = DIRECTORY_PATH
):
    llm = (lambda x: azure_openai_call(x))  # Envolver la llamada en una función lambda
    embedding_model = load_embedding_model(model_name=EMBEDDING_MODEL_NAME)

    #docs = load_documents(folder_path=directory)

    #Ensemble Retrieval
    # retriever = get_ensemble_retriever(docs, embedding_model, llm, collection_name=COLLECTION_NAME, top_k=MAX_CHUNKS_CONSIDERED)

    retriever = get_ensemble_retriever_check(directory, embedding_model, llm, collection_name=COLLECTION_NAME, top_k=MAX_CHUNKS_CONSIDERED)

    prompt_template = ChatPromptTemplate.from_template(
        (
            """
            Du bist eine erfahrene virtuelle Assistentin der Universität Graz und kennst alle Informationen über die Universität Graz. Deine Aufgabe ist es, auf der Grundlage der Benutzerfrage Informationen aus dem bereitgestellten KONTEXT zu extrahieren. 
            Denk Schritt für Schritt und verwende nur die Informationen aus dem KONTEXT, die für die Benutzerfrage relevant sind. 
            Wenn der KONTEXT keine Informationen enthält, um die ANFRAGE zu beantworten, gib nicht dein Wissen an, sondern antworte einfach:
            Ich habe derzeit nicht genügend Informationen, um die Anfrage zu beantworten. Bitte stelle eine andere Anfrage.
            Gib detaillierte Antworten auf Deutsch.

            ANFRAGE: ```{question}```\n
            KONTEXT: ```{context}```\n
            """
        )
    )

    chain = prompt_template | llm | StrOutputParser()
  
    while True:
        print("\n\n")

        # query = input("Benutzer-Eingabe: ")
        # Aplicar el estilo a la solicitud de entrada
        styled_prompt = f"{BLUE}{BOLD}Benutzer-Eingabe: {RESET}"
        # Solicitar la entrada al usuario con el estilo aplicado
        query = input(styled_prompt)

        if query == "exit":
            break

        context = retrieve_context_reranked(
            query, retriever=retriever, reranker_model=RERANKING_TYPE
        )

        text = ""
        sources = []
        filtered_context = []
        for document in context:
            if len(filtered_context) < MAX_CHUNKS_CONSIDERED and document.metadata.get('reranking_score', 0) > MIN_RERANKING_SCORE:
                text += "\n" + document.page_content
                source = f"{os.path.basename(document.metadata['source'])} (Seite {document.metadata.get('page', 'N/A')})"
                if source not in sources:
                    sources.append(source)
                filtered_context.append(document)

        # print("\n\nLLM-Antwort: ", end="")
        # Imprimir texto azul en negrita
        print(f"{BLUE}{BOLD}\n\nLLM-Antwort: {RESET}", end="")

        for e in chain.stream({"context": text, "question": query}):
            print(e, end="")
        print("\n\n")

        show_sources = True

        if show_sources:
            print(f"{BLUE}{BOLD}--------------------------------QUELLEN-------------------------------------{RESET}")
            for document in filtered_context:
                source = os.path.basename(document.metadata['source'])
                if document.metadata['source'].lower().endswith('.xlsx'):
                    sheet = document.metadata.get('sheet', 'Unbekannt')
                    print(f"- Dokument: {source} (Blatt: {sheet})")
                else:
                    page = document.metadata.get('page', 'N/A')
                    print(f"- Dokument: {source} (Seite: {page})")
        
        print("\n\n\n")

        show_chunks = False
        if show_chunks:
            print("\n\n\n--------------------------------CONTEXT-------------------------------------")
            for i, chunk in enumerate(filtered_context):
                print(f"-----------------------------------Chunk: {i}--------------------------------------")
                print(f"Context: {chunk.page_content}")
                print(f"Reranking Score: {chunk.metadata.get('reranking_score', 'N/A')}")
            print("\n\n\n")
    

if __name__ == "__main__":

    CLI(main)
