from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from jsonargparse import CLI
import os
from dotenv import load_dotenv

from rag import (
    load_embedding_model,
    azure_openai_call,
    get_ensemble_retriever,
    process_queries_and_combine_results,
)

# Al principio del archivo, después de las importaciones
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
GERMAN_EMBEDDING_MODEL_NAME = os.getenv("GERMAN_EMBEDDING_MODEL_NAME")
ENGLISH_EMBEDDING_MODEL_NAME = os.getenv("ENGLISH_EMBEDDING_MODEL_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
RERANKING_TYPE = os.getenv("RERANKING_TYPE")
MIN_RERANKING_SCORE = float(os.getenv("MIN_RERANKING_SCORE", 0.5))  # Convertir a flotante con valor por defecto
MAX_CHUNKS_CONSIDERED = int(os.getenv("MAX_CHUNKS_CONSIDERED", 3))  # Convertir a entero con valor por defecto
MAX_CHUNKS_LLM = int(os.getenv("MAX_CHUNKS_LLM", 3))  # Convertir a entero con valor por defecto
DIRECTORY_PATH = os.getenv("DIRECTORY_PATH")

# Códigos ANSI para colores y texto en negrita
BLUE = "\033[34m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RESET = "\033[0m"  # Para resetear el formato

LANGUAGE = "german"

async def main(
    directory: str = DIRECTORY_PATH
):
    print("\n")

    llm = (lambda x: azure_openai_call(x))  # Envolver la llamada en una función lambda
    german_embedding_model = load_embedding_model(model_name=GERMAN_EMBEDDING_MODEL_NAME)
    english_embedding_model = load_embedding_model(model_name=ENGLISH_EMBEDDING_MODEL_NAME)

    # Ensemble Retrieval
    german_retriever = await get_ensemble_retriever(
        f"{directory}/de",
        german_embedding_model,
        llm,
        collection_name=f"{COLLECTION_NAME}_de",
        top_k=MAX_CHUNKS_CONSIDERED,
        language="german"
    )
    english_retriever = await get_ensemble_retriever(
        f"{directory}/en",
        english_embedding_model,
        llm,
        collection_name=f"{COLLECTION_NAME}_en",
        top_k=MAX_CHUNKS_CONSIDERED,
        language="english"
    )

    prompt_template = ChatPromptTemplate.from_template(
        (
            """
            You are an experienced virtual assistant at the University of Graz and know all the information about the University of Graz.
            Your main task is to extract information from the provided CONTEXT based on the user's QUERY.
            Think step by step and only use the information from the CONTEXT that is relevant to the user's QUERY.
            If the CONTEXT does not contain information to answer the QUESTION, do not state your knowledge, just answer: Ich habe derzeit nicht genügend Informationen, um die Anfrage zu beantworten. Bitte stelle eine andere Anfrage.
            Give detailed answers in {language}.

            QUERY: ```{question}```\n
            CONTEXT: ```{context}```\n
            """
        )
    )

    chain = prompt_template | llm | StrOutputParser()

    print(f"\n{BLUE}{BOLD}------------------------- Willkommen im UniChatBot -------------------------{RESET}")
    
    # Añadir una lista para mantener el historial
    chat_history = []

    while True:
        print("\n")

        # Imprimir "Benutzer-Eingabe: " en azul y negrita
        print(f"{BLUE}{BOLD}>> Benutzer-Eingabe: {RESET}", end="")
        
        # Capturar la entrada del usuario y mostrarla en amarillo y negrita
        query = input(f"{ORANGE}{BOLD}")
        print(f"{RESET}")  # Resetear el formato después de la entrada
        
        if query == "exit":
            break
        
        # Llamada asincrónica
        context = await process_queries_and_combine_results(
            query,
            llm,
            german_retriever,
            english_retriever,
            RERANKING_TYPE,
            RERANKING_TYPE,
            chat_history,
            LANGUAGE
        )

        text = ""
        sources = []
        filtered_context = []
        for document in context:
            if len(filtered_context) <= MAX_CHUNKS_LLM: #and document.metadata.get('reranking_score', 0) > MIN_RERANKING_SCORE:
                text += "\n" + document.page_content
                source = f"{os.path.basename(document.metadata['source'])} (Seite {document.metadata.get('page', 'N/A')})"
                if source not in sources:
                    sources.append(source)
                filtered_context.append(document)

        # Imprimir "LLM-Antwort:" en azul negrita
        print(f"{BLUE}{BOLD}\n>> LLM-Antwort: {RESET}", end="")

        # Recolectar la respuesta del streaming
        response = ""
        for chunk in chain.stream({"context": text, "language": LANGUAGE, "question": query}):
            print(chunk, end="")
            response += chunk
        print("\n")

        # Guardar la interacción en el historial
        chat_history.append((query, response))  # response es la respuesta del LLM

        show_sources = True

        if show_sources:
            print(f"{BLUE}{BOLD}>> Quellen:{RESET}")
            unique_sources = {}
            for document in filtered_context:
                source = os.path.basename(document.metadata['source'])
                if document.metadata['source'].lower().endswith('.xlsx'):
                    sheet = document.metadata.get('sheet_name', 'Unbekannt')
                    key = (source, sheet)
                    if sheet != 'Unbekannt':
                        unique_sources[key] = f"- Dokument: {source} (Blatt: {sheet})"
                    else:
                        unique_sources[key] = f"- Dokument: {source}"
                else:
                    page = document.metadata.get('page_number', 'Unbekannt')
                    key = (source, page)
                    if page != 'Unbekannt':
                        unique_sources[key] = f"- Dokument: {source} (Seite: {page})"
                    else:
                        unique_sources[key] = f"- Dokument: {source}"
            
            for source in unique_sources.values():
                print(source)
        
        print(f"\n{BLUE}{BOLD}----------------------------------------------------------------------------{RESET}")
        print("\n")

        show_chunks = False
        if show_chunks:
            print("\n\n\n--------------------------------CONTEXT-------------------------------------")
            for i, chunk in enumerate(context):
                print(f"-----------------------------------Chunk: {i}--------------------------------------")
                print(f"Source: {os.path.basename(chunk.metadata.get('source', 'N/A'))}")
                print(f"Context: {chunk.page_content}")
                print(f"Reranking Score: {chunk.metadata.get('reranking_score', 'N/A')}")
            print("\n\n\n")
    

if __name__ == "__main__":

    CLI(main)
