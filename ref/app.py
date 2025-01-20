from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from jsonargparse import CLI
import os
from dotenv import load_dotenv

from coroutine_manager import coroutine_manager
from query_optimizer import QueryOptimizer
from time import time

from rag import (
    azure_openai_call,
    get_ensemble_retriever,
    process_queries_and_combine_results
)

from EmbeddingManager import EmbeddingManager

# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
GERMAN_EMBEDDING_MODEL_NAME = os.getenv("GERMAN_EMBEDDING_MODEL_NAME")
ENGLISH_EMBEDDING_MODEL_NAME = os.getenv("ENGLISH_EMBEDDING_MODEL_NAME")

# Al principio del archivo, después de las importaciones
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
RERANKING_TYPE = os.getenv("RERANKING_TYPE")
#MIN_RERANKING_SCORE = float(os.getenv("MIN_RERANKING_SCORE", 0.5))  # Convertir a flotante con valor por defecto
MAX_CHUNKS_CONSIDERED = int(os.getenv("MAX_CHUNKS_CONSIDERED", 3))  # Convertir a entero con valor por defecto

DIRECTORY_PATH = os.getenv("DIRECTORY_PATH")

# Códigos ANSI para colores y texto en negrita
BLUE = "\033[34m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RESET = "\033[0m"  # Para resetear el formato

LANGUAGE = "german"

# Inicializar el optimizador de consultas
query_optimizer = QueryOptimizer()
embedding_manager = EmbeddingManager()
embedding_manager.initialize_models(GERMAN_EMBEDDING_MODEL_NAME, ENGLISH_EMBEDDING_MODEL_NAME)

async def cleanup_resources():
    try:
        await coroutine_manager.cleanup()
    except Exception as e:
        print(f"Error during cleanup: {e}")

async def main(
    directory: str = DIRECTORY_PATH
):
    try:
        print("\n")
        show_sources = True
        
        # 1. Limpiar caché al inicio de la sesión
        entries_removed = query_optimizer.cleanup_cache()
        if entries_removed > 0:
            print(f"{BLUE}{BOLD}>> Cache bereinigt: {entries_removed} Einträge entfernt{RESET}")

        llm = (lambda x: azure_openai_call(x))  # Envolver la llamada en una función lambda

        # Ensemble Retrieval
        # En la función main, ajustar los parámetros de concurrencia
        german_retriever = await get_ensemble_retriever(
            f"{directory}/de",
            embedding_manager.german_model,
            llm,
            collection_name=f"{COLLECTION_NAME}_de",
            top_k=MAX_CHUNKS_CONSIDERED,
            language="german",
            max_concurrency=3  # Reducir la concurrencia
        )

        english_retriever = await get_ensemble_retriever(
            f"{directory}/en",
            embedding_manager.english_model,
            llm,
            collection_name=f"{COLLECTION_NAME}_en",
            top_k=MAX_CHUNKS_CONSIDERED,
            language="english",
            max_concurrency=3
        )

        print(f"\n{BLUE}{BOLD}------------------------- Willkommen im UniChatBot -------------------------{RESET}")
        
        # Añadir una lista para mantener el historial
        chat_history = []

        # Variables para control de limpieza periódica
        last_cleanup_time = time()
        CLEANUP_INTERVAL = 3600  # 1 hora en segundos

        while True:
            current_time = time()

            # 2. Limpieza periódica del caché
            if current_time - last_cleanup_time > CLEANUP_INTERVAL:
                entries_removed = query_optimizer.cleanup_cache()
                if entries_removed > 0:
                    print(f"{BLUE}{BOLD}>> Periodische Cache-Bereinigung: {entries_removed} Einträge entfernt{RESET}")
                last_cleanup_time = current_time

            print("\n")

            # Imprimir "Benutzer-Eingabe: " en azul y negrita
            print(f"{BLUE}{BOLD}>> Benutzer-Eingabe: {RESET}", end="")
            
            # Capturar la entrada del usuario y mostrarla en amarillo y negrita
            query = input(f"{ORANGE}{BOLD}")
            print(f"{RESET}")  # Resetear el formato después de la entrada
            
            if query.lower() in ["exit", "cls"]:
                break

            # Mostrar estadísticas del caché si se solicita
            if query.lower() == "cache stats":
                stats = query_optimizer.get_cache_stats()
                print(f"{BLUE}{BOLD}>> Cache-Statistiken:{RESET}")
                for key, value in stats.items():
                    print(f"- {key}: {value}")
                continue

            # Iniciamos el contador de tiempo
            start_time = time()
            
            # Llamada asincrónica
            context = await process_queries_and_combine_results(
                query,
                llm,
                german_retriever,
                english_retriever,
                RERANKING_TYPE,
                RERANKING_TYPE,
                chat_history,
                LANGUAGE,
                embedding_manager
            )

            if context.get('from_cache'):
                print(f"{BLUE}{BOLD}\n>> LLM-Antwort (aus Cache): {RESET}", end="")
                
                if isinstance(context, dict):
                    print(context.get('response', ''))  # Accedemos específicamente a la respuesta
                else:
                    print(context)
                
                if show_sources:
                    # Mostrar fuentes
                    print(f"{BLUE}{BOLD}>> Quellen:{RESET}")
                    for source in context['sources']:
                        if source.get('sheet_name'):
                            print(f"- Dokument: {os.path.basename(source['source'])} (Blatt: {source['sheet_name']})")
                        else:
                            print(f"- Dokument: {os.path.basename(source['source'])} (Seite: {source['page_number']})")
                    
                    # Guardar en historial
                    chat_history.append((query, context['response']))
            else:
                print(f"{BLUE}{BOLD}\n>> LLM-Antwort: {RESET}", end="")

                # Recolectar la respuesta del streaming
                response = context['response']
                print(response)

                # Guardar en historial de manera consistente
                chat_history.append((query, response))

                if show_sources:
                    print(f"{BLUE}{BOLD}>> Quellen:{RESET}")
                    unique_sources = {}
                    for document in context['sources']:
                        source = os.path.basename(document['source'])
                        if document['source'].lower().endswith('.xlsx'):
                            sheet = document.get('sheet_name', 'Unbekannt')
                            key = (source, sheet)
                            if sheet != 'Unbekannt':
                                unique_sources[key] = f"- Dokument: {source} (Blatt: {sheet})"
                            else:
                                unique_sources[key] = f"- Dokument: {source}"
                        else:
                            page = document.get('page_number', 'Unbekannt')
                            key = (source, page)
                            if page != 'Unbekannt':
                                unique_sources[key] = f"- Dokument: {source} (Seite: {page})"
                            else:
                                unique_sources[key] = f"- Dokument: {source}"
                    
                    for source in unique_sources.values():
                        print(source)

            # Registrar el tiempo de finalización
            end_time = time()

            # Guardar la interacción en el historial
            chat_history.append((query, response))  # response es la respuesta del LLM
            
            # Calcular y mostrar el tiempo de respuesta
            processing_time = end_time - start_time

            if context.get('from_cache'):
                print(f"\n{BLUE}{BOLD}>> Cache-Antwort in {GREEN}{processing_time:.2f}{BLUE} Sekunden gefunden.{RESET}")
            else:
                print(f"\n{BLUE}{BOLD}>> Es dauerte {GREEN}{processing_time:.2f}{BLUE} Sekunden, um zu antworten.{RESET}")
            
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
    finally:
        await cleanup_resources()

if __name__ == "__main__":

    CLI(main)
