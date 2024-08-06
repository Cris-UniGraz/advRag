from langchain.callbacks import FileCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from loguru import logger
from dotenv import load_dotenv
import os

from rag import (
    create_parent_retriever,
    load_embedding_model,
    load_documents,
    retrieve_context_reranked,
    create_multi_query_retriever,
    azure_openai_call
)

# Al principio del archivo, después de las importaciones
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
RERANKING_TYPE = os.getenv("RERANKING_TYPE")
MAX_CHUNKS_CONSIDERED = int(os.getenv("MAX_CHUNKS_CONSIDERED", 3))  # Convertir a entero con valor por defecto


class rag_client:
    embedding_model = load_embedding_model(model_name=EMBEDDING_MODEL_NAME) #model_name="openai"

    def __init__(self, folder_path):
        docs = load_documents(folder_path)
        self.retriever = create_parent_retriever(docs, self.embedding_model)
        
        # llm = ChatOpenAI(model_name="gpt-4o")
        # llm = ChatOllama(model="llama3")
        llm = (lambda x: azure_openai_call(x))  # Envolver la llamada en una función lambda

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
        self.chain = prompt_template | llm | StrOutputParser()

    def stream(self, query):
        try:
            context_list = self.retrieve_context_reranked(query)
            print(f"Context: {context_list}")
            context = ""
            for i,cont in enumerate(context_list):
                if i < MAX_CHUNKS_CONSIDERED:
                    context = context +"\n"+ cont
                else:
                    break
            print(context)
        except Exception as e:
            context = e.args[0]
        logger.info(context)
        for r in self.chain.stream({"context": context, "question": query}):
            yield r

    def retrieve_context_reranked(self, query):
        return retrieve_context_reranked(
            query, retriever=self.retriever, reranker_model=RERANKING_TYPE #"colbert" # reranker_model="cohere" # colbert for local model
        )

    def generate(self, query):
        contexts = self.retrieve_context_reranked(query)
        # print(contexts)
        text = ""
        for i,cont in enumerate(contexts):
            if i <3:
                text = text +"\n"+ cont
            else:
                break
        print(f"Here is the text: {text}")
        return {
            "contexts": text,
            "response": self.chain.invoke(
                {"context": text, "question": query}
            ),
        }
