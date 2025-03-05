from langchain_milvus import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


from .splitters import get_text_splitter
from .retrievers import get_parent_retriever
from .reranking import rerank_docs
from config.constants import (
    VECTOR_DATABASE_URI,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME_SPANISH,
    EMBEDDING_MODEL_NAME_ENGLISH,
)


def split_documents(documents, embedding_model_name):
    text_splitter = get_text_splitter(
        embedding_model_name,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0,
    )

    # Flatten list and remove duplicates more efficiently
    unique_texts = set()
    docs_processed_unique = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            if chunk.page_content not in unique_texts:
                unique_texts.add(chunk.page_content)
                docs_processed_unique.append(chunk)

    return docs_processed_unique


def create_collection_from_documents(
    documents, embedding_model, collection_name, language
):
    if language == "english":
        embedding_model_name = EMBEDDING_MODEL_NAME_ENGLISH
    else:
        embedding_model_name = EMBEDDING_MODEL_NAME_SPANISH

    docs_processed = split_documents(documents, embedding_model_name)
    vectorstore = Milvus.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        collection_name=collection_name,
        connection_args={"uri": VECTOR_DATABASE_URI},
    )

    return True


def create_parent_collection_from_documents(
    documents, embedding_model, collection_name, language
):
    if language == "english":
        embedding_model_name = EMBEDDING_MODEL_NAME_ENGLISH
    else:
        embedding_model_name = EMBEDDING_MODEL_NAME_SPANISH

    parent_retriever = get_parent_retriever(
        embedding_model, embedding_model_name, collection_name
    )
    parent_retriever.add_documents(documents)

    return True


def retrieve_reranked_docs(query, retriever, client):
    retrieved_docs = retriever.invoke(query)
    if len(retrieved_docs) == 0:
        print(
            f"Couldn't retrieve any relevant document with the query `{query}`. Try modifying your question!"
        )

    reranked_docs = rerank_docs(
        query=query,
        retrieved_docs=retrieved_docs,
        retriever=retriever,
        client=client,
    )

    return reranked_docs


def get_context(documents):
    context = ""
    for doc in documents:
        context += "\n" + doc.page_content

    return context


def get_sources(documents):
    sources = []
    unique_sources = set()

    for doc in documents:
        file_type = doc.metadata["file_type"]
        source = doc.metadata["source"]
        page_number = doc.metadata["page_number"]
        sheet_name = doc.metadata["sheet_name"]

        s = (file_type, source, page_number, sheet_name)
        if s not in unique_sources:
            unique_sources.add(s)
            sources.append(
                {
                    "file_type": file_type,
                    "source": source,
                    "page_number": page_number,
                    "sheet_name": sheet_name,
                }
            )

    sources_sorted = sorted(sources, key=lambda x: x["page_number"])

    return sources_sorted


def get_translated_query(query, llm):
    prompt_template = ChatPromptTemplate.from_template(
        """
    Traduce la siguiente PREGUNTA del español al inglés, sin agregar ningún texto adicional.
    No incluyas razonamientos, ni respondas la pregunta.
    Devuelve únicamente la traducción de la PREGUNTA, sin preámbulos ni etiquetas.

    PREGUNTA: {question}
    """
    )

    chain = prompt_template | llm | StrOutputParser()
    translated_query = chain.invoke({"question": query})
    translated_query = translated_query.strip()

    return translated_query


def get_response_from_llm(
    query,
    context,
    llm,
    previous_questions,
    previous_answers,
):
    previous_messages = []
    for h_msg, ai_msg in zip(previous_questions, previous_answers):
        previous_messages.append(HumanMessage(content=h_msg))
        previous_messages.append(AIMessage(content=ai_msg))

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""
    Eres un investigador académico experimentado. Tu tarea es extraer información del CONTEXTO proporcionado basado en la pregunta del usuario.
    Piensa paso a paso y utiliza solo la información del CONTEXTO que sea relevante para la pregunta del usuario. 
    Proporciona respuestas detalladas.
    Responde siempre en español. 
    Solo responde la pregunta, no incluyas tus razonamientos en las respuestas.
    Si no hay contexto, intenta responder a la pregunta con tus conocimientos, pero solo si la respuesta es adecuada.

    PREGUNTA: ```{question}```\n
    CONTEXTO: ```{context}```\n"""
            ),
            *previous_messages,
            HumanMessage(
                content=f"PREGUNTA: ```{query}```\nCONTEXTO: ```{context}```\n"
            ),
        ]
    )

    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({})
    response = response.strip()

    return response
