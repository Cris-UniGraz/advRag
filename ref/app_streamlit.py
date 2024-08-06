# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os
import base64
import gc
import tempfile
import uuid
import streamlit as st
from tkinter import filedialog
from rag_client import rag_client


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.client = None  # Añade esta línea

session_id = st.session_state.id

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

with st.sidebar:
    st.write("Seleccione una carpeta:")
    folder_path = st.text_input("Ruta de la carpeta:", key="folder_input")
    
    if st.button("Procesar documentos"):
        if folder_path and os.path.isdir(folder_path):
            with st.status("Procesando sus documentos", expanded=False, state="running"):
                st.write("Indexación en progreso...")
                if folder_path not in st.session_state.file_cache:
                    st.session_state.client = rag_client(folder_path=folder_path)
                    st.session_state.file_cache[folder_path] = st.session_state.client
                else:
                    st.session_state.client = st.session_state.file_cache[folder_path]
                st.write("Procesamiento completo, haga sus preguntas...")
        else:
            st.error("Por favor, ingrese una ruta de carpeta válida.")

    if 'folder_path' in st.session_state:
        st.write(f"Carpeta actual: {st.session_state.folder_path}")

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chatten mit Dateien")

with col2:
    st.button("Löschen ↺", on_click=reset_chat)


# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("¿Qué tiene en mente?"):
    if not folder_path or not os.path.isdir(folder_path):
        st.exception(FileNotFoundError("¡Por favor, ingrese una ruta de carpeta válida primero!"))
        st.stop()

    if st.session_state.client is None:
        st.error("El cliente no se ha inicializado. Por favor, procese los documentos primero.")
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in st.session_state.client.stream(prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
