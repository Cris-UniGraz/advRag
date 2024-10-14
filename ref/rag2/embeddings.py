import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, JinaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings


def load_embedding_model(model_name):
    """
    Loads an embedding model from the Hugging Face repository with specified configurations.

    Parameters:
    - model_name: The name of the model to load.

    Returns:
    - An instance of HuggingFaceBgeEmbeddings configured with the specified model and device.

    Raises:
    - ValueError: If an unsupported device is specified.
    - OSError: If the model cannot be loaded from the Hugging Face repository.
    """

    try:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model_kwargs = {"device": device}
        encode_kwargs = {
            "normalize_embeddings": True  # For cosine similarity computation
        }

        if model_name == "openai":
            embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        elif model_name == "hiiamsid/sentence_similarity_spanish_es":
            # Retorna un modelo de tipo HuggingFaceEmbeddings (SentenceTransformer)
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        elif model_name == "jinaai/jina-embeddings-v2-base-es":
            # Retorna un modelo de tipo JinaEmbeddings (Es una request a la API)
            embedding_model = JinaEmbeddings(model_name="jina-embeddings-v2-base-es")
        elif model_name == "Cohere/Cohere-embed-multilingual-v3.0":
            # Retorna un modelo de tipo CohereEmbeddings (Es una request a la API)
            embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0")
        elif model_name == "Cohere/Cohere-embed-english-v3.0":
            embedding_model = CohereEmbeddings(model="embed-english-v3.0")
        else:
            embedding_model = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

        return embedding_model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise
