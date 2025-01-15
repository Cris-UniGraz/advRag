import os
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
AZURE_OPENAI_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL")

class EmbeddingManager:
    _instance = None
    _german_model = None
    _english_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize_models(self, german_model_name, english_model_name):
        if self._german_model is None:
            self._german_model = load_embedding_model(model_name=german_model_name)
        if self._english_model is None:
            self._english_model = load_embedding_model(model_name=english_model_name)

    @property
    def german_model(self):
        return self._german_model

    @property
    def english_model(self):
        return self._english_model

def load_embedding_model(
    model_name = EMBEDDING_MODEL_NAME, 
):
    """
    Loads an embedding model from either Azure OpenAI or Hugging Face repository.

    Parameters:
    - model_name: The name of the model to load. Use "openai" for Azure OpenAI embeddings
                 or a Hugging Face model name (defaults to EMBEDDING_MODEL_NAME).

    Returns:
    - An instance of AzureOpenAIEmbeddings or HuggingFaceBgeEmbeddings

    Raises:
    - ValueError: If an unsupported device is specified.
    - OSError: If the model cannot be loaded.
    """

    try:
        if model_name == "azure_openai":
            embedding_model = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
            )
            # aoaimodel=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
            # print(f"Se ha cargado el embedding model de Azure OpenAI: {aoaimodel}")
        else:
            model_name = EMBEDDING_MODEL_NAME 
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            model_kwargs = {"device": device}
            encode_kwargs = {"normalize_embeddings": True}  # For cosine similarity computation

            embedding_model = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        return embedding_model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise