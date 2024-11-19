import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

# Load environment variables
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

# Existing configuration variables
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
AZURE_OPENAI_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL")

# LangSmith configuration
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

def load_llm_client():
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AAZURE_OPENAI_API_LLM_DEPLOYMENT_ID")
    )
    return wrap_openai(client)

@traceable
def azure_openai_call(prompt):
    client = load_llm_client()
    
    response = client.chat.completions.create(
        model=AZURE_OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
            {"role": "user", "content": str(prompt)}
        ]
    )
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    result = azure_openai_call("Hello, world!")
    print(result)