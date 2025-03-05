from jsonargparse import CLI
import os
from dotenv import load_dotenv
from time import time
import re

# ANSI color codes
BLUE = "\033[34m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Environment setup
ENV_VAR_PATH = "C:/Users/hernandc/RAG Test/apikeys.env"
load_dotenv(ENV_VAR_PATH)

def load_system_prompt():
    prompt_path = os.path.join("prompts", "systemprompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo de prompt del sistema no fue encontrado en: {prompt_path}")

SYSTEM_PROMPT = load_system_prompt()

def azure_openai_call(prompt: str, messages_history: list) -> str:
    from openai import AzureOpenAI
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_API_LLM_DEPLOYMENT_ID")
    )
    
    conversation_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation_messages.extend(messages_history)
    conversation_messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_LLM_MODEL"),
        messages=conversation_messages
    )
    
    response_text = response.choices[0].message.content
    
    # Primero imprimimos la respuesta normal
    print(f"\n{BLUE}{BOLD}>> LLM-Antwort: {RESET}", end="")
    
    # Aplicamos el formato de color
    colored_response = f"{RESET}" + re.sub(
        r'\[(.*?)\]',
        lambda m: f"{GREEN}{m.group(1)}{RESET}",
        response_text
    ) + f"{RESET}"
    
    print(colored_response)
    
    # Ahora imprimimos el debug de los corchetes
    # print_bracketed_text(response_text)
       
    return response_text  # Retornamos el texto original para el historial

def print_bracketed_text(text: str):
    matches = re.findall(r'\[(.*?)\]', text)
    print(f"{GREEN}Texto entre corchetes: {RESET}", end="")
    if matches:
        for i, match in enumerate(matches, 1):
            print(f"{GREEN}{i} - {match}{RESET}")
    else:
        print(f"{ORANGE}No se encontraron textos entre corchetes{RESET}")

async def main():
    try:
        print(f"\n{BLUE}{BOLD}------------------------- Willkommen im UniChatBot - System Prompt Demo -------------------------{RESET}")
        print(f"{BLUE}{BOLD}\n>> LLM-Antwort: {RESET}Willkommen beim smart travel planner der Universität Graz! Um zu beginnen, geben Sie mir bitte Ihren Namen.")
                
        # Lista para almacenar todo el historial de mensajes
        messages_history = []

        while True:
            print("\n")
            print(f"{BLUE}{BOLD}>> Benutzer-Eingabe: {RESET}", end="")
            query = input(f"{ORANGE}{BOLD}")
            print(f"{RESET}")
            
            if query.lower() in ["exit", "cls"]:
                break
            elif query.lower() in ["reset", "rst"]:
                messages_history = []
                print(f"{BLUE}{BOLD}>> Chat-Verlauf wurde zurückgesetzt.{RESET}")
                print(f"\n{BLUE}{BOLD}----------------------------------------------------------------------------{RESET}")
                continue

            start_time = time()
            
            # Obtener respuesta del LLM incluyendo el historial
            response = azure_openai_call(query, messages_history)
            
            # Actualizar el historial con el nuevo par de mensajes
            messages_history.append({"role": "user", "content": query})
            messages_history.append({"role": "assistant", "content": response})
            
            # print(f"{BLUE}{BOLD}\n>> LLM-Antwort: {RESET}", end="")

            end_time = time()
            processing_time = end_time - start_time
            
            # print(f"\n{BLUE}{BOLD}>> Es dauerte {GREEN}{processing_time:.2f}{BLUE} Sekunden, um zu antworten.{RESET}")
            # print(f"\n{BLUE}{BOLD}----------------------------------------------------------------------------{RESET}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    CLI(main)
