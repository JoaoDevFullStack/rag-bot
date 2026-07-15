import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

modelos = ["gemini-2.5-flash", "gemini-2.0-flash"]

def chamar_com_retry(pergunta):
    for modelo in modelos:
        for tentativa in range(3):
            try:
                resposta = client.models.generate_content(
                    model=modelo,
                    contents=pergunta
                )
                print(f"[Sucesso usando {modelo} na tentativa {tentativa + 1}]")
                return resposta.text
            except errors.ServerError as e:
                espera = 2 ** tentativa
                print(f"[503 em {modelo}, tentativa {tentativa + 1}. Esperando {espera}s...]")
                time.sleep(espera)
            except errors.ClientError as e:
                print(f"[Erro de cliente em {modelo}: {e}]")
                break
        print(f"[Pulando pra próximo modelo...]")
    raise Exception("Todos os modelos falharam após retries")

resposta = chamar_com_retry("Em uma frase: o que é RAG em IA?")
print("\n--- Resposta do Gemini ---")
print(resposta)