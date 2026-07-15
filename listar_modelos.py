import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Modelos disponíveis pra geração de texto:")
for m in client.models.list():
    if "generateContent" in m.supported_actions:
        print(f"  {m.name}")

print("\nModelos disponíveis pra embedding:")
for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(f"  {m.name}")