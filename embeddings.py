import os
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============ Setup ============
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ============ Helpers ============

def gerar_embedding(texto):
    """Converte um texto em vetor numérico (768 dimensões) usando o Gemini."""
    for tentativa in range(3):
        try:
            resultado = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texto,
            )
            return resultado.embeddings[0].values
        except errors.ServerError:
            espera = 2 ** tentativa
            print(f"  [503 ao gerar embedding, esperando {espera}s...]")
            time.sleep(espera)
    raise Exception("Falha ao gerar embedding após 3 tentativas")


def similaridade_cosseno(vetor_a, vetor_b):
    """Calcula similaridade entre dois vetores. Retorna número entre -1 e 1."""
    a = np.array(vetor_a)
    b = np.array(vetor_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ============ Demo conceitual: comparar 3 frases ============

print("="*60)
print("DEMO 1: Comparando similaridade entre frases")
print("="*60)

frase_1 = "Como configurar um trigger no n8n"
frase_2 = "Como criar um nó de início de workflow"
frase_3 = "Receita de bolo de chocolate"

print(f"\nGerando embeddings...")
emb_1 = gerar_embedding(frase_1)
emb_2 = gerar_embedding(frase_2)
emb_3 = gerar_embedding(frase_3)

print(f"\nDimensões de cada embedding: {len(emb_1)}")
print(f"Primeiros 5 valores do embedding 1: {emb_1[:5]}")

sim_1_2 = similaridade_cosseno(emb_1, emb_2)
sim_1_3 = similaridade_cosseno(emb_1, emb_3)
sim_2_3 = similaridade_cosseno(emb_2, emb_3)

print(f"\nSimilaridades:")
print(f"  '{frase_1}'")
print(f"  vs '{frase_2}': {sim_1_2:.3f} (devem ser parecidas)")
print(f"  vs '{frase_3}': {sim_1_3:.3f} (devem ser distantes)")
print(f"\n  '{frase_2}'")
print(f"  vs '{frase_3}': {sim_2_3:.3f} (devem ser distantes)")


# ============ Demo real: buscar no PDF ============

print("\n" + "="*60)
print("DEMO 2: Busca semântica no PDF da n8n")
print("="*60)

# Reusa o chunking que já validamos
caminho_pdf = "documentos/n8n_docs.pdf"
print(f"\nLendo e quebrando PDF...")

reader = PdfReader(caminho_pdf)
texto_completo = ""
for numero_pagina, pagina in enumerate(reader.pages, start=1):
    texto_completo += f"\n\n[Página {numero_pagina}]\n{pagina.extract_text()}"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(texto_completo)
print(f"Gerados {len(chunks)} chunks.")

# Gera embedding pra cada chunk
print(f"\nGerando embeddings dos chunks (isso leva ~20-30 segundos)...")
chunks_com_embeddings = []
for i, chunk in enumerate(chunks):
    emb = gerar_embedding(chunk)
    chunks_com_embeddings.append({"texto": chunk, "embedding": emb})
    print(f"  Chunk {i+1}/{len(chunks)} indexado")

print(f"\n✅ Base indexada: {len(chunks_com_embeddings)} chunks prontos pra busca")

# Função de busca
def buscar(pergunta, top_k=3):
    """Busca os top_k chunks mais relevantes pra pergunta."""
    emb_pergunta = gerar_embedding(pergunta)
    resultados = []
    for chunk_data in chunks_com_embeddings:
        score = similaridade_cosseno(emb_pergunta, chunk_data["embedding"])
        resultados.append({"texto": chunk_data["texto"], "score": score})
    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:top_k]


# Teste com algumas perguntas
perguntas = [
    "Como adiciono um trigger node?",
    "O que é um workflow no n8n?",
    "Como agendar a execução automática?",
]

for pergunta in perguntas:
    print(f"\n{'='*60}")
    print(f"PERGUNTA: {pergunta}")
    print(f"{'='*60}")
    resultados = buscar(pergunta, top_k=2)
    for i, r in enumerate(resultados, start=1):
        print(f"\n--- Resultado {i} (score: {r['score']:.3f}) ---")
        print(r["texto"][:300] + ("..." if len(r["texto"]) > 300 else ""))