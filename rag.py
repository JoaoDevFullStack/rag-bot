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


# ============ Funções auxiliares ============

def gerar_embedding(texto):
    for tentativa in range(3):
        try:
            resultado = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texto,
            )
            return resultado.embeddings[0].values
        except errors.ServerError:
            espera = 2 ** tentativa
            print(f"  [503, esperando {espera}s...]")
            time.sleep(espera)
    raise Exception("Falha ao gerar embedding")


def similaridade_cosseno(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def chamar_llm_com_retry(prompt, modelos=("gemini-2.5-flash", "gemini-2.0-flash")):
    """Chama o LLM com fallback de modelos e retry."""
    for modelo in modelos:
        for tentativa in range(3):
            try:
                resposta = client.models.generate_content(
                    model=modelo, contents=prompt
                )
                return resposta.text
            except errors.ServerError:
                espera = 2 ** tentativa
                print(f"  [503 em {modelo}, esperando {espera}s...]")
                time.sleep(espera)
    raise Exception("Todos os modelos falharam")


# ============ Indexação ============

def indexar_pdf(caminho_pdf):
    """Lê PDF, quebra em chunks, gera embeddings, retorna base indexada."""
    print(f"Indexando {caminho_pdf}...")
    reader = PdfReader(caminho_pdf)
    texto = ""
    for n, pag in enumerate(reader.pages, start=1):
        texto += f"\n\n[Página {n}]\n{pag.extract_text()}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(texto)

    base = []
    for i, chunk in enumerate(chunks):
        emb = gerar_embedding(chunk)
        base.append({"texto": chunk, "embedding": emb})
        print(f"  Chunk {i+1}/{len(chunks)}")

    print(f"✅ {len(base)} chunks indexados\n")
    return base


# ============ Busca ============

def buscar(base, pergunta, top_k=3):
    emb_pergunta = gerar_embedding(pergunta)
    resultados = []
    for chunk_data in base:
        score = similaridade_cosseno(emb_pergunta, chunk_data["embedding"])
        resultados.append({"texto": chunk_data["texto"], "score": score})
    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:top_k]


# ============ O coração do RAG: montar o prompt e gerar resposta ============

def responder_com_rag(base, pergunta):
    print(f"\n{'='*60}")
    print(f"❓ Pergunta: {pergunta}")
    print(f"{'='*60}")

    # 1. Busca os chunks relevantes
    print("\n🔍 Buscando chunks relevantes...")
    chunks_relevantes = buscar(base, pergunta, top_k=3)
    for i, c in enumerate(chunks_relevantes, start=1):
        print(f"  Top {i}: score {c['score']:.3f}")

    # 2. Monta o contexto pro LLM
    contexto = "\n\n---\n\n".join(
        [f"Trecho {i+1}:\n{c['texto']}" for i, c in enumerate(chunks_relevantes)]
    )

    # 3. Monta o prompt completo
    prompt = f"""Você é um assistente especializado em responder perguntas com base em documentos fornecidos.

REGRAS IMPORTANTES:
- Use APENAS as informações dos trechos abaixo para responder
- Se a resposta não estiver nos trechos, responda exatamente: "Não encontrei essa informação no documento."
- Cite o número da página entre colchetes ao final de cada afirmação (ex: "...conforme descrito [Página 2]")
- Responda de forma clara, objetiva e em português

TRECHOS DO DOCUMENTO:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA:"""

    # 4. Chama o LLM
    print("\n🤖 Gemini gerando resposta...")
    resposta = chamar_llm_com_retry(prompt)

    print(f"\n💬 Resposta:\n{resposta}\n")
    return resposta


# ============ Programa principal ============

if __name__ == "__main__":
    base = indexar_pdf("documentos/n8n_docs.pdf")

    perguntas = [
        "Como adiciono um trigger node?",
        "O que é um workflow no n8n?",
        "Como agendar a execução automática toda segunda às 9h?",
        "Qual a receita do bolo de chocolate?",  # pegadinha — não está no PDF
    ]

    for pergunta in perguntas:
        responder_com_rag(base, pergunta)