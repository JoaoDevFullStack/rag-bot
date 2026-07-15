import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# ============ Setup ============
load_dotenv()
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def gerar_embedding(texto):
    """Converte um texto em vetor numérico usando o Gemini (com retry em 503)."""
    for tentativa in range(3):
        try:
            resultado = client_gemini.models.embed_content(
                model="gemini-embedding-001",
                contents=texto,
            )
            return resultado.embeddings[0].values
        except errors.ServerError:
            espera = 2 ** tentativa
            print(f"  [503 ao gerar embedding, esperando {espera}s...]")
            time.sleep(espera)
    raise Exception("Falha ao gerar embedding após 3 tentativas")


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Adapta o gerar_embedding() do Gemini pro formato que o ChromaDB espera.
    Isso permite que o Chroma gere embeddings automaticamente tanto
    ao indexar documentos (add) quanto ao buscar (query) — sem você
    precisar chamar gerar_embedding() manualmente em nenhum dos dois casos.
    """

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [gerar_embedding(texto) for texto in input]


# ============ Setup do ChromaDB (persistente em disco) ============

CAMINHO_DB = "./chroma_db"
NOME_COLECAO = "n8n_docs"

chroma_client = chromadb.PersistentClient(path=CAMINHO_DB)

# "hnsw:space": "cosine" é importante — o padrão do Chroma é distância
# euclidiana (L2), mas embeddings do Gemini foram pensados pra similaridade
# de cosseno (a mesma métrica que você já usava no NumPy). Sem isso, os
# resultados do ranking podem sair diferentes do que você validou antes.
colecao = chroma_client.get_or_create_collection(
    name=NOME_COLECAO,
    embedding_function=GeminiEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"},
)

print(f"Coleção '{NOME_COLECAO}' já tem {colecao.count()} chunks indexados em '{CAMINHO_DB}'.")


# ============ Indexação (só roda se a coleção estiver vazia) ============
# Essa é a diferença central da Etapa 5: nas etapas anteriores, todo Python
# que você rodava reindexava o PDF do zero (chamando gerar_embedding pra
# cada chunk, de novo). Agora, uma vez indexado, o Chroma persiste em disco
# e as próximas execuções pulam direto pra busca.

if colecao.count() == 0:
    print("\nColeção vazia. Indexando PDF do zero...")

    caminho_pdf = "documentos/n8n_docs.pdf"
    reader = PdfReader(caminho_pdf)
    texto_completo = ""
    for numero_pagina, pagina in enumerate(reader.pages, start=1):
        texto_completo += f"\n\n[Página {numero_pagina}]\n{pagina.extract_text()}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(texto_completo)
    print(f"Gerados {len(chunks)} chunks.")

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print("Gerando embeddings e salvando no ChromaDB (isso leva ~20-30s)...")
    # Aqui NÃO chamamos gerar_embedding manualmente — o Chroma chama
    # GeminiEmbeddingFunction por trás dos panos, chunk por chunk.
    colecao.add(
        documents=chunks,
        ids=ids,
    )
    print(f"✅ {len(chunks)} chunks indexados e persistidos em '{CAMINHO_DB}'.")
else:
    print("Coleção já indexada — pulando reprocessamento do PDF.")


# ============ Busca ============

def buscar(pergunta, top_k=3):
    """Busca os top_k chunks mais relevantes pra pergunta, usando o ChromaDB."""
    resultados = colecao.query(
        query_texts=[pergunta],
        n_results=top_k,
    )
    saida = []
    for texto, distancia in zip(resultados["documents"][0], resultados["distances"][0]):
        # Com hnsw:space=cosine, "distância" é (1 - similaridade do cosseno).
        # Ou seja: quanto MENOR, mais parecido (é o inverso do score que
        # você usava antes com similaridade_cosseno).
        saida.append({"texto": texto, "distancia": distancia})
    return saida


# ============ Teste ============

if __name__ == "__main__":
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
            print(f"\n--- Resultado {i} (distância: {r['distancia']:.3f}, menor = mais parecido) ---")
            print(r["texto"][:300] + ("..." if len(r["texto"]) > 300 else ""))