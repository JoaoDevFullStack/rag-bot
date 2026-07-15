import os
import time
import shutil
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from google import genai
from google.genai import errors
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# ============ Setup ============
load_dotenv()
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PASTA_DOCUMENTOS = "documentos"
CAMINHO_DB = "./chroma_db"
NOME_COLECAO = "documentos"  # coleção genérica — não mais um PDF fixo
MODELO_GERACAO = "gemini-2.5-flash"  # troque aqui se usar outro modelo de texto

os.makedirs(PASTA_DOCUMENTOS, exist_ok=True)


# ============ Embeddings (igual etapas anteriores) ============

def gerar_embedding(texto):
    """Converte um texto em vetor numérico usando o Gemini (com retry em 503 e 429)."""
    for tentativa in range(5):
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
        except errors.ClientError as e:
            if e.code == 429:
                # O free tier do Gemini limita requisições/minuto. Em vez de
                # adivinhar o tempo certo, esperamos um pouco mais que os
                # ~40s que o próprio erro costuma recomendar.
                espera = 40
                print(f"  [429 quota excedida, esperando {espera}s...]")
                time.sleep(espera)
            else:
                raise
    raise Exception("Falha ao gerar embedding após 5 tentativas")


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Adapta o gerar_embedding() do Gemini pro formato que o ChromaDB espera."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for i, texto in enumerate(input):
            embeddings.append(gerar_embedding(texto))
            # Pausa pequena entre chamadas — o free tier do Gemini permite
            # ~100 requisições/minuto de embedding. Isso evita estourar
            # a quota em PDFs grandes (muitos chunks de uma vez).
            if i < len(input) - 1:
                time.sleep(0.7)
        return embeddings


# ============ ChromaDB (persistente, multi-documento) ============

chroma_client = chromadb.PersistentClient(path=CAMINHO_DB)

colecao = chroma_client.get_or_create_collection(
    name=NOME_COLECAO,
    embedding_function=GeminiEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"},
)


# ============ Indexação ============

def indexar_pdf(caminho_pdf: str, nome_documento: str) -> int:
    """
    Lê, quebra em chunks e indexa um PDF no ChromaDB, marcando cada chunk
    com metadata {"documento": nome_documento} — é isso que permite
    depois filtrar a busca por documento específico.

    Usa upsert (não add): se o mesmo nome_documento for reenviado, os
    chunks antigos são substituídos em vez de duplicados.
    """
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

    # ids prefixados com o nome do documento pra não colidir com chunks
    # de outros PDFs já indexados na mesma coleção
    ids = [f"{nome_documento}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"documento": nome_documento} for _ in chunks]

    colecao.upsert(
        documents=chunks,
        ids=ids,
        metadatas=metadatas,
    )

    return len(chunks)


# ============ Busca + Geração (pipeline RAG completo) ============

def buscar(pergunta: str, documento: Optional[str] = None, top_k: int = 3):
    """Busca os top_k chunks mais relevantes, opcionalmente filtrando por documento."""
    where_filter = {"documento": documento} if documento else None

    resultados = colecao.query(
        query_texts=[pergunta],
        n_results=top_k,
        where=where_filter,
    )

    chunks_encontrados = []
    documentos = resultados.get("documents", [[]])[0]
    metadatas = resultados.get("metadatas", [[]])[0]
    distancias = resultados.get("distances", [[]])[0]

    for texto, meta, distancia in zip(documentos, metadatas, distancias):
        chunks_encontrados.append({
            "texto": texto,
            "documento": meta.get("documento", "desconhecido"),
            "distancia": distancia,
        })
    return chunks_encontrados


def gerar_resposta_llm(pergunta: str, chunks: list) -> str:
    """
    Monta o contexto com os chunks recuperados e pede pro LLM responder
    SÓ com base nesse contexto — é a parte de 'Generation' do RAG que
    ainda faltava (antes o pipeline só ia até a busca).
    """
    if not chunks:
        return "Não encontrei nenhum trecho relevante nos documentos indexados pra responder essa pergunta."

    contexto = "\n\n---\n\n".join(
        f"[Fonte: {c['documento']}]\n{c['texto']}" for c in chunks
    )

    prompt = f"""Responda a pergunta do usuário usando APENAS as informações do contexto abaixo.
Se o contexto não tiver a resposta, diga claramente que não sabe — não invente informação.
Cite de qual documento veio a informação quando fizer sentido.

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA:"""

    for tentativa in range(5):
        try:
            resultado = client_gemini.models.generate_content(
                model=MODELO_GERACAO,
                contents=prompt,
            )
            return resultado.text
        except errors.ServerError:
            espera = 2 ** tentativa
            print(f"  [503 ao gerar resposta, esperando {espera}s...]")
            time.sleep(espera)
        except errors.ClientError as e:
            if e.code == 429:
                espera = 15
                print(f"  [429 quota excedida ao gerar resposta, esperando {espera}s...]")
                time.sleep(espera)
            else:
                raise
    raise Exception("Falha ao gerar resposta após 5 tentativas")


# ============ API ============

app = FastAPI(title="RAG Bot API", version="1.0")

# Libera o frontend (rodando em outra porta, ex: localhost:5173) a chamar
# essa API. Em produção, troque "*" pela URL real do frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    pergunta: str
    documento: Optional[str] = None  # se None, busca em todos os documentos
    top_k: int = 3


class QueryResponse(BaseModel):
    resposta: str
    fontes: list


@app.post("/documentos/upload")
async def upload_documento(
    arquivo: UploadFile = File(...),
    nome_documento: Optional[str] = Form(None),
):
    """
    Recebe um PDF via multipart/form-data e indexa no ChromaDB.
    Se nome_documento não for informado, usa o nome do arquivo (sem extensão).
    """
    if not arquivo.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Só arquivos .pdf são aceitos.")

    nome = nome_documento or os.path.splitext(arquivo.filename)[0]
    caminho_destino = os.path.join(PASTA_DOCUMENTOS, f"{nome}.pdf")

    with open(caminho_destino, "wb") as f:
        shutil.copyfileobj(arquivo.file, f)

    try:
        total_chunks = indexar_pdf(caminho_destino, nome)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao indexar PDF: {e}")

    return {
        "documento": nome,
        "chunks_indexados": total_chunks,
        "mensagem": f"'{nome}' indexado com sucesso.",
    }


@app.get("/documentos")
async def listar_documentos():
    """Lista os documentos já indexados e quantos chunks cada um tem."""
    todos = colecao.get(include=["metadatas"])
    contagem = {}
    for meta in todos["metadatas"]:
        nome = meta.get("documento", "desconhecido")
        contagem[nome] = contagem.get(nome, 0) + 1

    return {
        "total_documentos": len(contagem),
        "documentos": [
            {"nome": nome, "chunks": qtd} for nome, qtd in contagem.items()
        ],
    }


@app.get("/documentos/{nome}/download")
async def baixar_documento(nome: str):
    """Retorna o arquivo PDF original que foi indexado com esse nome."""
    caminho_pdf = os.path.join(PASTA_DOCUMENTOS, f"{nome}.pdf")
    if not os.path.exists(caminho_pdf):
        raise HTTPException(status_code=404, detail=f"Arquivo de '{nome}' não encontrado em disco.")

    return FileResponse(
        caminho_pdf,
        media_type="application/pdf",
        filename=f"{nome}.pdf",
    )


class RenomearRequest(BaseModel):
    novo_nome: str


@app.delete("/documentos/{nome}")
async def remover_documento(nome: str):
    """Remove todos os chunks de um documento do ChromaDB e o PDF salvo em disco."""
    existentes = colecao.get(where={"documento": nome})
    if not existentes["ids"]:
        raise HTTPException(status_code=404, detail=f"Documento '{nome}' não encontrado.")

    colecao.delete(where={"documento": nome})

    caminho_pdf = os.path.join(PASTA_DOCUMENTOS, f"{nome}.pdf")
    if os.path.exists(caminho_pdf):
        os.remove(caminho_pdf)

    return {"mensagem": f"'{nome}' removido — {len(existentes['ids'])} chunks apagados."}


@app.patch("/documentos/{nome}")
async def renomear_documento(nome: str, request: RenomearRequest):
    """
    Renomeia um documento já indexado. Atualiza a metadata "documento" de
    todos os chunks (os IDs internos continuam com o nome antigo — isso é
    só um detalhe interno, não afeta a busca nem a exibição).
    """
    novo_nome = request.novo_nome.strip()
    if not novo_nome:
        raise HTTPException(status_code=400, detail="novo_nome não pode ser vazio.")

    existentes = colecao.get(where={"documento": nome})
    if not existentes["ids"]:
        raise HTTPException(status_code=404, detail=f"Documento '{nome}' não encontrado.")

    conflito = colecao.get(where={"documento": novo_nome})
    if conflito["ids"]:
        raise HTTPException(status_code=409, detail=f"Já existe um documento chamado '{novo_nome}'.")

    novas_metadatas = [{"documento": novo_nome} for _ in existentes["ids"]]
    colecao.update(ids=existentes["ids"], metadatas=novas_metadatas)

    caminho_antigo = os.path.join(PASTA_DOCUMENTOS, f"{nome}.pdf")
    caminho_novo = os.path.join(PASTA_DOCUMENTOS, f"{novo_nome}.pdf")
    if os.path.exists(caminho_antigo):
        os.rename(caminho_antigo, caminho_novo)

    return {"mensagem": f"'{nome}' renomeado para '{novo_nome}'.", "nome": novo_nome}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Pergunta em linguagem natural -> busca semântica -> resposta gerada pelo LLM.
    Use "documento" pra restringir a busca a um PDF específico (o nome
    retornado em /documentos), ou deixe null pra buscar em todos.
    """
    chunks = buscar(request.pergunta, documento=request.documento, top_k=request.top_k)
    resposta = gerar_resposta_llm(request.pergunta, chunks)

    return QueryResponse(
        resposta=resposta,
        fontes=[
            {"documento": c["documento"], "distancia": round(c["distancia"], 3), "trecho": c["texto"][:200]}
            for c in chunks
        ],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)