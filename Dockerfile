FROM python:3.12-slim

WORKDIR /app

# Instala as dependências primeiro (camada separada do código,
# assim o Docker só reinstala tudo se o requirements.txt mudar)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API
COPY rag_api.py .

# Pastas de dados — normalmente sobrescritas por volumes no docker-compose,
# mas criadas aqui pra garantir que existam mesmo se rodar sem volume
RUN mkdir -p documentos chroma_db

EXPOSE 8000

# Sem --reload aqui (isso é só pra desenvolvimento local).
# Em produção, o container reinicia via docker-compose/Render se cair.
# $PORT é injetado por plataformas como o Render — localmente cai em 8000.
CMD ["sh", "-c", "uvicorn rag_api:app --host 0.0.0.0 --port ${PORT:-8000}"]