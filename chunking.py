from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

caminho_pdf = "documentos/n8n_docs.pdf"

print(f"Lendo PDF: {caminho_pdf}")
reader = PdfReader(caminho_pdf)
texto_completo = ""
for numero_pagina, pagina in enumerate(reader.pages, start=1):
    texto_pagina = pagina.extract_text()
    texto_completo += f"\n\n[Página {numero_pagina}]\n{texto_pagina}"

print(f"PDF tem {len(reader.pages)} páginas")
print(f"Total de caracteres: {len(texto_completo)}")
print(f"Primeiros 500 caracteres do texto extraído:\n{texto_completo[:500]}")
print("\n" + "="*60 + "\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

chunks = splitter.split_text(texto_completo)

print(f"Total de chunks gerados: {len(chunks)}")
print(f"Tamanho médio dos chunks: {sum(len(c) for c in chunks) // len(chunks)} caracteres")
print("\n" + "="*60 + "\n")

for i, chunk in enumerate(chunks[:3]):
    print(f"--- Chunk {i+1} ({len(chunk)} caracteres) ---")
    print(chunk)
    print()