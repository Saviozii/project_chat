from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os
load_dotenv()

infor_ebook = "data/info_Ebook.txt"
model_embedding = "bge-m3"
Ollama_Url = "http://localhost:11434"

Qdrant_Url= os.getenv("QDRANT_URL")
api_key_qdrant= os.getenv("QDRANT_API_KEY")


docs = []
COLLETION = "Guia_Ebook"

def ler_arquivo_txt(info_ebook):
    with open(info_ebook, "r", encoding="utf-8") as texto:
        conteudo = texto.read()

    trechos = [i.strip() for i in conteudo.split("---") if i.strip()]


    for bloco in trechos:
        page_num = "?"

        linhas = bloco.splitlines()

        for linha in linhas:
            if linha.startswith("[TRECHO") and "Página" in linha:
                page_num = int(
                    linha.split("Página")[-1]
                    .replace("]", "").strip()
                )

        resultado = Document(
            page_content=bloco,
            metadata={
                "pagina": page_num
            }
        )

        docs.append(resultado)

    return docs

ler_arquivo_txt(infor_ebook)

data = [doc.page_content for doc in docs]


model_embed = OllamaEmbeddings(
    model = model_embedding,
    base_url = Ollama_Url,
)

def embed_arquivo(data):
    embeddings = model_embed.embed_documents(data)
    return embeddings

embeddings = embed_arquivo(data)

Qdrant_client = QdrantClient(url=Qdrant_Url,
                            api_key=api_key_qdrant,)

def criar_points(docs,embeddings):
    ponto = []
    for i in range(len(docs)):
        pontos = models.PointStruct(
            id = i,
            vector = embeddings[i],
            payload = {
                "texto": docs[i].page_content,
                "pagina": docs[i].metadata["pagina"],
            }
        )
        ponto.append(pontos)
    return ponto

ponto = criar_points(docs,embeddings)

def enviar_points(ponto):
    Qdrant_client.upsert(
        collection_name=COLLETION,
        points=ponto,
    )
    print("Todos os pontos foram enviados com sucesso.")

enviar_points(ponto)
    


    