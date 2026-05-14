from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
import os

Ebook_pdf = "data/Ebook.pdf"

model_embedding = "bge-m3"
Ollama_Url = "http://localhost:11434"
model_dimencoes = 1024

Qdrant_Url= os.getenv("QDRANT_URL")
api_key_qdrant= os.getenv("QDRANT_API_KEY")
COLLETION = "Guia_Ebook"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


#Pego o arquivo
def ler_pdf():
    loader = PyPDFLoader(Ebook_pdf)
    documentos = loader.load()
    return documentos

documentos = ler_pdf()

#Divido o arquivo
def dividir_documento(documentos):
    chunk = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap  = CHUNK_OVERLAP,
        )
    chunks = chunk.split_documents(documentos)
    return chunks

chunks = dividir_documento(documentos)

#Embedding do arquivo
model = OllamaEmbeddings(
    model = model_embedding,
    base_url = Ollama_Url,
)
def gerar_embedding(chunks):
    data = [chunk.page_content for chunk in chunks]
    embeddings = model.embed_documents(data)
    return embeddings

embeddings = gerar_embedding(chunks)

#Adcionar ao banco vetorial.

Qdrant_client = QdrantClient(url=Qdrant_Url,api_key=api_key_qdrant)

def criar_banco_qdrant():
    colecoes = [i.name for i in Qdrant_client.get_collections().collections]


    if COLLETION in colecoes:
        print("Ja tem")
    else:
        Qdrant_client.create_collection(
            collection_name= COLLETION,
            vectors_config=models.VectorParams(size= model_dimencoes,
            distance=models.Distance.COSINE),
            )


criar_banco_qdrant()

def adcionar_points(chunks, embeddings):
    pontos = []

    for i in range(len(chunks)):
        ponto = models.PointStruct(
            id= i,
            vector= embeddings[i],
            payload={
                "texto": chunks[i].page_content,
                "pagina": chunks[i].metadata.get("page", None),
                "fonte": chunks[i].metadata.get("source", None),
            },
        )
        pontos.append(ponto)
    return pontos

pontos = adcionar_points(chunks,embeddings)


def enviar_qdrant(points, batch_size=100):
    total = len(points)
    
    for i in range(0, total, batch_size):
        batch = points[i : i + batch_size]
        Qdrant_client.upsert(
            collection_name=COLLETION,
            points=batch,)
    
    print("enviado")

enviar_qdrant(pontos)
