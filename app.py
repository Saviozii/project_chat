from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from ChatBot.Rag import embedding_pergunta, busca_qdrant, resposta_guida

app = FastAPI()

class Mensagem(BaseModel):
    usuario_msg: str
    guida_msg: str


class PerguntaRequest(BaseModel):
    pergunta: str
    historico: List[Mensagem] = []


class PerguntaResponse(BaseModel):
    resposta: str
    historico: List[Mensagem]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/perg", response_model=PerguntaResponse)
def perguntar(body: PerguntaRequest):
    historico = [msg.model_dump() for msg in body.historico]
    pergunta_vec = embedding_pergunta(body.pergunta)
    contexto = busca_qdrant(pergunta_vec)
    resposta = resposta_guida(contexto, body.pergunta, historico)
    return PerguntaResponse(
        resposta=resposta,
        historico=[Mensagem(**msg) for msg in historico]
    )

app.mount("/guida", StaticFiles(directory="static", html=True), name="static")