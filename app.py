from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from ChatBot.Rag import embedding_pergunta, busca_qdrant, resposta_guida

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    try:
        pergunta_vec = embedding_pergunta(body.pergunta)
        contexto = busca_qdrant(pergunta_vec)
        resposta = resposta_guida(contexto, body.pergunta, historico)
    except ChatGoogleGenerativeAIError as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Atingi o limite de requisições do modelo gratuito por hoje. Tente novamente mais tarde, ou ative a cobrança no Google AI Studio para não ter essa limitação."
                },
            )
        return JSONResponse(
            status_code=502,
            content={"detail": "Não consegui gerar uma resposta agora. Tente novamente."},
        )
    return PerguntaResponse(
        resposta=resposta,
        historico=[Mensagem(**msg) for msg in historico]
    )

app.mount("/guida", StaticFiles(directory="static", html=True), name="static")