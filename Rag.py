#Rag para busca no qdrant.
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import Ollama
from langchain_ollama.embeddings import OllamaEmbeddings
import os
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
load_dotenv()

PROMPT_TEMPLATE = """
Você é Guida, assistente especializada no Guia de Acesso e Permanência nas Universidades.
Responda sempre em português, de forma clara e objetiva.
Regras obrigatórias:
- Responda como uma pessoa real conversando, não como um relatório formal.
- Use linguagem simples e acolhedora, como se estivesse ajudando um amigo.
- Responda somente com base no CONTEXTO DO DOCUMENTO ou no HISTÓRICO DA CONVERSA.
- Se o usuário mencionou algo sobre si mesmo no histórico (como nome, curso,
  situação), use essa informação para personalizar a resposta.
- Se a informação não estiver nem no contexto nem no histórico, diga exatamente:
  "Não encontrei essa informação no documento."
- NUNCA complemente uma resposta com suposições, sugestões ou informações
  que não estejam literalmente no CONTEXTO ou HISTÓRICO abaixo.
- Após responder a pergunta com os dados do contexto, PARE. Não adicione
  frases como "posso ajudá-lo com mais informações" ou "se precisar de
  mais detalhes". Responda apenas o que foi perguntado.
- Quando possível, cite o número da página entre colchetes, ex: [Página 12].
- Se a resposta envolver uma lista de itens, use marcadores (•).
- PROIBIDO inventar qualquer dado como nomes, e-mails, telefones,
  redes sociais, endereços ou valores numéricos. Se não estiver
  literalmente no CONTEXTO, diga que não encontrou.
- Se a pergunta for uma saudação ou conversa informal, responda
  normalmente e de forma simpática.
- NUNCA comece a resposta com "Infelizmente" ou frases negativas. Se tiver
  a informação, vá direto ao ponto.
- NUNCA agrupe informações em "Parte 1", "Parte 2" ou estruturas que não
  estejam literalmente no documento. Cite apenas o que está escrito.
- NUNCA invente intervalos de páginas como "Páginas 10-23". Cite apenas
  páginas que aparecem literalmente no contexto recebido.

CONTEXTO DO DOCUMENTO:
{contexto}

PERGUNTA ATUAL:
{pergunta}

RESPOSTA:
  
  """

Ebook_pdf = os.getenv("EBOOK_PDF")

Ollama_Url = "http://localhost:11434"
embedding_model = "bge-m3"
Ollama_model= "gemma4"

Qdrant_Url= os.getenv("QDRANT_URL")
api_key_qdrant= os.getenv("QDRANT_API_KEY")
COLLETION = "Guia_Ebook"

model = OllamaEmbeddings(
  model= embedding_model,
  base_url= Ollama_Url, 
)

client_qdrant= QdrantClient(
  url = Qdrant_Url,
  api_key = api_key_qdrant,
)

def embedding_pergunta(a):
  pergunta = model.embed_query(a)
  return pergunta 


def busca_qdrant(pergunta,top_k=5):
  
  busca = client_qdrant.query_points(
    collection_name=COLLETION,
    query=pergunta,
    limit=top_k 
  )
  contexto = "\n".join([p.payload["texto"] for p in busca.points])
  return contexto


prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
#So transformei o prompt em funcao
def guida_prompt(contexto,pergunta):
  
  return prompt_template.format(
    contexto=contexto,
    pergunta=pergunta,
  )

guida_llm = OllamaLLM(
  model= Ollama_model,
  base_url= Ollama_Url,
)

def resposta_guida(sequisabe,sequisabess):
  prompt = prompt_template.format(
    contexto=sequisabe,
    pergunta=sequisabess,
  )
  resposta = guida_llm.invoke(prompt)
  return resposta


duvida = str(input("Qual sua duvida? "))
pergunta = embedding_pergunta(duvida)
contexto = busca_qdrant(pergunta)
guida_prompt(contexto,duvida)
resposta = resposta_guida(contexto,duvida)
print(resposta)