# Guida - Chatbot RAG para Universidades

Chatbot inteligente baseado em **RAG (Retrieval-Augmented Generation)** para auxiliar estudantes com informações sobre o Guia de Acesso e Permanência nas Universidades.

## Funcionalidades

- Busca semântica em documentos usando embeddings
- Memória de conversa (histórico das últimas 6 mensagens)
- Interface web intuitiva
- Respostas em português com citação de páginas
- API REST com FastAPI

## Tecnologias

| Componente | Tecnologia |
|------------|------------|
| LLM | Ollama (gemma4) |
| Embeddings | Ollama (bge-m3) |
| Banco Vetorial | Qdrant |
| Backend | FastAPI |
| Frontend | HTML/CSS/JS |

## Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.ai/) instalado e rodando
- Qdrant Cloud (ou local) configurado

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/Saviozii/project_chat.git
cd project_chat

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Edite o arquivo .env com suas credenciais
```

## Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
QDRANT_URL=sua_url_do_qdrant
QDRANT_API_KEY=sua_chave_api
```

## Executar

```bash
# Iniciar o servidor
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Acessar a interface
# http://localhost:8000/guida
```

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Verificação de saúde |
| POST | `/perg` | Enviar pergunta ao chatbot |
| GET | `/guida` | Interface web |

### Exemplo de uso da API

```bash
curl -X POST "http://localhost:8000/perg" \
  -H "Content-Type: application/json" \
  -d '{
    "pergunta": "Como me matricular?",
    "historico": []
  }'
```

## Estrutura do Projeto

```
project_chat/
├── app.py              # Aplicação FastAPI
├── ChatBot/
│   ├── Rag.py          # Lógica RAG principal
│   ├── pipeline.py     # Pipeline de processamento
│   └── pipeline_inf.py # Pipeline de inferência
├── static/
│   └── index.html      # Interface web
├── data/               # Documentos para indexação
├── requirements.txt    # Dependências Python
└── .env                # Variáveis de ambiente
```

## Licença

MIT
