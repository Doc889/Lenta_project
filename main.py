import os
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from creation_of_embeddings import creating_vector_store
from prompts.prompts import *

from langchain_gigachat import GigaChat
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = None
chat_history: List[HumanMessage] = []


class Question(BaseModel):
    question: str


@app.on_event("startup")
def init_ai_agent():
    global rag_chain, history_aware_retriever, question_answer_chain

    creating_vector_store()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, "db", "chroma_db_lenta")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")

    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embeddings)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    model = GigaChat(
        credentials=GIGACHAT_API_KEY,
        verify_ssl_certs=False
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


@app.post("/ask")
def ask_question(query: Question):
    print("Start chatting with the AI!")
    global rag_chain, chat_history

    if rag_chain is None:
        return {"error": "AI agent not initialized"}

    result = rag_chain.invoke({"input": query.question, "chat_history": chat_history})

    chat_history.append(HumanMessage(content=query.question))

    return {"answer": result["answer"]}
