from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import re


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatOllama(model="deepseek-r1:1.5b")  # or "llama2", "gemma", etc.

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')

greetings = ["hi", "hello", "hey", "good morning", "good afternoon"]

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    
    if msg.lower() in greetings:
        final_answer = "Hello! I am good, how can I assist you today?"
    else:
        response = rag_chain.invoke({"input": msg})
        # Remove <think>...</think> section
        final_answer = re.sub(r"<think>.*?</think>", "", response["answer"], flags=re.DOTALL).strip()
    
    print("Final Answer:", final_answer)
    return final_answer

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)