from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
app = Flask(__name__)
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HF_API_KEY"] = HF_API_KEY
embeddings = download_hugging_face_embeddings()
index_name = "medibot"
# Load Existing Index
# Embed each chunk and upsert the embeddings into the Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(index_name = index_name, embedding = embeddings)
retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs = {'k': 3})
hf_llm = HuggingFaceEndpoint(
    model="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HF_API_KEY,
    task="conversational",
    temperature=0.4,
    max_new_tokens=500
)

llm = ChatHuggingFace(llm=hf_llm)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
@app.route("/")
def index():
    return render_template("chat.html")
@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    print("Response: ", response["answer"])
    return str(response["answer"])
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080, debug = True)