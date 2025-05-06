from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import bs4


from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()


app = Flask(__name__)

DOC_URL = "https://docs.docker.com/get-started/docker-overview/"
GROQ_API_KEY = "Your_Groq_api"



#Data Ingestion
loader = WebBaseLoader(
    web_paths=(DOC_URL,),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("flex w-full gap-8")))
)
docs = loader.load()

# Transforma the Data into chuncks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(docs)

# convert the Text into Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(documents, embeddings)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retriever=vector_db.as_retriever()



# LLM
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
llm = init_chat_model("llama3-8b-8192", model_provider="groq")


# Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
<context>
{context}
</context>
Question: {input}"""
)
parser = StrOutputParser()


# Create Chain == prompt | llm | OutputPraser
document_chain = create_stuff_documents_chain(llm, prompt_template, output_parser=parser)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# UI route handles GET (empty form) & POST (process question)
@app.route('/', methods=['GET', 'POST'])
def home():
    answer = None
    if request.method == 'POST':
        question = request.form.get('prompt', '').strip()
        if question:
            res = retrieval_chain.invoke({'input': question})
            answer = res.get('answer') or res.get('output')
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(host=os.getenv('FLASK_HOST', '0.0.0.0'), port=int(os.getenv('FLASK_PORT', 5000)))
