from flask import Flask, render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
import pinecone

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

embeddings = download_hugging_face_embeddings()

pc = pinecone.Pinecone(PINECONE_API_KEY)
index_name = 'llamedibot'

docsearch = Pinecone.from_existing_index(index_name=index_name,embedding=embeddings)

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context','question'])

chain_type_kwargs = {'prompt':prompt}

llm = CTransformers(model=r'C:\Users\amish\.cache\huggingface\hub\models--TheBloke--Llama-2-7B-Chat-GGML\snapshots\76cd63c351ae389e1d4b91cab2cf470aab11864b',
                    model_type='llama',
                    config={'max_new_tokens':512,
                            'temperature':0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route('/get',methods = ['GET','POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    result=qa({'query':input})
    print('Response :',result['result'])
    return str(result['result'])

if __name__ == '__main__':
    app.run(debug=True)