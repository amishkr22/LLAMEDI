from src.helper import load_pdf,text_chunking,download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf('data/')
text_chunks = text_chunking(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = pinecone.Pinecone(PINECONE_API_KEY)
index_name = 'llamedibot'
index = pc.Index('llamedibot')
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks],embedding=embeddings,index_name=index_name)

