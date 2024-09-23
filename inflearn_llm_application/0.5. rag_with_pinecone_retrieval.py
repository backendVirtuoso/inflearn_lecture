from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import RetrievalQA

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)

loader = Docx2txtLoader('./tax_with_markdown.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)

print(document_list[52])

load_dotenv()
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

index_name = 'tax-markdown-index'
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# 데이터를 추가할 때는 `from_documents()` 데이터를 추가한 이후에는 `from_existing_index()`를 사용합니다
# database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)
database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

query = '연봉 5천만원인 직장인의 종합소득세는?'
retriever = database.as_retriever(search_kwargs={'k': 4})
retriever.invoke(query)

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model='gpt-4o')

qa_chain = RetrievalQA.from_chain_type(
    llm, 
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

ai_message = qa_chain.invoke({"query": query})

print(ai_message)