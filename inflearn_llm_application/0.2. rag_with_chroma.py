from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200,
)

loader = Docx2txtLoader('./tax.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large')
#embedding = UpstageEmbeddings(model="solar-embedding-1-large")

database = Chroma.from_documents(documents=document_list, embedding=embedding, collection_name='chroma-tax', persist_directory="./chroma")
# database = Chroma(collection_name='chroma-tax', persist_directory="./chroma", embedding_function=embedding)

query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'
retrieved_docs = database.similarity_search(query, k=3)

llm = ChatOpenAI(model='gpt-4o')
#llm = ChatUpstage()

prompt = hub.pull("rlm/rag-prompt")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = database.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

ai_message = qa_chain({"query": query})
ai_message = qa_chain.invoke({"query": query})
print(ai_message)

'''
# 패키지 설치
%pip install python-dotenv langchain langchain-openai langchain-community langchain-text-splitters 
%pip install --upgrade --quiet docx2txt langchain-community
%pip install -qU langchain-text-splitters
%pip install langchain-chroma
%pip install -U langchain langchainhub --quiet
%pip install python-dotenv langchain langchain-upstage langchain-community langchain-text-splitters 
'''