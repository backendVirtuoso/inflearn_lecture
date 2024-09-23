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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200,
)

loader = Docx2txtLoader('./tax_with_markdown.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)

print(document_list[52])

# 환경변수를 불러옴
load_dotenv()

# OpenAI에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

index_name = 'tax-markdown-index'
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)

query = '연봉 5천만원인 거주자의 종합소득세는?'
retriever = database.as_retriever(search_kwargs={'k': 2})
retriever.invoke(query)

llm = ChatOpenAI(model='gpt-4o')
prompt = hub.pull("rlm/rag-prompt")

retriever = database.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = retriever,
    chain_type_kwargs={"prompt": prompt}
)
#query -> 직장인 -> 거주자 chain 추가
ai_message = qa_chain.invoke({"query": query})

print(ai_message)

dictionary = ["사람을 나타내는 표현 -> 거주자"]

prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
    그런 경우에는 질문만 리턴해주세요
    사전: {dictionary}
    
    질문: {{question}}
""")

dictionary_chain = prompt | llm | StrOutputParser()
new_question = dictionary_chain.invoke({"question": query})
tax_chain = {"query": dictionary_chain} | qa_chain
ai_response = tax_chain.invoke({"question": query})

print(ai_response)

