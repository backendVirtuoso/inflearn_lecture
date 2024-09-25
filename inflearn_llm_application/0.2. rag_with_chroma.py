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

embedding = OpenAIEmbeddings(model='text-embedding-3-large')        # OpenAIEmbeddings
# embedding = UpstageEmbeddings(model="solar-embedding-1-large")    # UpstageEmbeddings

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

# 내용 설명

## 1. 패키지 설치

## 2. Knowledge Base 구성을 위한 데이터 생성
- [RecursiveCharacterTextSplitter](https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/)를 활용한 데이터 chunking
    - split 된 데이터 chunk를 Large Language Model(LLM)에게 전달하면 토큰 절약 가능
    - 비용 감소와 답변 생성시간 감소의 효과
    - LangChain에서 다양한 [TextSplitter](https://python.langchain.com/v0.2/docs/how_to/#text-splitters)들을 제공
- `chunk_size` 는 split 된 chunk의 최대 크기
- `chunk_overlap`은 앞 뒤로 나뉘어진 chunk들이 얼마나 겹쳐도 되는지 지정

## 3. 답변 생성을 위한 Retrieval
- `Chroma`에 저장한 데이터를 유사도 검색(`similarity_search()`)를 활용해서 가져옴

## 4. Augmentation을 위한 Prompt 활용
- Retrieval된 데이터는 LangChain에서 제공하는 프롬프트(`"rlm/rag-prompt"`) 사용

## 5. 답변 생성
- [RetrievalQA](https://docs.smith.langchain.com/old/cookbook/hub-examples/retrieval-qa-chain)를 통해 LLM에 전달
    - `RetrievalQA`는 [create_retrieval_chain](https://python.langchain.com/v0.2/docs/how_to/qa_sources/#using-create_retrieval_chain)으로 대체됨
    - 실제 ChatBot 구현 시 `create_retrieval_chain`으로 변경하는 과정을 볼 수 있음

'''