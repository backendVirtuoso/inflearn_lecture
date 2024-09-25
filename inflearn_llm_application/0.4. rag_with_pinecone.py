from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
import os

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200,
)

loader = Docx2txtLoader('./tax.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large')
# embedding = UpstageEmbeddings(model="solar-embedding-1-large")

index_name = 'tax-index'
# index_name = 'tax-upstage-index'

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)

query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'
retrieved_docs = database.similarity_search(query, k=3)

print(retrieved_docs)

llm = ChatOpenAI(model='gpt-4o')
# llm = ChatUpstage()

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
### 3.4
# 1. 패키지 설치

# 2. Knowledge Base 구성을 위한 데이터 생성
- Chroma를 활용한 [3.2 LangChain과 Chroma를 활용한 RAG 구성](https://github.com/jasonkang14/inflearn-rag-notebook/blob/main/3.2%20LangChain%EA%B3%BC%20Chroma%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20RAG%20%EA%B5%AC%EC%84%B1.ipynb)과 동일함
- Vector Database만 [Pinecone](https://www.pinecone.io/)으로 변경

# 3. 답변 생성을 위한 Retrieval
- `RetrievalQA`에 전달하기 위해 `retriever` 생성
- `search_kwargs` 의 `k` 값을 변경해서 가져올 문서의 갯수를 지정할 수 있음
- `.invoke()` 를 호출해서 어떤 문서를 가져오는지 확인 가능

# 4. Augmentation을 위한 Prompt 활용
- Retrieval된 데이터는 LangChain에서 제공하는 프롬프트(`"rlm/rag-prompt"`) 사용

# 5. 답변 생성
- [RetrievalQA](https://docs.smith.langchain.com/old/cookbook/hub-examples/retrieval-qa-chain)를 통해 LLM에 전달
    - `RetrievalQA`는 [create_retrieval_chain](https://python.langchain.com/v0.2/docs/how_to/qa_sources/#using-create_retrieval_chain)으로 대체됨
    - 실제 ChatBot 구현 시 `create_retrieval_chain`으로 변경하는 과정을 볼 수 있음


### 3.4.1
# 1. 패키지 설치

# 2. Knowledge Base 구성을 위한 데이터 생성
- [RecursiveCharacterTextSplitter](https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/)를 활용한 데이터 chunking
    - split 된 데이터 chunk를 Large Language Model(LLM)에게 전달하면 토큰 절약 가능
    - 비용 감소와 답변 생성시간 감소의 효과
    - LangChain에서 다양한 [TextSplitter](https://python.langchain.com/v0.2/docs/how_to/#text-splitters)들을 제공
- `chunk_size` 는 split 된 chunk의 최대 크기
- `chunk_overlap`은 앞 뒤로 나뉘어진 chunk들이 얼마나 겹쳐도 되는지 지정

# 3. 답변 생성을 위한 Retrieval
- `Chroma`에 저장한 데이터를 유사도 검색(`similarity_search()`)를 활용해서 가져옴

# 4. Augmentation을 위한 Prompt 활용
- Retrieval된 데이터는 LangChain에서 제공하는 프롬프트(`"rlm/rag-prompt"`) 사용

# 5. 답변 생성
- [RetrievalQA](https://docs.smith.langchain.com/old/cookbook/hub-examples/retrieval-qa-chain)를 통해 LLM에 전달
    - `RetrievalQA`는 [create_retrieval_chain](https://python.langchain.com/v0.2/docs/how_to/qa_sources/#using-create_retrieval_chain)으로 대체됨
    - 실제 ChatBot 구현 시 `create_retrieval_chain`으로 변경하는 과정을 볼 수 있음
'''