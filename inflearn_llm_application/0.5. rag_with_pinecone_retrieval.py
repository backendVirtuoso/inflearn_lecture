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


'''
# 1. 패키지 설치

# 2. Knowledge Base 구성을 위한 데이터 생성
- [3.4 LangChain을 활용한 Vector Database 변경 (Chroma ➡️ Pinecone)](https://github.com/jasonkang14/inflearn-rag-notebook/blob/main/3.4%20LangChain%EC%9D%84%20%ED%99%9C%EC%9A%A9%ED%95%9C%20Vector%20Database%20%EB%B3%80%EA%B2%BD%20(Chroma%20%E2%9E%A1%EF%B8%8F%20Pinecone).ipynb)와 동일함

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

'''