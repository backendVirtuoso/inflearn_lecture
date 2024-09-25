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
- 하단의 `dictionary_chain` 과 연계하여 사용

# 6. Retrieval을 위한 keyword 사전 활용
- Knowledge Base에서 사용되는 keyword를 활용하여 사용자 질문 수정
- LangChain Expression Language (LCEL)을 활용한 Chain 연계
'''