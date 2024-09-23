from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage
from langchain_community.chat_models import ChatOllama

# 환경변수 불러오기 - .env 파일에 OPENAI_API_KEY 등록
load_dotenv()  

# LLM 답변 생성 
# OpenAI 대시보드에서 발급받은 API Key를 OPENAI_API_KEY라고 저장하면 별도의 설정 없이 ChatOpenAI를 사용할 수 있음
# Upstage Console에서 발급받은 API Key를 UPSTAGE_API_KEY라고 저장하면 별도의 설정 없이 ChatUpstage를 사용할 수 있음
# ChatOllama 를 활용한 LLM 답변 생성
llm = ChatOpenAI()
llm = ChatUpstage()
llm = ChatOllama(model="llama3") 

ai_message = llm.invoke("인프런에 어떤 강의가 있나요?")
print(ai_message.content)

'''
# 패키지 설치
%pip install python-dotenv langchain-openai
%pip install python-dotenv langchain-upstage
%pip install python-dotenv langchain-community
'''