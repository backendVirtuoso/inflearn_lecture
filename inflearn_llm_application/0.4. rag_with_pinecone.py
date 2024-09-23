from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200,
)

loader = Docx2txtLoader('./tax.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large')

index_name = 'tax-index'
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)

query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'
retrieved_docs = database.similarity_search(query, k=3)

llm = ChatOpenAI(model='gpt-4o')

prompt = hub.pull("rlm/rag-prompt")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = database.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

ai_message = qa_chain({"query": query})
ai_message = qa_chain.invoke({"query": query})
print(ai_message)
