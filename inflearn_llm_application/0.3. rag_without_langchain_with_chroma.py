from docx import Document
import tiktoken
import chromadb
import os
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

def split_text(full_text, chunk_size):
    encoder = tiktoken.encoding_for_model("gpt-4o")
    total_encoding = encoder.encode(full_text)
    total_token_count = len(total_encoding)
    text_list = []

    for i in range(0, total_token_count, chunk_size):
        chunk = total_encoding[i : i + chunk_size]
        decoded = encoder.decode(chunk)
        text_list.append(decoded)
    
    return text_list

document = Document('./tax.docx')
print(f'document == {dir(document)}')

full_text = ''
for index, paragraph in enumerate(document.paragraphs):
    print(f'paragraph == {paragraph.text}')
    full_text += f'{paragraph.text}\n'

chunk_list = split_text(full_text, 1500)

chroma_client = chromadb.Client()
collection_name = 'tax_collection'
tax_collection = chroma_client.create_collection(collection_name)

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_embedding = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name='text-embedding-3-large')
tax_collection = chroma_client.get_or_create_collection(collection_name, embedding_function=openai_embedding)

id_list = []
for index in range(len(chunk_list)):
    id_list.append(f'{index}')

tax_collection.add(documents=chunk_list, ids=id_list)

query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'
retrieved_doc = tax_collection.query(query_texts=query, n_results=3)

print(retrieved_doc['documents'][0])

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": f"당신은 한국의 소득세 전문가 입니다. 아래 내용을 참고해서 사용자의 질문에 답변해주세요 {retrieved_doc['documents'][0]}"},
    {"role": "user", "content": query}
  ]
)

print(response.choices[0].message.content)