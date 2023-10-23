import os
import openai
from langchain.memory import ConversationSummaryMemory
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI()

loader = WebBaseLoader(["https://rosexperts.ru/", "https://rosexperts.ru/o-kompanii/"])  # загрузка данных с сайта
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)  # разделение текста

# класс Chroma предназначен для создания векторного представления текста на основе вложений (embeddings)
# в данном случае используется `OpenAIEmbeddings` для создания вложений текста
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
memory = ConversationSummaryMemory(llm=chat, memory_key="chat_history",
                                   return_messages=True)  # хранение предыдущих сообщений в памяти чата

# реализация поиска наиболее подходящих ответов на основе векторных представлений текста
retriever = vectorstore.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=retriever, memory=memory)

answer = qa(input('Введите ваш вопрос '))
print(answer['answer'])
