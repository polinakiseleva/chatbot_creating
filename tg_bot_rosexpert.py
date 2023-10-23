from langchain.memory import ConversationSummaryMemory
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

import os
import openai
import telebot
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
bot = telebot.TeleBot(os.getenv("TG_KEY"))

chat = ChatOpenAI()

loader = WebBaseLoader("https://rosexperts.ru/")  # загрузка данных с сайта
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


# answer = qa(input('Введите ваш вопрос '))
# print(answer['answer'])

@bot.message_handler(commands=['start'])  # указываем, какие команды отслеживаем
def start(message):
    bot.send_message(message.chat.id, "Привет! Чем могу помочь?")


@bot.message_handler(content_types=['text'])  # отслеживаем только ввод текста
def get_text_messages(message):

    answer = qa(message.text)
    # """
    #
    # :param message:
    # :return:
    # """
    # reply = ''
    # response = openai.ChatCompletion.create(
    #     messages=[
    #         {"role": "user", "content": message.text}
    #     ],  # какой текст будет принимать бот в качестве запроса
    #     model="gpt-4",  # модель чата
    #     max_tokens=300,  # максимальное количество возвращаемых слов
    #     temperature=1,  # креативность
    #     n=1,  # количество ответов
    #     stop=None  # стоп-слово, которое стопарит чат
    # )
    # if response and response.choices:  # если есть ответ и его варианты
    #     reply = response.choices[0].message.content
    # else:
    #     reply = 'Что-то пошло не по плану!'

    bot.send_message(message.chat.id, answer['answer'])  # ответ бота


# чтобы бот не останавливался
bot.polling(none_stop=True)
