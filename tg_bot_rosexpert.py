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

loader = WebBaseLoader(
    ["https://rosexperts.ru/", "https://rosexperts.ru/poleznoe/voprosy-i-otvety/"])  # загрузка данных с сайта
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


@bot.message_handler(commands=['start'])  # указываем, какие команды отслеживаем
def start(message):
    bot.send_message(message.chat.id, "Выберите нужный блок в меню")


@bot.message_handler(commands=['question'])  # указываем, какие команды отслеживаем
def question(message):
    bot.send_message(message.chat.id, "Напишите вопрос в строке ввода")


@bot.message_handler(commands=['aboutcompany'])
def description(message):
    bot.send_message(message.chat.id, "Работая с 2012 года, команда центра сертификации «Росэксперт» подготовила "
                                      "более 28 000 документов для компаний из России, Беларуси, Казахстана и "
                                      "других стран\n\nКаждый день мы решаем нестандартные задачи, гарантируя "
                                      "качественное выполнение заказов любого объема. Показательный кейс – "
                                      "сертификация российского издания книги «Гарри Поттер и философский камень»")


@bot.message_handler(commands=['contacts'])
def contacts(message):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = telebot.types.KeyboardButton('WhatsApp')
    item2 = telebot.types.KeyboardButton('Телефон')
    item3 = telebot.types.KeyboardButton('Почта')
    item4 = telebot.types.KeyboardButton('Адрес')

    markup.add(item1, item2, item3, item4)
    bot.send_message(message.chat.id, 'Выберите удобный способ связи из предложенных вариантов', reply_markup=markup)


@bot.message_handler(content_types=['text'])  # отслеживаем только ввод текста
def get_text_messages(message):
    if message.text == 'WhatsApp':
        answer = 'Ссылка на наш WhatsApp \nhttps://wtsp.cc/79673550951'
        bot.send_message(message.chat.id, answer)  # ответ бота

    elif message.text == 'Телефон':
        answer = 'Наш номер телефона \n8 800 775-27-45'
        bot.send_message(message.chat.id, answer)  # ответ бота

    elif message.text == 'Почта':
        answer = 'Наша почта \ninfo@rosexperts.ru'
        bot.send_message(message.chat.id, answer)  # ответ бота

    elif message.text == 'Адрес':
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = telebot.types.KeyboardButton('Москва')
        item2 = telebot.types.KeyboardButton('Калининград')
        back = telebot.types.KeyboardButton('Назад')

        markup.add(item1, item2, back)
        bot.send_message(message.chat.id, 'Выберите город', reply_markup=markup)

    elif message.text == 'Москва':
        answer = 'Наш адрес: \nг. Москва, ул. Расплетина, д. 12, корп. 1 (почтовый индекс 123060)\n' \
                 'График работы: понедельник-пятница 09:00-18:00'
        bot.send_message(message.chat.id, answer)

    elif message.text == 'Калининград':
        answer = 'Наш адрес: \nг. Калининград, ул. Зоологическая, д. 50 (почтовый индекс 236022)\n' \
                 'График работы: понедельник-пятница 09:00-18:00'
        bot.send_message(message.chat.id, answer)

    elif message.text == 'Назад':
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = telebot.types.KeyboardButton('WhatsApp')
        item2 = telebot.types.KeyboardButton('Телефон')
        item3 = telebot.types.KeyboardButton('Почта')
        item4 = telebot.types.KeyboardButton('Адрес')

        markup.add(item1, item2, item3, item4)
        bot.send_message(message.chat.id, 'Выберите удобный способ связи из предложенных вариантов',
                         reply_markup=markup)

    else:
        msg = bot.send_message(message.chat.id, '*Ищем ответ на ваш вопрос. Скоро вернемся*', parse_mode='Markdown')
        answer = qa(message.text)
        answer = answer['answer'] + '\n\nЕсли у вас есть другой вопрос, пожалуйста, напишите его'
        bot.edit_message_text(chat_id=message.chat.id, message_id=msg.message_id, text=answer)


# чтобы бот не останавливался
bot.polling(none_stop=True)
