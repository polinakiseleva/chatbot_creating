import os
import openai
import telebot
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
bot = telebot.TeleBot(os.getenv("TG_KEY"))


@bot.message_handler(commands=['start'])  # указываем, какие команды отслеживаем
def start(message):
    bot.send_message(message.chat.id, "Привет! Чем могу помочь?")


@bot.message_handler(content_types=['text'])  # отслеживаем только ввод текста
def get_text_messages(message):
    """
    Функция для составления запроса и ответа на вопрос
    :param message: вопрос пользователя, введенный в чат
    """
    reply = ''
    response = openai.ChatCompletion.create(
        messages=[
            {"role": "user", "content": message.text}
        ],  # какой текст будет принимать бот в качестве запроса
        model="gpt-4",  # модель чата
        max_tokens=300,  # максимальное количество возвращаемых слов
        temperature=1,  # креативность
        n=1,  # количество ответов
        stop=None  # слово, которое стопарит чат
    )
    if response and response.choices:  # если есть ответ и его варианты
        reply = response.choices[0].message.content
    else:
        reply = 'Что-то пошло не по плану!'

    bot.send_message(message.chat.id, reply)  # ответ бота


# чтобы бот не останавливался
bot.polling(none_stop=True)
