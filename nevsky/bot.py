import os

import telebot

token = os.environ["TELEGRAM_TOKEN"]


def run_bot():
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start", "help"])
    def send_welcome(message):
        bot.reply_to(message, "Howdy, how are you doing?")

    @bot.message_handler(func=lambda message: True)
    def echo_all(message):
        bot.reply_to(message, message.text)

    bot.polling()
