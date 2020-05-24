import logging
import os
import random
from typing import List

import telebot
from nevsky.dictionary import select_translation, take_random_words
from nevsky.models import TranslatorInferenceModel
from nevsky.ocr import recognize_binary_image

logger = logging.getLogger(__name__)


def format_words(word: str, translations: List[str]) -> str:
    """Format dictionary-translated words response.
    Args:
        word (str): Original word.
        translations (List[str]): word translations.
    Returns:
        str: formated response.
    """

    if len(translations) == 0:
        return f"Для слова {word} не найден словарный перевод"

    message = [f"Варианты перевода слова {word}:"]
    for i, t in enumerate(translations, 1):
        message.append(f"{i}: {t}")
    return "\n".join(message)


def run_bot(model: str):
    """Run bot.
    Args:
        model (str): name of translation model.
    """
    token = os.environ["TELEGRAM_TOKEN"]
    bot = telebot.TeleBot(token)
    translator = TranslatorInferenceModel(model)

    test_state = {}

    @bot.message_handler(commands=["start", "help"])
    def help(m):
        logger.info(f"New help message {m.chat.username}")

        answer = (
            "Используйте команду /translate или /t чтобы перевести текст, "
            "а команду /dict или /d чтобы найти слово в словаре"
        )
        bot.send_message(m.chat.id, answer)

    @bot.message_handler(commands=["translate", "t"])
    def translate(m):
        logger.info(f"New translate message from {m.chat.username}")

        text = " ".join(m.text.split(" ")[1:])
        if len(text.strip()) == 0:
            bot.send_message(m.chat.id, "Введите текст после команды")
            return

        bot.send_chat_action(m.chat.id, "typing")
        translation = translator.translate(text)

        if len(translation) == 0:
            logger.error(f"Translation error for '{text}': translation is empty.")
            translation = "Ошибка перевода"

        bot.send_message(m.chat.id, translation)

    @bot.message_handler(commands=["dict", "d"])
    def dictionary(m):
        logger.info(f"New dict message from {m.chat.username}")

        words = m.text.lower().split(" ")[1:]
        if len(words) == 0:
            bot.send_message(m.chat.id, "Введите слово после команды")

        for word in words:
            translations = select_translation(word)
            bot.send_message(m.chat.id, format_words(word, translations))

    @bot.message_handler(content_types=["photo"])
    def photo(m):
        logger.info(f"Got photo recognition request {m.chat.username}")
        bot.send_chat_action(m.chat.id, "typing")

        fileID = m.photo[-1].file_id
        file_info = bot.get_file(fileID)
        raw_img = bot.download_file(file_info.file_path)
        text = recognize_binary_image(raw_img).strip()

        if len(text) == 0:
            bot.send_message(m.chat.id, "На изображении не найдено текста")
            return

        translation = translator.translate(text)

        if len(translation) == 0:
            logger.error(f"Translation error for '{text}': translation is empty.")
            translation = "Ошибка перевода"

        bot.send_message(m.chat.id, translation)

    @bot.message_handler(commands=["quiz", "q"])
    def run_test(m):
        logger.info(f"New test request from {m.chat.username}")

        params = m.text.split(" ")[1:]
        if len(params) < 1:
            n_words = 10
        else:
            n_words = max(min(int(params[0]), 30), 4)

        test_state[m.chat.username] = {
            "words": take_random_words(n_words),
            "state": 0,
            "score": 0,
        }
        bot.send_message(
            m.chat.id, "Тест запущен! Напишите перевод для предлагаемых сообщений."
        )
        send_word(m)

    def send_word(m):
        state = test_state[m.chat.username]["state"]
        words = test_state[m.chat.username]["words"]
        word = words[state]

        candidate_words = random.sample(words[:state] + words[state + 1 :], 3)
        candidate_words = random.sample(candidate_words + [word], 4)
        markup = telebot.types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
        buttons = [telebot.types.KeyboardButton(w[0]) for w in candidate_words]
        markup.add(*buttons)
        bot.send_message(
            m.chat.id, f"Выберите перевод слова {word[1]}", reply_markup=markup
        )

    @bot.message_handler(func=lambda message: not message.text.startswith("/"))
    def check_answer(m):
        username = m.chat.username
        if username not in test_state:
            return

        state = test_state[username]["state"]
        words = test_state[username]["words"]
        n_words = len(words)
        word = words[state][0]

        test_state[username]["state"] += 1

        answer = m.text.lower().strip()
        if answer == word:
            test_state[username]["score"] += 1
            response = f"Верно!"
        else:
            response = f"Неверно, правильный перевод: {word}"

        bot.send_message(m.chat.id, response)

        if state + 1 == n_words:
            score = test_state[username]["score"]
            response = f"Вы перевели {score} из {n_words} слов\n"
            response += f"Ваша оценка: {score / n_words * 100}%"
            bot.send_message(m.chat.id, response)
            del test_state[username]
        else:
            send_word(m)

    logger.info(f"Start pooling messages")
    bot.polling()