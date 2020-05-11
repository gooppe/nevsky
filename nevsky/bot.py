import logging
import os
from typing import List

import torch
from youtokentome import BPE

import telebot
from nevsky.dictionary import select_translation
from nevsky.models import TransformerModel

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
    dump_dir = os.path.join("dumps/", model)

    source_bpe_dump = os.path.join(dump_dir, "source_bpe.model")
    target_bpe_dump = os.path.join(dump_dir, "target_bpe.model")
    model_dump = os.path.join(dump_dir, "model.pth")

    logger.info(f"Start loading {model} dump")
    source_bpe = BPE(source_bpe_dump)
    target_bpe = BPE(target_bpe_dump)
    model = TransformerModel.load(model_dump)
    model.eval()
    gen_limit = model.transformer.max_seq_len

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
            logger.error(f"Translation error for '{text}': translation is empty.")
            translation = "Ошибка перевода"

        bot.send_message(m.chat.id, translation)
            return

        encoded_source = source_bpe.encode(text, bos=True, eos=True)[:gen_limit]
        source = torch.LongTensor([encoded_source])
        prediction = model.generate(source, gen_limit).tolist()
        translation = target_bpe.decode(prediction, ignore_ids=(0, 1, 2))[0]

        if len(translation.strip()) == 0:
            bot.send_message(m.chat.id, "Ошибка перевода")
        else:
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

    logger.info(f"Start pooling messages")
    bot.polling()
