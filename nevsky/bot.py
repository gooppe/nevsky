import os

import logging
import telebot
import torch
from nevsky.models import TransformerModel
from youtokentome import BPE

logger = logging.getLogger(__name__)


def run_bot(model: str):
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

    @bot.message_handler(func=lambda message: True)
    def echo_all(message):
        logger.info(f"New message from {message.chat.username}")

        encoded_source = source_bpe.encode(message.text, bos=True, eos=True)[:gen_limit]
        source = torch.LongTensor([encoded_source])
        prediction = model.generate(source, gen_limit).tolist()
        translation = target_bpe.decode(prediction, ignore_ids=(0, 1, 2))[0]

        bot.reply_to(message, translation)

    logger.info(f"Start pooling messages")
    bot.polling()
