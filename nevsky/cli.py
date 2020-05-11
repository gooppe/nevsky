import os

import click
import nevsky.bot
import nevsky.manager
import nevsky.utils

nevsky.utils.setup_logging()


@click.group()
def main():
    """Simple translation bot. Supports full-text and dictionary translation."""
    pass


@click.command()
@click.argument(
    "dump", type=click.Path(exists=True, file_okay=False),
)
@click.argument("sentence", type=click.STRING)
def translate(dump, sentence: str):
    """Translate sentence using comand line interface."""
    import torch

    from nevsky.models import TransformerModel
    from youtokentome import BPE

    GEN_LIMIT = 50

    source_bpe_dump = os.path.join(dump, "source_bpe.model")
    target_bpe_dump = os.path.join(dump, "target_bpe.model")
    model_dump = os.path.join(dump, "model.pth")

    source_bpe = BPE(source_bpe_dump)
    target_bpe = BPE(target_bpe_dump)
    model = TransformerModel.load(model_dump)
    model.eval()

    source = torch.LongTensor([source_bpe.encode(sentence, bos=True, eos=True)])
    prediction = model.generate(source, GEN_LIMIT).tolist()
    translation = target_bpe.decode(prediction, ignore_ids=(0, 1, 2))[0]

    print(translation)


@click.command()
@click.argument("model_name", type=click.STRING)
def bot(model_name):
    """Run bot using MODEL_NAME dump file."""
    nevsky.bot.run_bot(model_name)


@click.command()
@click.argument("model_name", type=click.STRING)
def download(model_name):
    """Download pretrained MODEL_NAME dump file."""
    nevsky.manager.install_model(model_name)


main.add_command(translate)
main.add_command(bot)
main.add_command(download)
