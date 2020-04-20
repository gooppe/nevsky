import click
import os


@click.group()
def main():
    pass


@click.command()
@click.argument(
    "dump", type=click.Path(exists=True, file_okay=False),
)
@click.argument("sentence", type=click.STRING)
def translate(dump, sentence: str):
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
def bot():
    from nevsky import bot

    bot.run_bot()

main.add_command(translate)

main.add_command(bot)
