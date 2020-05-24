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
    "model", type=click.Path(),
)
@click.argument("sentence", type=click.STRING)
def translate(model, sentence: str):
    """Translate sentence using comand line interface."""

    from nevsky.models import TranslatorInferenceModel

    translator = TranslatorInferenceModel(model)
    translation = translator.translate(sentence)

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
