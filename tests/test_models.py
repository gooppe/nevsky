from nevsky.models import TransformerModel

import pytest
import torch


@pytest.fixture(scope="module")
def embeddings_file(tmpdir_factory):
    vocab_size, hidden_size = 100, 32
    embeddings = torch.rand(vocab_size, hidden_size)
    fm = str(tmpdir_factory.mktemp("data").join("embeddings.pth"))
    torch.save(embeddings, fm)
    return fm


def test_TransformerModel_get_embeddings_shared_str(embeddings_file):
    enc_emb, dec_emb = TransformerModel.get_embeddings(embeddings_file)

    assert enc_emb.weight.shape == dec_emb.weight.shape == (100, 32)


def test_TransformerModel_get_embeddings_str(embeddings_file):
    enc_emb, dec_emb = TransformerModel.get_embeddings(
        (embeddings_file, embeddings_file)
    )

    assert enc_emb.weight.shape == dec_emb.weight.shape == (100, 32)


def test_TransformerModel_get_embeddings_shared_int():
    emb_size = 100, 32

    enc_emb, dec_emb = TransformerModel.get_embeddings(emb_size)

    assert enc_emb.weight.shape == dec_emb.weight.shape == emb_size


def test_TransformerModel_get_embeddings_int():
    emb_size = 100, 32

    enc_emb, dec_emb = TransformerModel.get_embeddings((emb_size, emb_size))

    assert enc_emb.weight.shape == dec_emb.weight.shape == emb_size


def test_TransformerModel_save_and_load(tmpdir):
    dump_filename = str(tmpdir.join("model.pth"))
    emb_size, hidden_size, inner_size = (100, 32), 32, 64

    model = TransformerModel(emb_size, hidden_size, inner_size)
    model.save(dump_filename)
    TransformerModel.load(dump_filename)
