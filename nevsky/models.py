from typing import Collection, Tuple, Union

import torch
from torch import nn

from nevsky.modules import Transformer

EmbeddingDescription = Union[
    str, Tuple[str, str], Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]],
]


class TransformerModel(nn.Module):
    """Transformer module.

    Args:
        embeddings (EmbeddingDescription): Four ways to initialize embeddings:
            - from shared pretrained embedding file
            - from two different pretrained embedding files
            - from shared embedding sizes
            - from two different pretrained embedding sizes
        hidden_size (int): hidden size. Defaults to 512.
        inner_size (int): inner size. Defaults to 2048.
        num_layers (int): number of layers. Defaults to 6.
        num_heads (int): number of attention heads. Defaults to 8.
        max_seq_len (int, optional): maximum sequence lenght. Defaults to 300.
        dropout (float, optional): dropout probability. Defaults to 0.1.

    Inputs:
        source (torch.LongTensor): source tensor of shape `batch_size, source_len`.
        target (torch.LongTensor): target tensor of shape `batch_size, target_len`.

    Returns:
        torch.Tensor: result tensor of shape `batch_size, target_len, vocab_size`.
    """

    def __init__(
        self,
        embeddings: EmbeddingDescription,
        hidden_size: int = 512,
        inner_size: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 300,
        dropout: float = 0.1,
    ):

        super().__init__()
        enc_embs, dec_embs = self.get_embeddings(embeddings)
        hidden_size_check = (
            enc_embs.embedding_dim == dec_embs.embedding_dim == hidden_size
        )
        assert hidden_size_check, "Hidden sizes must be compatible."

        emb_shapes = (enc_embs.weight.shape, dec_embs.weight.shape)
        params = (hidden_size, inner_size, num_layers, num_heads, max_seq_len)

        self.transformer = Transformer(enc_embs, dec_embs, *params)

        self.register_buffer("embedding_shapes", torch.LongTensor(emb_shapes))
        self.register_buffer("model_params", torch.LongTensor(params))

    def forward(
        self, source: torch.LongTensor, target: torch.LongTensor,
    ) -> torch.Tensor:
        return self.transformer(source, target)

    def generate(
        self,
        source: torch.LongTensor,
        limit: int,
        bos_token: int = 1,
        eos_token: int = 2,
    ) -> torch.LongTensor:
        """Generate predicted sequence.

        Args:
            source (torch.LongTensor): source tensor of shape `batch_size, seq_len`.
            limit (int): generation limit.
            bos_token (int): begin of sequence token. Defaults to 1.
            eos_token (int): end of sequence token. Defaults to 2.

        Returns:
            torch.LongTensor: generated tensor of shape `batch_size, gen_len`.
        """
        training = self.training
        self.eval()
        prediction = self.transformer.generate(source, limit, bos_token, eos_token)
        self.train(training)

        return prediction

    def criterion(
        self, prediction: torch.FloatTensor, target: torch.LongTensor
    ) -> torch.FloatTensor:
        """Transformer encoder decoder loss function.

        Args:
            prediction (torch.FloatTensor): tensor of shape
                `batch_size, seq_len, vocab_size`.
            target (torch.LongTensor): tensor of shape `batch_size, seq_len`.

        Returns:
            torch.FloatTensor: loss value.
        """
        return self.transformer.criterion(prediction, target)

    @classmethod
    def get_embeddings(
        cls, emb: EmbeddingDescription,
    ) -> Tuple[nn.Embedding, nn.Embedding]:
        """Load or instantiate embedding modules.

        Args:
            emb (EmbeddingDescription): Four ways to initialize embeddings:
                - from shared pretrained embedding file
                - from two different pretrained embedding files
                - from shared embedding sizes
                - from two different pretrained embedding sizes

        Returns:
            Tuple[nn.Embedding, nn.Embedding]: encoder and decoder embedding modules.
        """
        if isinstance(emb, str):
            emb_tensor = torch.load(emb, map_location="cpu")
            embeddings = nn.Embedding.from_pretrained(emb_tensor, padding_idx=0)
            return embeddings, embeddings

        elif isinstance(emb, Collection):
            if isinstance(emb[0], str) and isinstance(emb[1], str):
                source_emb = cls.get_embeddings(emb[0])[0]
                target_emb = cls.get_embeddings(emb[1])[0]
                return source_emb, target_emb
            elif isinstance(emb[0], int) and isinstance(emb[1], int):
                embeddings = nn.Embedding(*emb, padding_idx=0)
                return embeddings, embeddings
            elif isinstance(emb[0], Collection) and isinstance(emb[1], Collection):
                source_emb = cls.get_embeddings(emb[0])[0]
                target_emb = cls.get_embeddings(emb[1])[1]
                return source_emb, target_emb

        raise ValueError("Please, check supported embedding instantiation arguments")

    def save(self, dump_filename: str):
        """Save model.

        Args:
            dump_filename (str): file name of model dump.
        """
        state_dict = self.state_dict()
        torch.save(state_dict, dump_filename)

    @classmethod
    def load(cls, dump_filename: str) -> "TransformerModel":
        """Load model from dump.

        Args:
            dump_filename (str): file name of model dump.

        Returns:
            TransformerModel: model.
        """
        state_dict = torch.load(dump_filename, map_location="cpu")
        params = state_dict["model_params"].tolist()
        emb_sizes = state_dict["embedding_shapes"].tolist()

        model = TransformerModel(emb_sizes, *params)
        model.load_state_dict(state_dict)

        return model
