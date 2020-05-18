import math
from typing import Tuple

import torch
from torch import nn


def generate_autoregressive_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Generate autoregressive mask for transformer model.

    Args:
        tensor (torch.Tensor): input tensor of shape `batch_size, seq_len, *`.

    Returns:
        torch.Tensor: mask tensor of shape `batch_size, seq_len, seq_len`.
    """
    batch_size, seq_len, *_ = tensor.shape
    mask = torch.ones(seq_len, seq_len, device=tensor.device).float().triu_(1)
    return mask.repeat(batch_size, 1, 1)


class ScaledDotProductAttention(nn.Module):
    """Calculate Scaled Dot Product Attention.

    Args:
        hidden_size (int): hidden size.

    Inputs:
        query (torch.Tensor): tensor of shape `batch_size, q_len, hidden_size`.
        key (torch.Tensor): tensor of shape `batch_size, k_v_len, hidden_size`.
        value (torch.Tensor): tensor of shape `batch_size, k_v_len, hidden_size`.
        mask (torch.Tensor, optional): mask tensor of shape
            `batch_size, q_len, k_v_len`. Defaults to None.

    Outputs:
        torch.Tensor: attention tensor of shape `batch_size, q_len, hidden_size`.
    """

    def __init__(
        self, hidden_size: int,
    ):
        super().__init__()

        self.scaling = 1 / math.sqrt(hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        logits = self.scaling * torch.bmm(query, key.transpose(1, 2))
        if mask is not None:
            logits.masked_fill_(mask.bool(), -float("inf"))

        attention = torch.softmax(logits, dim=-1)

        return torch.bmm(attention, value)


class MultiHeadAttention(nn.Module):
    """Scaled Dot Product Multi Head Attention.

    Args:
        hidden_size (int): hidden size. Hidden size must be divisible by
            :attr:`num_heads`.
        num_heads (int): number of attention heads. Defaults to 8.
        dropout (float): dropout probability. Defaults to 0.1.

    Inputs:
        query (torch.Tensor): tensor of shape `batch_size, q_len, hidden_size`.
        key (torch.Tensor): tensor of shape `batch_size, k_v_len, hidden_size`.
        value (torch.Tensor): tensor of shape `batch_size, k_v_len, hidden_size`.
        mask (torch.Tensor, optional): mask tensor of shape
            `batch_size, q_len, k_v_len`. Defaults to None.

    Outputs:
        torch.Tensor: result tensor of shape ``batch_size, q_len, hidden_size`.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        self.q_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False))
        self.k_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False))
        self.v_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False))

        self.attention = ScaledDotProductAttention(self.head_size)
        self.fully_connected = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape

        query_proj = self.split_heads(self.q_proj(query))
        key_proj = self.split_heads(self.k_proj(key))
        value_proj = self.split_heads(self.v_proj(value))

        if mask is not None:
            mask = (
                mask.unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .view(-1, query_len, key_len)
            )

        attention = self.attention(query_proj, key_proj, value_proj, mask)
        attention = self.join_heads(attention)
        projected = self.fully_connected(attention)
        output = self.dropout(projected)

        return self.layer_norm(query + output)

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Split tensor onto heads.

        Args:
            tensor (torch.Tensor): input tensor of shape
                `batch_size, seq_len, hidden_size`.

        Returns:
            torch.Tensor: result tensor of shape
                `batch_size * num_heads, seq_len, hidden_size / num_heads`.
        """
        batch_size, seq_len, _ = tensor.shape
        return (
            tensor.view(batch_size, seq_len, self.num_heads, self.head_size)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, seq_len, self.head_size)
        )

    def join_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Join tensor splitted onto heads.

        Args:
            tensor (torch.Tensor): input tensor of shape
                `batch_size * num_heads, seq_len, hidden_size / num_heads`.

        Returns:
            torch.Tensor: result tensor of shape `batch_size, seq_len, hidden_size`.
        """
        _, seq_len, _ = tensor.shape
        return (
            tensor.view(-1, self.num_heads, seq_len, self.head_size)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, seq_len, self.hidden_size)
        )


class PositionWise(nn.Module):
    """PositionWise transformer block.

    Args:
        hidden_size (int): hidden size.
        inner_size (int): inner size.
        dropout (float, optional): dropout probability. Defaults to 0.1.

    Inputs:
        tensor (torch.Tensor): input tensor of shape
            `batch_size, seq_len, hidden_size`.

    Outptus:
        torch.Tensor: output tensor of shape `batch_size, seq_len, hidden_size`.
    """

    def __init__(self, hidden_size: int, inner_size: int, dropout: float = 0.1):
        super().__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(tensor + self.feedforward(tensor))


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer.

    Args:
        hidden_size (int): hidden size.
        inner_size (int): inner size.
        num_heads (int): number of attention heads.
        dropout (float, optional): dropout probability. Defaults to 0.1.

    Inputs:
        input (torch.Tensor): input tensor of shape `batch_size, seq_len, hidden_size`.
        mask (torch.Tensor, optional): attention mask tensor of shape
            `batch_size, seq_len, seq_len`. Defaults to None.

    Outputs:
        torch.Tensor: result tensor of shape `batch_size, seq_len, hidden_size`.
    """

    def __init__(
        self, hidden_size: int, inner_size: int, num_heads: int, dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.positionwise = PositionWise(hidden_size, inner_size, dropout)

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attention = self.attention(input, input, input, mask)
        return self.positionwise(attention)


class TransformerEncoder(nn.Module):
    """Transformer decoder.

    Args:
        hidden_size (int): hidden size.
        inner_size (int): inner size.
        num_layers (int): number of encoder layers.
        num_heads (int): number of attention heads.
        dropout (float, optional): dropout probability. Defaults to 0.1.

    Inputs:
        input (torch.Tensor): input tensor of shape `batch_size, seq_len, hidden_size`.
        mask (torch.Tensor, optional): mask tensor of shape
            `batch_size, seq_len, seq_len`. Defaults to None.

    Outputs:
        torch.Tensor: output tensor of shape `batch_size, seq_len, hidden_size`.
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.scale_factor = 1 / math.sqrt(2 * num_layers)
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(hidden_size, inner_size, num_heads, dropout)
            for _ in range(num_layers)
        )

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            input = layer(input, mask)

        return input


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer.

    Args:
        hidden_size (int): hidden size.
        inner_size (int): inner size.
        num_heads (int): number of attention heads.
        dropout (float, optional): dropout probabilities. Defaults to 0.1.

    Inputs:
        target_input (torch.Tensor): target tensor of shape
            `batch_size, target_len, hidden_size`.
        source_input (torch.Tensor): source tensor of shape
            `batch_size, source_len, hidden_size`.
        target_mask (torch.Tensor, optional): target mask tensor of shape
            `batch_size, target_len, target_len` or `batch_size, target_len, cache_len`.
            Defaults to None.
        source_target_mask (torch.Tensor, optional): source mask tensor of shape
            `batch_size, target_len, source_len`. Defaults to None.
        target_cache (torch.Tensor, optional): cache tensor of shape
            `batch_size, cache_len, hidden_size`. Defaults to None.

    Outputs:
        torch.Tensor: result tensor of shape `batch_size, target_len, hidden_size`.
    """

    def __init__(
        self, hidden_size: int, inner_size: int, num_heads: int, dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.positionwise = PositionWise(hidden_size, inner_size, dropout)

    def forward(
        self,
        target_input: torch.Tensor,
        source_input: torch.Tensor,
        target_mask: torch.Tensor = None,
        source_target_mask: torch.Tensor = None,
        *,
        target_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        if target_cache is None:
            target_cache = target_input

        self_repr = self.self_attention(
            target_input, target_cache, target_cache, target_mask
        )

        cross_repr = self.cross_attention(
            self_repr, source_input, source_input, source_target_mask
        )

        return self.positionwise(cross_repr)


class TransformerDecoder(nn.Module):
    """Transformer decoder.

    Args:
        hidden_size (int): hidden size.
        inner_size (int): inner size.
        num_layers (int): number of decoder layers.
        num_heads (int): number of attention heads.
        dropout (float, optional): dropout probability. Defaults to 0.1.

    Inputs:
        input (torch.Tensor): input tensor of shape
            `batch_size, target_len, hidden_size`.
        encoder_out (torch.Tensor): encoder output tensor of shape
            `batch_size, source_len, hidden_size`.
        decoder_mask (torch.Tensor): decoder mask tensor of shape
            `batch_size, target_len, target_len` or
            `batch_size, target_len, cache_len + target_len`
            depends on caching.
        encoder_decoder_mask (torch.Tensor): encoder decoder mask tensor of shape
            `batch_size, target_len, source_len`.
        cache (torch.Tensor, optional): decoder cache tensor of shape
            `batch_size, num_layers, cache_len, hidden_size`. Defaults to None

    Outptus:
        Tuple[torch.Tensor, torch.Tensor]: output tensor of shape
            `batch_size, target_len, hidden_size` and cache tensor of shape
            `batch_size, num_layers, cache_len + target_len, hidden_size`.
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            TransformerDecoderLayer(hidden_size, inner_size, num_heads, dropout)
            for _ in range(num_layers)
        )

        self.scale_factor = 1 / math.sqrt(3 * num_layers)

    def forward(
        self,
        input: torch.Tensor,
        encoder_out: torch.Tensor,
        decoder_mask: torch.Tensor = None,
        encoder_decoder_mask: torch.Tensor = None,
        *,
        cache: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        _gen_cache = []
        if cache is None:
            for layer in self.layers:
                _gen_cache.append(input)
                input = layer(input, encoder_out, decoder_mask, encoder_decoder_mask)
        else:
            for i, layer in enumerate(self.layers):
                _cache = torch.cat((cache[:, i, ...], input), dim=1)
                _gen_cache.append(_cache)
                input = layer(
                    input,
                    encoder_out,
                    decoder_mask,
                    encoder_decoder_mask,
                    target_cache=_cache,
                )

        out_cache = torch.stack(_gen_cache, dim=1)
        return input, out_cache


class PositionalEncoding(nn.Module):
    """Apply positional encoding.

    Args:
        hidden_size (int): hiden size.
        dropout (float, optional): dropout probability. Defaults to 0.1.
        max_len (int, optional): max sequence lenght. Defaults to 5000.

    Inputs:
        x (torch.Tensor): input tensor of shape `batch_size, seq_len, hidden_size`.

    Outputs:
        torch.Tensor: result tensor of shape `batch_size, seq_len, hidden_size`.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(
            1e4, -torch.arange(0, hidden_size, 2).float() / hidden_size
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """Transformer module.

    Args:
        encoder_embeddings (nn.Embedding): encoder embeddings layer.
        decoder_embeddings (nn.Embedding): decoder embeddings layer.
        hidden_size (int): hidden size.
        inner_size (int): inner size.
        num_layers (int): number of layers.
        num_heads (int): number of attention heads.
        max_seq_len (int, optional): maximum sequence lenght. Defaults to 1000.
        dropout (float, optional): dropout probability. Defaults to 0.1.

    Inputs:
        source (torch.LongTensor): source tensor of shape `batch_size, source_len`.
        target (torch.LongTensor): target tensor of shape `batch_size, target_len`.

    Returns:
        torch.Tensor: result tensor of shape `batch_size, target_len, vocab_size`.
    """

    def __init__(
        self,
        encoder_embeddings: nn.Embedding,
        decoder_embeddings: nn.Embedding,
        hidden_size: int,
        inner_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_size == encoder_embeddings.embedding_dim
        assert hidden_size == decoder_embeddings.embedding_dim

        self.hidden_size = hidden_size
        self.vocab_size = decoder_embeddings.num_embeddings
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        positional_encoding = PositionalEncoding(hidden_size, dropout, max_seq_len)
        self.encoder_embeddings = nn.Sequential(encoder_embeddings, positional_encoding)
        self.decoder_embeddings = nn.Sequential(decoder_embeddings, positional_encoding)

        self.encoder = TransformerEncoder(
            hidden_size, inner_size, num_layers, num_heads, dropout
        )
        self.decoder = TransformerDecoder(
            hidden_size, inner_size, num_layers, num_heads, dropout
        )
        self.out_to_vocab = nn.Linear(hidden_size, self.vocab_size)

    def forward(
        self, source: torch.LongTensor, target: torch.LongTensor,
    ) -> torch.Tensor:
        source_emb = self.encoder_embeddings(source)
        target_emb = self.decoder_embeddings(target)

        pad_mask = torch.eq(source, 0)

        # batch_size, seq_len -> batch_size, 1, seq_len -> batch_size, seq_len, seq_len
        encoder_mask = pad_mask.unsqueeze(1).repeat(1, pad_mask.size(-1), 1)
        # batch_size, seq_len -> batch_size, 1, seq_len -> batch_size, tar_len, seq_len
        encoder_decoder_mask = pad_mask.unsqueeze(1).repeat(1, target.size(-1), 1)
        decoder_mask = generate_autoregressive_mask(target)

        encoder_repr = self.encoder(source_emb, encoder_mask)
        decoder_repr, _ = self.decoder(
            target_emb, encoder_repr, decoder_mask, encoder_decoder_mask
        )

        return self.out_to_vocab(decoder_repr)

    @staticmethod
    def criterion(
        prediction: torch.FloatTensor, target: torch.LongTensor
    ) -> torch.FloatTensor:
        """Transformer encoder decoder loss function.

        Args:
            prediction (torch.FloatTensor): tensor of shape
                `batch_size, seq_len, vocab_size`.
            target (torch.LongTensor): tensor of shape `batch_size, seq_len`.

        Returns:
            torch.FloatTensor: loss value.
        """
        seq_len = target.size(1) - 1

        prediction = prediction.narrow(1, 0, seq_len).flatten(0, 1)
        target = target.narrow(1, 1, seq_len).flatten()

        return nn.functional.cross_entropy(prediction, target, ignore_index=0)

    @torch.no_grad()
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
        beam_size = 3
        batch_size, _ = source.shape
        assert limit <= self.max_seq_len
        assert batch_size == 1

        device = source.device

        encoder_emb = self.encoder_embeddings(source)
        pad_mask = source.eq(0)
        encoder_mask = pad_mask.unsqueeze(1).repeat(1, pad_mask.size(-1), 1)
        encoder_repr = self.encoder(encoder_emb, encoder_mask)

        prediction = torch.full(
            (batch_size, 1), bos_token, dtype=torch.long, device=device
        )
        initial_tokens = prediction.repeat(beam_size, 1)
        beamsearch = BeamSearch1B(initial_tokens, beam_size)
        cache = None

        for i in range(1, limit):
            decoder_emb = self.decoder_embeddings(prediction).narrow(1, -1, 1)
            decoder_repr, cache = self.decoder(decoder_emb, encoder_repr, cache=cache)
            distribution = self.out_to_vocab(decoder_repr)[:, 0].softmax(-1)
            prediction, (encoder_repr, cache) = beamsearch.update(
                distribution, (encoder_repr, cache)
            )

            generated = torch.sum(prediction == eos_token, dim=-1, dtype=torch.bool)
            if torch.any(generated):
                return prediction[generated]

        return prediction[0:1]


class BeamSearch1B:
    def __init__(self, initial_tokens: torch.LongTensor, beam_size: int = 6):
        """One batch beams search decoding alogrithm.

        Args:
            initial_tokens (torch.LongTensor): tensor of shape `Beam, N`.
            beam_size (int): number of beams.
            trailing_token (int): eos trailing token.
        """
        # beam, n
        self.candidates = initial_tokens
        # beam, 1
        self.scores = torch.zeros(beam_size, 1, device=initial_tokens.device)
        self.beam_size = beam_size

    def update(
        self, prediction: torch.FloatTensor, cache: Tuple[torch.Tensor, ...] = None,
    ):
        """Update beam search.

        Args:
            prediction (torch.FloatTensor): tensor of shape `Beam, Vocab`.
            cache (Tuple[torch.FloatTensor, ...], optional): cache tensors of shape
                `Beam, ...`. Defaults to None.
        Returns:
            Union[torch.LongTensor, Tuple[torch.LongTensor, Tuple[torch.Tensor, ...]]]:
                reordered candidate seqeunce and cache tensor if provided.
        """
        beam_size, vocab_size = prediction.size()

        # For beam size equals to `1` perform initial scoring
        if beam_size == 1:
            rescored_prediction = torch.log(prediction)
        else:
            rescored_prediction = self.scores + torch.log(prediction)
        rescored_prediction = rescored_prediction.view(-1)

        # beam
        values, indices = torch.topk(rescored_prediction, self.beam_size)
        beam_indices = indices // vocab_size
        selected_tokens = indices % vocab_size
        # beam, n
        reordered_candidates = self.candidates[beam_indices]
        self.candidates = torch.cat(
            (reordered_candidates, selected_tokens.unsqueeze(-1)), -1
        )
        self.scores = values.unsqueeze(-1)

        if cache is not None:
            reordered_cache = tuple(t[beam_indices] for t in cache)
            return self.candidates, reordered_cache

        return self.candidates
