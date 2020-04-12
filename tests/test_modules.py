import torch

from nevsky import modules


def test_generate_autoregressive_mask():
    batch_size, seq_len, hidden_size = 8, 10, 42
    tensor = torch.rand(batch_size, seq_len, hidden_size)

    mask = modules.generate_autoregressive_mask(tensor)

    assert mask.shape == (batch_size, seq_len, seq_len)


def test_ScaledDotProductAttention_forward():
    batch_size, q_len, k_v_len, hidden_size = 8, 10, 15, 42
    query = torch.rand(batch_size, q_len, hidden_size)
    key = torch.rand(batch_size, k_v_len, hidden_size)
    value = torch.rand(batch_size, k_v_len, hidden_size)
    mask = torch.zeros(batch_size, q_len, k_v_len)

    sdp_attention = modules.ScaledDotProductAttention(hidden_size)
    attention = sdp_attention(query, key, value, mask)

    assert attention.shape == (batch_size, q_len, hidden_size)


def test_MultiHeadAttention_forward():
    batch_size, q_len, k_v_len, hidden_size, num_heads = 8, 10, 15, 64, 8
    query = torch.rand(batch_size, q_len, hidden_size)
    key = torch.rand(batch_size, k_v_len, hidden_size)
    value = torch.rand(batch_size, k_v_len, hidden_size)
    mask = torch.zeros(batch_size, q_len, k_v_len)

    mh_attention = modules.MultiHeadAttention(hidden_size, num_heads)
    attention = mh_attention(query, key, value, mask)

    assert attention.shape == (batch_size, q_len, hidden_size)


def test_PositionWise_forward():
    batch_size, seq_len, hidden_size, inner_size = 8, 10, 42, 84
    tensor = torch.rand(batch_size, seq_len, hidden_size)

    positionwise = modules.PositionWise(hidden_size, inner_size)
    result = positionwise(tensor)

    assert result.shape == (batch_size, seq_len, hidden_size)


def test_TransformerEncoderLayer_forward():
    batch_size, seq_len, hidden_size, inner_size, num_heads = 8, 10, 32, 64, 8
    tensor = torch.rand(batch_size, seq_len, hidden_size)
    mask = torch.zeros(batch_size, seq_len, seq_len)

    encoder_layer = modules.TransformerEncoderLayer(hidden_size, inner_size, num_heads)
    result = encoder_layer(tensor, mask)

    assert result.shape == (batch_size, seq_len, hidden_size)


def test_TransformerEncoder_forward():
    batch_size, seq_len, hidden_size, inner_size = 8, 10, 32, 64
    num_heads, num_layers = 8, 3
    tensor = torch.rand(batch_size, seq_len, hidden_size)
    mask = torch.zeros(batch_size, seq_len, seq_len)

    encoder = modules.TransformerEncoder(hidden_size, inner_size, num_layers, num_heads)
    result = encoder(tensor, mask)

    assert result.shape == (batch_size, seq_len, hidden_size)


def test_TransformerDecoderLayer_forward():
    batch_size, hidden_size, inner_size, num_heads = 8, 32, 64, 8
    source_len, target_len = 10, 9
    source = torch.rand(batch_size, source_len, hidden_size)
    target = torch.rand(batch_size, target_len, hidden_size)
    target_mask = torch.zeros(batch_size, target_len, target_len)
    source_target_mask = torch.zeros(batch_size, source_len, target_len)

    decoder_layer = modules.TransformerDecoderLayer(hidden_size, inner_size, num_heads)
    result = decoder_layer(target, source, target_mask, source_target_mask)

    assert result.shape == (batch_size, target_len, hidden_size)


def test_TransformerDecoderLayer_forward_cached():
    batch_size, hidden_size, inner_size, num_heads = 8, 32, 64, 8
    source_len, target_len, cache_len = 10, 1, 9
    source = torch.rand(batch_size, source_len, hidden_size)
    target = torch.rand(batch_size, target_len, hidden_size)
    cache = torch.rand(batch_size, cache_len, hidden_size)
    target_mask = torch.zeros(batch_size, target_len, cache_len)
    source_target_mask = torch.zeros(batch_size, target_len, source_len)

    decoder_layer = modules.TransformerDecoderLayer(hidden_size, inner_size, num_heads)
    result = decoder_layer(
        target, source, target_mask, source_target_mask, target_cache=cache
    )

    assert result.shape == (batch_size, target_len, hidden_size)


def test_TransformerDecoder_forward():
    batch_size, hidden_size, inner_size = 8, 32, 64
    num_heads, num_layers = 8, 3
    source_len, target_len = 10, 9
    source = torch.rand(batch_size, source_len, hidden_size)
    target = torch.rand(batch_size, target_len, hidden_size)
    target_mask = torch.zeros(batch_size, target_len, target_len)
    source_target_mask = torch.zeros(batch_size, target_len, source_len)

    decoder = modules.TransformerDecoder(hidden_size, inner_size, num_layers, num_heads)
    result, cache = decoder(target, source, target_mask, source_target_mask)

    assert result.shape == (batch_size, target_len, hidden_size)
    assert cache.shape == (batch_size, num_layers, target_len, hidden_size)


def test_TransformerDecoder_forward_cached():
    batch_size, hidden_size, inner_size = 8, 32, 64
    num_heads, num_layers = 8, 3
    source_len, target_len, cache_len = 10, 1, 9
    source = torch.rand(batch_size, source_len, hidden_size)
    target = torch.rand(batch_size, target_len, hidden_size)
    cache = torch.rand(batch_size, num_layers, cache_len, hidden_size)
    target_mask = torch.zeros(batch_size, target_len, cache_len + target_len)
    source_target_mask = torch.zeros(batch_size, target_len, source_len)

    decoder = modules.TransformerDecoder(hidden_size, inner_size, num_layers, num_heads)
    result, cache = decoder(
        target, source, target_mask, source_target_mask, cache=cache
    )

    assert result.shape == (batch_size, target_len, hidden_size)
    assert cache.shape == (batch_size, num_layers, cache_len + target_len, hidden_size)


def test_PositionalEncoding_forward():
    batch_size, seq_len, hidden_size, max_len = 8, 10, 32, 100
    tensor = torch.rand(batch_size, seq_len, hidden_size)

    pe = modules.PositionalEncoding(hidden_size, max_len=max_len)
    result = pe(tensor)

    assert result.shape == tensor.shape


def test_Transformer_forward():
    batch_size, hidden_size, inner_size, vocab_size = 8, 32, 64, 100
    num_layers, num_heads = 3, 8
    max_seq_len, source_len, target_len = 100, 10, 9
    embeddings = torch.nn.Embedding(vocab_size, hidden_size)
    source = torch.randint(vocab_size, (batch_size, source_len))
    target = torch.randint(vocab_size, (batch_size, target_len))

    transformer = modules.Transformer(
        embeddings,
        embeddings,
        hidden_size,
        inner_size,
        num_layers,
        num_heads,
        max_seq_len,
    )
    result = transformer(source, target)

    assert result.shape == (batch_size, target_len, vocab_size)


def test_Transformer_generate():
    batch_size, hidden_size, inner_size, vocab_size = 8, 32, 64, 100
    num_layers, num_heads = 3, 8
    max_seq_len, source_len, limit = 100, 10, 9
    embeddings = torch.nn.Embedding(vocab_size, hidden_size)
    source = torch.randint(vocab_size, (batch_size, source_len))

    transformer = modules.Transformer(
        embeddings,
        embeddings,
        hidden_size,
        inner_size,
        num_layers,
        num_heads,
        max_seq_len,
    )
    result = transformer.generate(source, limit)

    assert result.shape == (batch_size, limit)


def test_Transformer_criterion():
    batch_size, seq_len, vocab_size = 8, 10, 100
    predicted = torch.randn(batch_size, seq_len, vocab_size)
    target = torch.randint(vocab_size, (batch_size, seq_len))

    loss = modules.Transformer.criterion(predicted, target)

    assert not torch.isnan(loss).any()
    assert loss.item is not None
