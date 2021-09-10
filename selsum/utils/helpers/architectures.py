DEFAULT_MAX_TARGET_POSITIONS = 100

def bart_encoder_architecture(args, prefix=""):
    """Produces the standard bart architecture."""
    prefix = f'{prefix}_' if prefix != "" else prefix

    setattr(args, f"{prefix}encoder_embed_path",
            getattr(args, f"{prefix}encoder_embed_path", None))
    setattr(args, f"{prefix}encoder_embed_dim",
            getattr(args, f"{prefix}encoder_embed_dim", 768))
    setattr(args, f"{prefix}encoder_ffn_embed_dim",
            getattr(args, f"{prefix}encoder_ffn_embed_dim", 4 * 768))
    setattr(args, f"{prefix}encoder_layers",
            getattr(args, f"{prefix}encoder_layers", 6))

    setattr(args, f"{prefix}encoder_attention_heads",
            getattr(args, f"{prefix}encoder_attention_heads", 12))
    setattr(args, f"{prefix}encoder_normalize_before",
            getattr(args, f"{prefix}encoder_normalize_before", False))
    setattr(args, f"{prefix}encoder_learned_pos",
            getattr(args, f"{prefix}encoder_learned_pos", True))
    setattr(args, f"{prefix}attention_dropout",
            getattr(args, f"{prefix}attention_dropout", 0.0))
    setattr(args, f"{prefix}activation_dropout",
            getattr(args, f"{prefix}activation_dropout", 0.0))
    setattr(args, f"{prefix}activation_fn",
            getattr(args, f"{prefix}activation_fn", "relu"))
    setattr(args, f"{prefix}dropout",
            getattr(args, f"{prefix}dropout", 0.1))
    setattr(args, f"{prefix}no_scale_embedding",
            getattr(args, f"{prefix}no_scale_embedding", True))
    setattr(args, f"{prefix}layernorm_embedding",
            getattr(args, f"{prefix}layernorm_embedding", True))
    setattr(args, f"{prefix}adaptive_input",
            getattr(args, f"{prefix}adaptive_input", False))
    setattr(args, f"{prefix}no_token_positional_embeddings",
            getattr(args,f"{prefix}no_token_positional_embeddings", False))
    setattr(args, f"{prefix}no_token_positional_embeddings",
            getattr(args, f"{prefix}no_token_positional_embeddings", False))

    # adding missing arguments to make it compliant with the standard
    # Transformer encoder (default values)
    setattr(args, f"{prefix}quant_noise_pq",
            getattr(args, f"{prefix}quant_noise_pq", 0))
    setattr(args, f'{prefix}quant_noise_pq_block_size',
            getattr(args, f"{prefix}quant_noise_pq_block_size", 8))
    setattr(args, f'{prefix}quant_noise_scalar',
            getattr(args, f'{prefix}quant_noise_scalar', 0))
    setattr(args, f'{prefix}encoder_layerdrop',
            getattr(args, f'{prefix}encoder_layerdrop', 0))


def posterior_architecture(args):
    """Architecture of the inference network q(r_{1:K}| r_{1:N}, s)."""
    # encoder
    args.q_encoder_hidden_dim = getattr(args, 'q_encoder_hidden_dim', 150)
    args.q_encoder_dropout = getattr(args, 'q_encoder_dropout', 0.1)
    args.q_nlayers = getattr(args, 'q_encoder_nlayers', 2)
    # decoder
    args.q_decoder_embed_dim = args.q_encoder_hidden_dim
    args.q_decoder_learned_pos = getattr(args, 'q_decoder_learned_pos', True)
    args.q_max_positions = getattr(args, 'q_max_positions', 100)
    return args


# def prior_architecture(args):
#     """The architecture of the prior network."""
#     # default BART encoder
#     bart_encoder_architecture(args, 'p')
#     args.p_freeze_base_encoder = getattr(args, 'p_freeze_base_encoder', True)
#     args.p_encoder_hidden_dim = getattr(args, 'p_encoder_hidden_dim', 100)
#     # decoder
#     args.p_decoder_embed_dim = getattr(args, 'p_encoder_embed_dim')
#     args.p_decoder_learned_pos = getattr(args, 'p_decoder_learned_pos', True)
#     args.p_decoder_max_positions = getattr(args, 'p_decoder_max_positions', 100)


def contextualizer_architecture(args):
    """contextualizer module used in the mode predictor."""
    bart_encoder_architecture(args, 'contxt')
    args.contxt_encoder_embed_dim = getattr(args, 'contxt_encoder_embed_dim')
    args.contxt_dropout = getattr(args, 'contxt_dropout', 0.1)
    args.contxt_attention_dropout = getattr(args, 'contxt_attention_dropout', 0.1)
    args.contxt_max_source_positions = getattr(args, 'contxt_max_source_positions', DEFAULT_MAX_TARGET_POSITIONS)
    args.contxt_learned_pos = True
    args.contxt_encoder_ffn_embed_dim = getattr(args, 'contxt_encoder_ffn_embed_dim', 2 * 768)
    args.contxt_encoder_layers = getattr(args, 'contxt_encoder_layers', 2)
    args.contxt_encoder_attention_heads = getattr(args, 'contxt_encoder_attention_heads', 8)
    args.contxt_encoder_normalize_before = False
    args.contxt_encoder_normalize_output = True
