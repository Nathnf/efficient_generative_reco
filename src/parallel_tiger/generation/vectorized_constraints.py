import os
import re
import torch
import logging
from typing import List, Optional, Tuple, Any, Set

logger = logging.getLogger(__name__)


def parse_item(item_str: str) -> List[str]:
    """
    Parse a single item string into tokens.
    
    Args:
        item_str: A string containing tokens enclosed in angle brackets.
                 E.g., "<a_1><b_3><c_9><d_56>"
    
    Returns:
        A list of token strings extracted from the input.
        E.g., ["<a_1>", "<b_3>", "<c_9>", "<d_56>"]
    """
    return re.findall(r"<[^<>]+>", item_str)


def compute_or_load_transition_constraints_codebook_fast(
    cfg: Any,
    tokenizer: Any,
    all_items: Set[str],
    first_token_constraints_path: Optional[str] = None,
    transition_constraints_t1_path: Optional[str] = None,
    transition_constraints_t2_path: Optional[str] = None,
    prefix_to_uidx_t3_path: Optional[str] = None,
    uidx_to_next_tokens_t3_path: Optional[str] = None,
    num_special_tokenizer_tokens: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute or load codebook-aware transition constraints for sequence generation.

    This function is SPECIFICALLY DESIGNED FOR `n_query = 4`**, meaning it assumes
    each item code is represented by exactly four discrete tokens (e.g. "<a_12><b_87><c_5><d_201>").

    This function builds transition constraints that enforce valid token sequences
    based on observed item patterns. It creates constraints for different sequence
    positions (t=0, t=1, t=2, t=3) to guide generation toward valid item sequences.

    Implementation Detail
    ---------------------
    --- 1. Indices
    In our setup, the transformer vocabulary is structured as:

        <pad>, <eos>, <unk>, <extra_id_0>,
        <a_0> ... <a_255>,   # codebook 0 (1st token position)
        <b_0> ... <b_255>,   # codebook 1 (2nd token position)
        <c_0> ... <c_255>,   # codebook 2 (3rd token position)
        <d_0> ... <d_255>    # codebook 3 (4th token position)

    The `<a_*>` tokens always appear in position 0, `<b_*>` in position 1, etc.
    Therefore, for the 0→1 transition, we don't need to consider `<c_*>` or `<d_*>`
    tokens because their presence there is invalid by design.  
    To save memory, we map global vocabulary indices to codebook-local indices
    in `[0, codebook_size)`, by subtracting the number of special tokens and taking
    modulo `codebook_size`.  
    This lets us store transition constraints with shape `(codebook_size, ...)`
    instead of `(vocab_size, ...)`, reducing memory by a factor of `n_query` (e.g.,
    4x smaller for `n_query=4`).

    --- 2. Why not a full boolean mask for t=3?
    In principle, for t=3 we could use:

        transition_mask_t3: (codebook_size, codebook_size, codebook_size, codebook_size)
    where `transition_mask_t3[tok1, tok2, tok3]` is a boolean vector over possible
    `tok4` candidates.  
    However, with `codebook_size = 256` and `n_query = 4`, this would require:

        256^4 ≈ 4.3 trillion booleans
        → torch uses 1 byte per boolean, which would create a 256^4 / 1024^3 = 4GB object

    --- 3. Memory Efficient Approach?
    Instead, we store only valid 3-token prefixes observed in the dataset:

        - `prefix_to_uidx`: maps (tok1, tok2, tok3) → unique integer ID
        - `uidx_to_next_tokens`: boolean mask of shape (|U|, codebook_size),
          where |U| is the number of unique valid prefixes.

    If |U| is much smaller than `codebook_size^3`, memory usage is manageable.
    (Scalability on very large datasets remains to be tested.)

    ---

    Args:
        cfg: The config object containing n_query (sequence length) and code_num 
               (codebook size) attributes.
        tokenizer: Tokenizer object with a get_vocab() method that returns a 
                  dictionary mapping tokens to indices.
        all_items: List of item strings, where each string contains tokens 
                  enclosed in angle brackets (e.g., "<a_1><b_3><c_9><d_56>").
        first_token_constraints_path: Optional path to save/load first token 
                                    constraints tensor.
        transition_constraints_t1_path: Optional path to save/load t=1 transition 
                                      constraints tensor.
        transition_constraints_t2_path: Optional path to save/load t=2 transition 
                                      constraints tensor.
        prefix_to_uidx_t3_path: Optional path to save/load prefix-to-unique-index 
                               mapping for t=3 transitions.
        uidx_to_next_tokens_t3_path: Optional path to save/load unique-index-to-next-tokens 
                                   mapping for t=3 transitions.
        num_special_tokenizer_tokens: Number of special tokens at the beginning of 
                                    the vocabulary (default: 4).

    Returns:
        A tuple containing:
        - first_token_constraints: (codebook_size,) boolean tensor indicating 
          valid first tokens.
        - transition_mask_t1: (codebook_size, codebook_size) boolean tensor 
          for t=0→t=1 transitions.
        - transition_mask_t2: (codebook_size, codebook_size, codebook_size) 
          boolean tensor for t=0→t=1→t=2 transitions.
        - prefix_to_uidx: (codebook_size, codebook_size, codebook_size) int32 
          tensor mapping 3-token prefixes to unique indices.
        - uidx_to_next_tokens: (|U|, codebook_size) boolean tensor mapping 
          unique prefix indices to valid next tokens.
          
    Raises:
        AssertionError: If cfg.n_query is different from 4. 
        AssertionError: If any item doesn't have the expected number of tokens 
                       (cfg.n_query).
        KeyError: If a token from all_items is not found in the tokenizer vocabulary.
    """
    codebook_size = cfg.code_num  # Number of tokens per codebook
    n_query = cfg.n_query
    assert n_query==4, "n_query != 4 not supported for now"

    if (
        first_token_constraints_path
        and os.path.exists(first_token_constraints_path)
        and transition_constraints_t1_path
        and os.path.exists(transition_constraints_t1_path)
        and transition_constraints_t2_path
        and os.path.exists(transition_constraints_t2_path)
        and prefix_to_uidx_t3_path
        and os.path.exists(prefix_to_uidx_t3_path)
        and uidx_to_next_tokens_t3_path
        and os.path.exists(uidx_to_next_tokens_t3_path)
    ):
        logger.info("Loading precomputed constraints...")
        first_token_constraints = torch.load(first_token_constraints_path)
        transition_mask_t1 = torch.load(transition_constraints_t1_path)
        transition_mask_t2 = torch.load(transition_constraints_t2_path)
        prefix_to_uidx_t3 = torch.load(prefix_to_uidx_t3_path)
        uidx_to_next_tokens_t3 = torch.load(uidx_to_next_tokens_t3_path)

        logger.info("Total unique 3-token prefixes (|U|): {}".format(uidx_to_next_tokens_t3.shape[0]))
        total_size_bytes = (
            prefix_to_uidx_t3.numel() * prefix_to_uidx_t3.element_size()
            + uidx_to_next_tokens_t3.numel() * uidx_to_next_tokens_t3.element_size()
        )
        logger.info(
            "Estimated memory usage (t=3, both objects): {:.2f} MB".format(total_size_bytes / (1024 * 1024))
        )

        return (
            first_token_constraints,
            transition_mask_t1,
            transition_mask_t2,
            prefix_to_uidx_t3,
            uidx_to_next_tokens_t3,
        )

    # Initialize masks t0, t1 and t2 masks
    first_token_constraints = torch.zeros(codebook_size, dtype=torch.bool)
    transition_mask_t1 = torch.zeros(codebook_size, codebook_size, dtype=torch.bool)
    transition_mask_t2 = torch.zeros(
        codebook_size, codebook_size, codebook_size, dtype=torch.bool
    )

    # Initialize objects for t3 transitions
    ## Mapping from 3-token prefix to unique index
    prefix_to_uidx_dict = dict()
    uidx_to_next_tokens = []
    ## For building prefix to index tensor (dense)
    prefix_to_uidx = -torch.ones(
        (codebook_size, codebook_size, codebook_size), dtype=torch.int32
    )
    uidx_counter = 0

    # Build token to codebook index mapping (excluding special tokens)
    tokens_to_indices = tokenizer.get_vocab()

    for item_str in all_items:
        item_tokens = parse_item(
            item_str
        )
        assert (
            len(item_tokens) == n_query
        ), f"Item {item_str} has {len(item_tokens)} tokens, expected {n_query}."

        # Convert token IDs to codebook-local indices
        idx0 = (
            tokens_to_indices[item_tokens[0]] - num_special_tokenizer_tokens
        )  % codebook_size # % codebook_size shouldn't be needed here but we keep it for safety
        first_token_constraints[idx0] = True

        idx1 = (
            tokens_to_indices[item_tokens[1]] - num_special_tokenizer_tokens
        ) % codebook_size
        transition_mask_t1[idx0, idx1] = True

        idx2 = (
            tokens_to_indices[item_tokens[2]] - num_special_tokenizer_tokens
        ) % codebook_size
        transition_mask_t2[idx0, idx1, idx2] = True

        # t=3 logic
        idx3 = (
            tokens_to_indices[item_tokens[3]] - num_special_tokenizer_tokens
        ) % codebook_size

        prefix = (idx0, idx1, idx2)

        if prefix not in prefix_to_uidx_dict:
            prefix_to_uidx_dict[prefix] = uidx_counter
            prefix_to_uidx[idx0, idx1, idx2] = uidx_counter
            uidx_to_next_tokens.append(torch.zeros(codebook_size, dtype=torch.bool))
            uidx_counter += 1

        uidx = prefix_to_uidx_dict[prefix]
        uidx_to_next_tokens[uidx][idx3] = True

    uidx_to_next_tokens = torch.stack(uidx_to_next_tokens, dim=0)  # (|U|, code_num)

    logger.info("Total unique 3-token prefixes (|U|): {}".format(uidx_to_next_tokens.shape[0]))
    total_size_bytes = (
        prefix_to_uidx.numel() * prefix_to_uidx.element_size()
        + uidx_to_next_tokens.numel() * uidx_to_next_tokens.element_size()
    )
    logger.info(
        "Estimated memory usage (t=3, both objects): {:.2f} MB".format(total_size_bytes / (1024 * 1024))
    )

    # Save if path provided
    if first_token_constraints_path:
        torch.save(first_token_constraints, first_token_constraints_path)
    if transition_constraints_t1_path:
        torch.save(transition_mask_t1, transition_constraints_t1_path)
    if transition_constraints_t2_path:
        torch.save(transition_mask_t2, transition_constraints_t2_path)
    if prefix_to_uidx_t3_path:
        torch.save(prefix_to_uidx, prefix_to_uidx_t3_path)
    if uidx_to_next_tokens_t3_path:
        torch.save(uidx_to_next_tokens, uidx_to_next_tokens_t3_path)

    return (
        first_token_constraints,
        transition_mask_t1,
        transition_mask_t2,
        prefix_to_uidx,
        uidx_to_next_tokens,
    )
