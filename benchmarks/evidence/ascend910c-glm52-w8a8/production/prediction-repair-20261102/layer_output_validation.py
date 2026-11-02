import torch


def hidden_valid_rows(rows, sequence_lengths, global_sequence_lengths, attention_rank, attention_size):
    if not 0 <= attention_rank < attention_size:
        raise AssertionError("Invalid attention rank")
    local_rows = len(sequence_lengths)
    if rows == local_rows:
        lengths = sequence_lengths
    elif rows == sum(len(lengths) for lengths in global_sequence_lengths):
        lengths = [length for rank_lengths in global_sequence_lengths for length in rank_lengths]
    elif local_rows % attention_size == 0 and rows == local_rows // attention_size:
        lengths = sequence_lengths[attention_rank * rows:(attention_rank + 1) * rows]
    else:
        raise AssertionError(f"Unknown decoder hidden layout: rows={rows}, local={local_rows}")
    if any(length < 0 for length in lengths):
        raise AssertionError("Negative hidden-row KV sequence length")
    return [index for index, length in enumerate(lengths) if length > 0]


def check_hidden_tensor(observed, expected, sequence_lengths, global_sequence_lengths, attention_rank, attention_size):
    if observed.shape != expected.shape or observed.dtype != expected.dtype or observed.device != expected.device:
        raise AssertionError("Hidden output tensor identity differs from the serving call")
    if expected.ndim != 2 or expected.shape[1] != 6144 or not expected.is_floating_point():
        raise AssertionError("Unknown decoder hidden tensor contract")
    valid_rows = hidden_valid_rows(expected.shape[0], sequence_lengths, global_sequence_lengths,
                                   attention_rank, attention_size)
    valid = torch.tensor(valid_rows, dtype=torch.int64, device=expected.device)
    torch.testing.assert_close(observed.index_select(0, valid), expected.index_select(0, valid), rtol=0.02, atol=0.02)
    padding_rows = [index for index in range(expected.shape[0]) if index not in valid_rows]
    padding = torch.tensor(padding_rows, dtype=torch.int64, device=expected.device)
    differences = (observed.index_select(0, padding) != expected.index_select(0, padding)).sum().item()
    return {"valid_rows": len(valid_rows), "padding_rows": len(padding_rows), "padding_differences": differences}


def check_topk_outputs(observed, expected, sequence_lengths):
    if expected is None or observed is None:
        if expected is not None or observed is not None:
            raise AssertionError("Indexer output presence differs from the serving call")
        return {"valid_rows": 0, "padding_rows": 0, "padding_differences": 0}
    if observed.shape != expected.shape or observed.dtype != expected.dtype or observed.device != expected.device:
        raise AssertionError("Indexer output tensor identity differs from the serving call")
    if expected.ndim != 2 or expected.is_floating_point() or len(sequence_lengths) != expected.shape[0]:
        raise AssertionError("Unknown indexer validity contract")
    valid_rows = [index for index, length in enumerate(sequence_lengths) if length > 0]
    padding_rows = [index for index, length in enumerate(sequence_lengths) if length == 0]
    if len(valid_rows) + len(padding_rows) != len(sequence_lengths):
        raise AssertionError("Negative KV sequence length")
    valid = torch.tensor(valid_rows, dtype=torch.int64, device=expected.device)
    torch.testing.assert_close(observed.index_select(0, valid), expected.index_select(0, valid), rtol=0, atol=0)
    padding = torch.tensor(padding_rows, dtype=torch.int64, device=expected.device)
    differences = (observed.index_select(0, padding) != expected.index_select(0, padding)).sum().item()
    return {"valid_rows": len(valid_rows), "padding_rows": len(padding_rows), "padding_differences": differences}
