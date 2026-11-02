import pytest
import torch

from layer_output_validation import check_hidden_tensor, check_topk_outputs, hidden_valid_rows


def test_padding_is_observed_but_every_valid_index_remains_exact():
    expected = torch.tensor([[0, 1, 2], [5, 6, 7]], dtype=torch.int32)
    observed = expected.clone()
    observed[1] = -999
    result = check_topk_outputs(observed, expected, [8192, 0])
    assert result == {"valid_rows": 1, "padding_rows": 1, "padding_differences": 3}
    observed[0, 2] = 99
    with pytest.raises(AssertionError):
        check_topk_outputs(observed, expected, [8192, 0])


def test_idle_rank_has_no_valid_indexer_rows():
    expected = torch.zeros((4, 2048), dtype=torch.int32)
    result = check_topk_outputs(expected + 9, expected, [0, 0, 0, 0])
    assert result["valid_rows"] == 0
    assert result["padding_differences"] == 8192


@pytest.mark.parametrize("lengths", [[8192], [8192, -1]])
def test_unknown_validity_contract_fails(lengths):
    indices = torch.zeros((2, 2048), dtype=torch.int32)
    with pytest.raises(AssertionError):
        check_topk_outputs(indices, indices, lengths)


def test_missing_or_wrong_dtype_output_fails():
    indices = torch.zeros((1, 2048), dtype=torch.int32)
    with pytest.raises(AssertionError):
        check_topk_outputs(None, indices, [8192])
    with pytest.raises(AssertionError):
        check_topk_outputs(indices.float(), indices, [8192])


def test_global_hidden_output_checks_remote_real_rows_even_on_idle_dp():
    lengths = [[8192, 0, 0, 0], [8192, 0, 0, 0], [0] * 4, [0] * 4]
    assert hidden_valid_rows(16, lengths[2], lengths, 0, 4) == [0, 4]
    assert hidden_valid_rows(4, lengths[2], lengths, 0, 4) == []
    assert hidden_valid_rows(1, lengths[0], lengths, 0, 4) == [0]
    assert hidden_valid_rows(1, lengths[0], lengths, 1, 4) == []
    expected = torch.zeros((16, 6144), dtype=torch.bfloat16)
    observed = expected.clone()
    observed[4, 13] = 1.0
    with pytest.raises(AssertionError):
        check_hidden_tensor(observed, expected, lengths[2], lengths, 0, 4)


def test_undefined_hidden_padding_is_recorded_without_relaxing_real_rows():
    lengths = [[8192, 0, 0, 0]] * 4
    expected = torch.zeros((4, 6144), dtype=torch.bfloat16)
    observed = expected.clone()
    observed[1] = 10.0
    result = check_hidden_tensor(observed, expected, lengths[0], lengths, 0, 4)
    assert result == {"valid_rows": 1, "padding_rows": 3, "padding_differences": 6144}
    observed[0, 17] = 1.0
    with pytest.raises(AssertionError):
        check_hidden_tensor(observed, expected, lengths[0], lengths, 0, 4)


def test_unknown_hidden_layout_is_not_silently_skipped():
    with pytest.raises(AssertionError):
        hidden_valid_rows(2, [8192, 0, 0, 0], [[8192, 0, 0, 0]] * 4, 0, 4)
