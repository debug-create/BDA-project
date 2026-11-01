import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "analysis" / "ops_v2" / "dl_model.py"
spec = importlib.util.spec_from_file_location("ops_v2_dl_model", MODULE)
dl_model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dl_model
spec.loader.exec_module(dl_model)


def tiny_model(**changes):
    vocab = {name: 4 for name in dl_model.EMBED_DIMS}
    options = dict(dense_dim=3, vocab_sizes=vocab, state_channels={"origin": 3}, width=16, dropout=0.0)
    options.update(changes)
    return dl_model.FlightDelayResidualNet(**options)


def inputs(batch=4):
    return (
        torch.randn(batch, 3),
        torch.ones(batch, len(dl_model.EMBED_DIMS), dtype=torch.long),
        {"origin": torch.randn(batch, 6, 3)},
        torch.linspace(-1, 1, batch),
    )


def test_zero_main_head_equals_lr_offset():
    model = tiny_model().eval()
    dense, categories, states, offset = inputs()
    output = model(dense, categories, states, offset)
    torch.testing.assert_close(output["main"], offset)


def test_unknown_ids_are_supported():
    model = tiny_model().eval()
    dense, categories, states, offset = inputs()
    categories.zero_()
    output = model(dense, categories, states, offset)
    assert output["main"].shape == (4,)


def test_auxiliary_targets_are_not_forward_arguments_and_gradients_are_valid():
    model = tiny_model()
    assert "arr_minutes" not in model.forward.__code__.co_varnames
    dense, categories, states, offset = inputs()
    output = model(dense, categories, states, offset)
    (output["main"].sum() + output["minutes"].sum() + output["dep"].sum() + output["bucket"].sum()).backward()
    assert model.main.weight.grad is not None
    assert model.aux_minutes.weight.grad is not None


def test_same_seed_gives_identical_initial_scores():
    torch.manual_seed(17)
    first = tiny_model().eval()
    torch.manual_seed(17)
    second = tiny_model().eval()
    batch = inputs()
    torch.testing.assert_close(first(*batch)["main"], second(*batch)["main"])


def test_forbidden_outcomes_are_not_model_inputs():
    forbidden = {"ArrDel15", "ArrDelay", "DepDelay", "DepDel15", "bucket", "Cancelled", "Diverted", "Tail_Number"}
    assert forbidden.isdisjoint(dl_model.FlightDelayResidualNet.forward.__code__.co_varnames)
