# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from online_draft.runtime.weights import DraftWeightInstaller, DraftWeightSnapshot


def _slots() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    active = {
        "weight": torch.tensor([1.0, 2.0]),
        "frozen": torch.tensor([7.0]),
    }
    staging = {name: tensor.clone() for name, tensor in active.items()}
    return active, staging


def test_slot_derived_weights_are_prepared_before_commit_and_reset():
    derived = [torch.zeros(2), torch.zeros(2)]

    def prepare(slot, weights):
        derived[slot].copy_(weights["weight"] * 2)

    installer = DraftWeightInstaller(
        _slots(),
        mutable_names=("weight",),
        prepare_slot=prepare,
    )
    installer.stage(
        DraftWeightSnapshot.from_named_tensors(
            [("weight", torch.tensor([3.0, 4.0]))],
            version=1,
            pin_memory=False,
        )
    )
    assert installer.active_version == 0
    torch.testing.assert_close(derived[1], torch.tensor([6.0, 8.0]))
    assert installer.commit_if_ready() == 1
    installer.reset((("weight", torch.tensor([1.0, 2.0])),))
    torch.testing.assert_close(derived[1], torch.tensor([2.0, 4.0]))


def test_commit_only_queries_incomplete_transfer():
    class PendingEvent:
        def query(self):
            return False

        def synchronize(self):
            pytest.fail("commit must not wait for a transfer")

    installer = DraftWeightInstaller(_slots(), mutable_names=("weight",))
    snapshot = DraftWeightSnapshot.from_named_tensors(
        [("weight", torch.tensor([3.0, 4.0]))],
        version=1,
        pin_memory=False,
    )
    installer._staged = (snapshot, PendingEvent())
    assert installer.commit_if_ready() is None
    assert installer.active_version == 0


def test_snapshot_owns_cpu_copy() -> None:
    source = [("weight", torch.tensor([1.0, 2.0]))]
    snapshot = DraftWeightSnapshot.from_named_tensors(
        source,
        version=1,
        pin_memory=False,
    )

    source[0][1].fill_(9.0)

    assert snapshot.version == 1
    assert torch.equal(snapshot.tensors[0][1], torch.tensor([1.0, 2.0]))
    assert snapshot.tensors[0][1].device.type == "cpu"


def test_installer_switches_only_after_staging() -> None:
    installer = DraftWeightInstaller(
        _slots(),
        mutable_names=("weight",),
    )
    snapshot = DraftWeightSnapshot.from_named_tensors(
        [("weight", torch.tensor([3.0, 4.0]))],
        version=1,
        pin_memory=False,
    )

    assert installer.active_version == 0
    assert installer.commit_if_ready() is None

    installer.stage(snapshot)

    assert installer.staged_version == 1
    assert installer.active_version == 0
    assert installer.commit_if_ready() == 1
    assert installer.staged_version is None
    assert torch.equal(
        installer.active_weights["weight"],
        torch.tensor([3.0, 4.0]),
    )
    assert torch.equal(installer.active_weights["frozen"], torch.tensor([7.0]))


def test_installer_rejects_two_pending_updates() -> None:
    installer = DraftWeightInstaller(
        _slots(),
        mutable_names=("weight",),
    )
    first = DraftWeightSnapshot.from_named_tensors(
        [("weight", torch.tensor([3.0, 4.0]))],
        version=1,
        pin_memory=False,
    )
    second = DraftWeightSnapshot.from_named_tensors(
        [("weight", torch.tensor([5.0, 6.0]))],
        version=2,
        pin_memory=False,
    )

    installer.stage(first)
    with pytest.raises(RuntimeError, match="already staged"):
        installer.stage(second)


def test_installer_reset_restores_initial_weights() -> None:
    slots = _slots()
    initial_weights = (("weight", slots[0]["weight"].clone()),)
    installer = DraftWeightInstaller(
        slots,
        mutable_names=("weight",),
    )

    for version, values in (
        (1, [3.0, 4.0]),
        (2, [5.0, 6.0]),
    ):
        snapshot = DraftWeightSnapshot.from_named_tensors(
            [("weight", torch.tensor(values))],
            version=version,
            pin_memory=False,
        )
        installer.stage(snapshot)
        assert installer.commit_if_ready() == version

    assert installer.active_version == 2
    assert torch.equal(
        installer.active_weights["weight"],
        torch.tensor([5.0, 6.0]),
    )

    installer.reset(initial_weights)

    assert installer.active_version == 0
    assert installer.staged_version is None
    assert torch.equal(
        installer.active_weights["weight"],
        torch.tensor([1.0, 2.0]),
    )
    assert torch.equal(
        installer.active_weights["frozen"],
        torch.tensor([7.0]),
    )


def test_installer_reset_discards_staged_update() -> None:
    slots = _slots()
    initial_weights = (("weight", slots[0]["weight"].clone()),)
    installer = DraftWeightInstaller(
        slots,
        mutable_names=("weight",),
    )

    first = DraftWeightSnapshot.from_named_tensors(
        [("weight", torch.tensor([3.0, 4.0]))],
        version=1,
        pin_memory=False,
    )
    pending = DraftWeightSnapshot.from_named_tensors(
        [("weight", torch.tensor([5.0, 6.0]))],
        version=2,
        pin_memory=False,
    )

    installer.stage(first)
    assert installer.commit_if_ready() == 1

    installer.stage(pending)
    assert installer.staged_version == 2

    installer.reset(initial_weights)

    assert installer.active_version == 0
    assert installer.staged_version is None
    assert installer.commit_if_ready() is None
    assert torch.equal(
        installer.active_weights["weight"],
        torch.tensor([1.0, 2.0]),
    )
