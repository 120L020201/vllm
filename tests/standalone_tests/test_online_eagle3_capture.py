# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.online_eagle3 import OnlineEagle3Adapter


class _CaptureBridge:
    def observe_step(self, request_id, step_id, payload):
        self.observation = (request_id, step_id, payload)

    def reset_request(self, request_id):
        self.reset_id = request_id

    def maybe_apply_pending_weights(self):
        return None

    def shutdown(self):
        self.closed = True


class _Speculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


def _embedding(ids):
    return ids.float().unsqueeze(-1).expand(-1, 8)


@pytest.mark.parametrize("count", [1, 3, 5])
def test_verify_projection_and_confirmed_path_alignment(count):
    bridge = _CaptureBridge()
    model = SimpleNamespace(
        draft_id_to_target_id=torch.tensor([1, 2]), embed_input_ids=_embedding
    )
    adapter = OnlineEagle3Adapter(model, bridge, 4)
    adapter._pending = {}
    adapter._request_id = "request"
    adapter._step_id = 7
    logits = torch.arange(20).view(5, 4).float()
    adapter.capture_teacher_logits(logits)
    logits.fill_(-100)  # Sampling may reuse or change its buffers.
    tokens = torch.tensor([10, 11, 12, 13, 14])
    sampled = torch.full((1, 5), -1)
    sampled[0, : count - 1] = tokens[1:count]
    sampled[0, count - 1] = 99
    auxiliary = torch.arange(5).view(5, 1).float()
    adapter.observe_verify(
        SimpleNamespace(
            req_ids=["request"], input_ids=tokens, positions=torch.arange(3, 8)
        ),
        sampled,
        torch.tensor([count]),
        [auxiliary, auxiliary + 10, auxiliary + 20],
    )
    request_id, step_id, payload = bridge.observation
    assert (request_id, step_id) == ("request", 7)
    expected_probs = torch.tensor([1.0, 3.0]).softmax(-1).expand(4, -1)
    torch.testing.assert_close(payload["teacher_probs"], expected_probs)
    assert torch.equal(
        payload["confirmed_input_embeds"][:, 0], sampled[0, :count].float()
    )
    assert torch.equal(payload["confirmed_positions"], torch.arange(3, 3 + count))
    torch.testing.assert_close(
        payload["confirmed_aux_hidden_states"],
        torch.cat((auxiliary, auxiliary + 10, auxiliary + 20), -1)[:count],
    )
    assert adapter._pending is None


def test_decode_replay_records_inputs_before_each_graph_execution():
    speculator = _Speculator.__new__(_Speculator)
    speculator.num_speculative_steps = 4
    speculator.model = SimpleNamespace(embed_input_ids=_embedding)
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.tensor([10]),
        positions=torch.tensor([3]),
        query_start_loc=torch.tensor([0, 1]),
    )
    speculator.idx_mapping = torch.tensor([0])
    speculator.req_indices = torch.tensor([0])
    speculator.current_draft_step = torch.tensor(0)
    speculator.hidden_states = torch.zeros(1, 8)
    speculator.online_training = OnlineEagle3Adapter(
        speculator.model, _CaptureBridge(), 4
    )
    speculator.online_training._trace = {
        key: []
        for key in (
            "proposal_input_ids",
            "proposal_input_embeds",
            "proposal_positions",
            "proposal_hidden_states",
        )
    }

    def replay(_descriptor):
        speculator.input_buffers.input_ids.add_(1)
        speculator.input_buffers.positions.add_(1)
        speculator.hidden_states.add_(1)

    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=replay)
    speculator._multi_step_decode(
        1, True, SimpleNamespace(cg_mode=CUDAGraphMode.FULL), None
    )
    trace = speculator.online_training._trace
    assert torch.equal(
        torch.cat(trace["proposal_input_ids"]), torch.tensor([10, 11, 12])
    )
    assert torch.equal(torch.cat(trace["proposal_positions"]), torch.tensor([3, 4, 5]))
    assert torch.equal(
        torch.cat(trace["proposal_hidden_states"])[:, 0], torch.arange(3).float()
    )


def test_adapter_captures_owned_inputs_and_clears_request():
    bridge = _CaptureBridge()
    adapter = OnlineEagle3Adapter(
        SimpleNamespace(embed_input_ids=_embedding), bridge, 1
    )
    auxiliary = torch.randn(3, 24)
    ids, positions, hidden = torch.arange(3), torch.arange(3), torch.randn(3, 8)
    original_hidden = hidden.clone()
    adapter.start_proposal(SimpleNamespace(num_reqs=1, req_ids=["req"]), auxiliary)
    adapter.record_prefill(ids, positions, hidden, torch.tensor([2]))
    adapter.finish_proposal()
    hidden.zero_()
    assert torch.equal(adapter._pending["proposal_hidden_states"], original_hidden[-1:])
    adapter.reset_request("unrelated")
    assert adapter._pending is not None
    adapter.reset_request("req")
    assert adapter._pending is None and bridge.reset_id == "req"
    adapter.start_proposal(
        SimpleNamespace(num_reqs=1, req_ids=["_warmup_0"]), auxiliary
    )
    assert adapter._trace is None
    adapter.shutdown()
    assert bridge.closed


def test_adapter_rejects_incomplete_and_wrong_request():
    adapter = OnlineEagle3Adapter(SimpleNamespace(), _CaptureBridge(), 4)
    adapter._pending = {}
    adapter._request_id = "req"
    with pytest.raises(ValueError, match="complete verification"):
        adapter.capture_teacher_logits(torch.zeros(3, 8))
    with pytest.raises(ValueError, match="does not match"):
        adapter.observe_verify(SimpleNamespace(req_ids=["wrong"]), None, None, None)
