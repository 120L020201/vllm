# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dedicated per-request CPU RNG, independent of vLLM sampling RNG."""

import math

import torch


class BernoulliGate:
    def __init__(self, probability: float, seed: int, stride: int):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("update probability must be in [0, 1]")
        if seed < 0 or stride < 1:
            raise ValueError("seed must be nonnegative and stride positive")
        self.probability = probability
        self.seed = seed
        self.stride = stride
        self.ordinal = 0
        self.rounds = {}
        self.generators = {}

    def select(self, request_id: str) -> bool:
        if request_id not in self.generators:
            generator = torch.Generator(device="cpu")
            generator.manual_seed((self.seed + self.ordinal) % (2**63 - 1))
            self.generators[request_id] = generator
            self.ordinal += 1
        round_number = self.rounds.get(request_id, 0) + 1
        self.rounds[request_id] = round_number
        if round_number % self.stride:
            return False
        return bool(
            torch.rand((), generator=self.generators[request_id]) < self.probability
        )

    def reset(self, request_id: str):
        self.rounds.pop(request_id, None)
        self.generators.pop(request_id, None)
