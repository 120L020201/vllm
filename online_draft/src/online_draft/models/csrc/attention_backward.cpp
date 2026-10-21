// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/CPUBlas.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using at::Tensor;
using at::native::cpublas::brgemm;

// BF16 BRGEMM reads B as [K/2, N, 2]; pad odd reduction lengths with zero.
template <typename T>
Tensor pack(const Tensor& input) {
  if constexpr (!std::is_same_v<T, at::BFloat16>) return input;
  const auto h = input.size(0), rows = input.size(1), cols = input.size(2);
  const auto pairs = (rows + 1) / 2;
  auto output = at::empty({h, pairs * 2, cols}, input.options());
  auto* dst = output.data_ptr<T>();
  const auto* src = input.data_ptr<T>();
  at::parallel_for(0, h * pairs, 1, [&](int64_t begin, int64_t end) {
    for (auto task = begin; task < end; ++task) {
      const auto head = task / pairs, row = task % pairs * 2;
      for (int64_t j = 0; j < cols; ++j) {
        dst[(task * cols + j) * 2] = src[(head * rows + row) * cols + j];
        dst[(task * cols + j) * 2 + 1] =
            row + 1 < rows ? src[(head * rows + row + 1) * cols + j] : T(0);
      }
    }
  });
  return output;
}

template <typename T>
Tensor pack_transpose(const Tensor& input) {
  if constexpr (!std::is_same_v<T, at::BFloat16>)
    return input.transpose(1, 2).contiguous();
  const auto h = input.size(0), rows = input.size(1), d = input.size(2);
  const auto tiles = (rows + 511) / 512;
  auto result = at::empty(input.sizes(), input.options());
  auto* dst = result.data_ptr<T>();
  const auto* src = input.data_ptr<T>();
  at::parallel_for(0, h * tiles, 1, [&](int64_t begin, int64_t end) {
    for (auto task = begin; task < end; ++task) {
      const auto head = task / tiles, start = task % tiles * 512;
      const auto count = std::min(int64_t(512), rows - start);
      const auto base = (head * rows + start) * d;
      for (int64_t pair = 0; pair < d / 2; ++pair)
        for (int64_t j = 0; j < count; ++j) {
          dst[base + (pair * count + j) * 2] = src[base + j * d + pair * 2];
          dst[base + (pair * count + j) * 2 + 1] =
              src[base + j * d + pair * 2 + 1];
        }
    }
  });
  return result;
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> backward_impl(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o,
    const Tensor& lse, const Tensor& go, const Tensor& mask,
    int64_t train_start) {
  const auto heads = q.size(0), nq = q.size(1), d = q.size(2);
  const auto nk = k.size(1), nt = nk - train_start;
  constexpr int64_t q_block = 128;
  const auto tiles = (nq + q_block - 1) / q_block;
  constexpr int64_t k_block = 512;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  auto opts = q.options().dtype(at::kFloat);
  auto dq = at::zeros(q.sizes(), opts);
  auto dk = at::zeros({heads, tiles, nt, d}, opts);
  auto dv = at::zeros_like(dk);
  constexpr bool packed = std::is_same_v<T, at::BFloat16>;
  Tensor kt = pack_transpose<T>(k);
  Tensor vt = pack_transpose<T>(v);
  Tensor kr = pack<T>(k), qr = pack<T>(q), gr = pack<T>(go);
  const auto* qp = q.data_ptr<T>();
  const auto* ktp = kt.data_ptr<T>();
  const auto* vtp = vt.data_ptr<T>();
  const auto* krp = kr.data_ptr<T>();
  const auto* qrp = qr.data_ptr<T>();
  const auto* grp = gr.data_ptr<T>();
  const auto* op = o.data_ptr<T>();
  const auto* gp = go.data_ptr<T>();
  const auto* lp = lse.data_ptr<float>();
  const bool float_mask = mask.scalar_type() == at::kFloat;
  const auto* mp =
      mask.numel() && float_mask ? mask.data_ptr<float>() : nullptr;
  const auto* low_mask =
      mask.numel() && !float_mask ? mask.data_ptr<T>() : nullptr;
  auto* dqp = dq.data_ptr<float>();
  auto* dkp = dk.data_ptr<float>();
  auto* dvp = dv.data_ptr<float>();

  // Each query tile owns dQ and private suffix dK/dV; reduce after the barrier.
  at::parallel_for(0, heads * tiles, 1, [&](int64_t begin, int64_t end) {
    std::vector<float> p(q_block * k_block), dp(p.size()), delta(q_block);
    std::vector<float> mask_row(k_block);
    std::vector<T> probability(p.size()), ds(p.size());
    std::vector<T> pt(p.size()), dst(p.size());
    for (auto task = begin; task < end; ++task) {
      const auto h = task / tiles, tile = task % tiles;
      const auto qs = tile * q_block, rows = std::min(q_block, nq - qs);
      const auto offset = (h * nq + qs) * d;
      const auto* query = qp + offset;
      const auto* grad = gp + offset;
      for (int64_t i = 0; i < rows; ++i) {
        float sum = 0;
        for (int64_t j = 0; j < d; ++j)
          sum += float(grad[i * d + j]) * float(op[offset + i * d + j]);
        delta[i] = sum;
      }
      for (int64_t ks = 0; ks < nk; ks += k_block) {
        const auto cols = std::min(k_block, nk - ks);
        const auto cs = packed ? (cols + 1) / 2 * 2 : cols;
        brgemm(rows, cols, d, d, packed ? cols : nk, cols, false, query,
               ktp + h * d * nk + ks * (packed ? d : 1), p.data(), packed);
        brgemm(rows, cols, d, d, packed ? cols : nk, cols, false, grad,
               vtp + h * d * nk + ks * (packed ? d : 1), dp.data(), packed);
        for (int64_t i = 0; i < rows; ++i) {
          auto* row = p.data() + i * cols;
          const auto norm = lp[h * nq + qs + i];
          const auto* bias = mp ? mp + (qs + i) * nk + ks : nullptr;
          if (low_mask) {
            at::vec::convert(low_mask + (qs + i) * nk + ks, mask_row.data(),
                             cols);
            bias = mask_row.data();
          }
          using Vec = at::vec::Vectorized<float>;
          if (bias) {
            at::vec::map2<float>(
                [=](Vec a, Vec b) {
                  return (a * Vec(scale) + b - Vec(norm)).exp();
                },
                row, row, bias, cols);
          } else {
            at::vec::map<float>(
                [=](Vec a) { return (a * Vec(scale) - Vec(norm)).exp(); }, row,
                row, cols);
          }
          at::vec::convert(row, probability.data() + i * cols, cols);
          const float row_delta = delta[i];
          auto* grad_row = dp.data() + i * cols;
          at::vec::map2<float>(
              [=](Vec a, Vec b) { return a * (b - Vec(row_delta)); }, grad_row,
              row, grad_row, cols);
          at::vec::convert(grad_row, ds.data() + i * cs, cols);
          if (cs > cols) ds[i * cs + cols] = T(0);
        }
        brgemm(rows, d, cs, cs, d, d, true, ds.data(),
               krp + (h * kr.size(1) + ks) * d, dqp + offset, packed);
        // Frozen context contributes to dQ, but needs no dK or dV.
        const auto first = std::max(train_start, ks);
        const auto count = ks + cols - first;
        if (count > 0) {
          const auto partial = (task * nt + first - train_start) * d;
          const auto rs = packed ? (rows + 1) / 2 * 2 : rows;
          for (int64_t j = 0; j < count; ++j)
            for (int64_t i = 0; i < rs; ++i) {
              dst[j * rs + i] = i < rows ? ds[i * cs + first - ks + j] : T(0);
              pt[j * rs + i] =
                  i < rows ? probability[i * cols + first - ks + j] : T(0);
            }
          brgemm(count, d, rs, rs, d, d, true, dst.data(),
                 qrp + (h * qr.size(1) + qs) * d, dkp + partial, packed);
          brgemm(count, d, rs, rs, d, d, true, pt.data(),
                 grp + (h * gr.size(1) + qs) * d, dvp + partial, packed);
        }
      }
    }
    if (packed) at::native::cpublas::brgemm_release();
  });
  return {dq.mul_(scale).to(q.scalar_type()),
          dk.sum(1).mul_(scale).to(q.scalar_type()),
          dv.sum(1).to(q.scalar_type())};
}

std::tuple<Tensor, Tensor, Tensor> backward(const Tensor& q, const Tensor& k,
                                            const Tensor& v, const Tensor& o,
                                            const Tensor& lse, const Tensor& go,
                                            const Tensor& mask,
                                            int64_t train_start) {
  if (q.scalar_type() == at::kBFloat16)
    return backward_impl<at::BFloat16>(q, k, v, o, lse, go, mask, train_start);
  return backward_impl<float>(q, k, v, o, lse, go, mask, train_start);
}

}  // namespace

TORCH_LIBRARY(online_draft_cpu, m) {
  m.def(
      "attention_backward(Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, "
      "Tensor go, "
      "Tensor mask, int train_start) -> (Tensor, Tensor, Tensor)");
  m.impl("attention_backward", c10::kCPU, &backward);
}
