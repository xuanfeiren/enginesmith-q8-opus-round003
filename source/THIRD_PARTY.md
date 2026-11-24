# Third-party code

## nkilib (AWS Neuron NKI kernel library)

- Project: nkilib, shipped with the AWS Neuron SDK (`nkilib` package in the Neuron PyTorch/vLLM virtual environment,
  nki 0.6.0 toolchain), https://github.com/aws-neuron/nki-library
- License: Apache License 2.0 (each file keeps its `Copyright Amazon.com, Inc. or its affiliates` header).
- Files: `kernels/nkilib/**` — the subset of the package that the engine's kernels import (core/attention,
  core/qkv, core/mlp, core/output_projection, core/embeddings, core/max, core/quantization, core/subkernels,
  core/utils, experimental/transformer).
- Changes (marked `enginesmith patch` in the source):
  - `core/mlp/mlp.py`, `core/mlp/mlp_parameters.py`, `core/mlp/mlp_cte/mlp_cte_constants.py`,
    `core/mlp/mlp_cte/basic/mlp_cte_basic.py`: per-call name prefix for SBUF/PSUM tensors (several MLP calls per
    kernel), and `inter_chunk_offset` / `inter_chunk_size` so one call computes an intermediate-column chunk of
    the full gate/up/down weights (the CTE path supports at most 4096 intermediate columns per call).

  Apache License 2.0: http://www.apache.org/licenses/LICENSE-2.0
