#!/bin/bash
# One-command rollback script for CUDA graph key change
set -e
cd /opt/llama.cpp-optium
git checkout -- ggml/src/ggml-cuda/ggml-cuda.cu
echo "Rollback complete: restored original ggml_cuda_graph_get_key"
