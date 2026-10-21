"""Keep the VQ codebook embedding table in host memory and read it over PCIe.

`codebook_embeddings` is 40960 x 2560 fp16 - 200 MiB, about a twentieth of a
4 GB card - and a decode step reads ten of its rows: 50 KB. Everything else on
the card is read in full every frame, so this table is the cheapest 200 MiB to
give back. And it has to be given back: with it resident, the model, codec, KV
cache and CUDA context leave no headroom, and on Windows the driver then pages
GPU allocations out to system memory instead of failing - the codec's
convolutions went from ~40 ms to 2.5 s a call, silently.

Pinned host memory is mapped into the GPU's address space (UVA), so a kernel
can gather rows from it directly: no copy is issued, which also keeps the
lookup legal inside a captured CUDA graph. Ten rows cost a few microseconds of
PCIe latency; a 350-token prompt gathers 3500 rows in about a millisecond.
The values read are the table's own, so nothing about the output changes.
"""

from __future__ import annotations

import ctypes
import logging

from app.engine import nvrtc

logger = logging.getLogger(__name__)

THREADS = 128

_SOURCE = r"""
extern "C" __global__ void gather_rows(
    const uint4* __restrict__ table, const long long* __restrict__ idx,
    uint4* __restrict__ out, int rows, int vecs_per_row)
{
  const int r = blockIdx.x;
  long long i = idx[r];
  i = i < 0 ? 0 : (i >= rows ? rows - 1 : i);  // never read outside the table
  const uint4* src = table + i * vecs_per_row;
  uint4* dst = out + (long long)r * vecs_per_row;
  for (int v = threadIdx.x; v < vecs_per_row; v += blockDim.x) dst[v] = src[v];
}
"""

_kernel = None


class HostEmbedding:
    """Drop-in for `nn.Embedding.forward` over a pinned host table."""

    def __init__(self, weight_cpu_pinned, device) -> None:
        self.weight = weight_cpu_pinned
        self.device = device
        self.rows, self.dim = weight_cpu_pinned.shape
        self.vecs = self.dim * weight_cpu_pinned.element_size() // 16

    def __call__(self, idx):
        import torch

        flat = idx.reshape(-1).to(torch.long).contiguous()
        out = torch.empty((flat.numel(), self.dim), dtype=self.weight.dtype, device=self.device)
        if flat.numel():
            argv, _keep = nvrtc.pack(
                [
                    ctypes.c_void_p(self.weight.data_ptr()),
                    ctypes.c_void_p(flat.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_int(self.rows),
                    ctypes.c_int(self.vecs),
                ]
            )
            nvrtc.launch(_kernel, flat.numel(), THREADS, argv)
        return out.view(*idx.shape, self.dim)


def install(model) -> bool:
    """Move `model.codebook_embeddings` to pinned host memory. False if it cannot."""
    global _kernel
    import torch

    table = getattr(model, "codebook_embeddings", None)
    weight = getattr(table, "weight", None)
    if weight is None or weight.device.type != "cuda":
        return False
    if weight.shape[1] * weight.element_size() % 16:
        return False
    try:
        if _kernel is None:
            _kernel = nvrtc.build(_SOURCE, ["gather_rows"], label="hostembed")["gather_rows"]
    except Exception as exc:  # noqa: BLE001 - the table simply stays on the GPU
        logger.warning("host embedding gather unavailable (%s); table stays in VRAM", exc)
        return False

    device = weight.device
    main = getattr(model, "embeddings", None)
    if getattr(model, "embeddings_offloaded", False) and main is not None:
        # The 155776-row vocabulary table already lives in host memory; pinning
        # it lets prompt tokens be gathered by the GPU directly.
        pinned = main.weight.detach().pin_memory()
        main.weight.data = pinned
        model.host_embeddings = HostEmbedding(pinned, device)

    host = weight.detach().to("cpu").pin_memory()
    # Replacing the submodule drops the last reference to the device copy.
    del model._modules["codebook_embeddings"]
    object.__setattr__(model, "codebook_embeddings", HostEmbedding(host, device))
    del weight, table
    torch.cuda.empty_cache()
    logger.info(
        "codebook embeddings moved to pinned host memory (%.0f MiB of VRAM freed)",
        host.numel() * host.element_size() / 2**20,
    )
    return True
