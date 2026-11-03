<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-kernels-capi — Architecture

C ABI for `edgefirst-kernels`. Ships as **`libedgefirst_kernels`** with `edgefirst/kernels.h`.

Tensors cross as caller-owned `ef_kernel_view`s, so this library has **no** `DT_NEEDED` on tensor or any other sibling; `scripts/check-single-home.sh` gate G2t checks the built library for it.

Source layout: `types.rs` (the `#[repr(C)]` values), `view.rs` (view to tensor conversion), `pool.rs` (C pool to `Pool` trampoline), `error.rs` (errno plus thread-local last error, `catch_unwind` guard), `plans.rs` (entry points).

Every `plan_new` and `plan_run` entry point returns 0 or -1 with `errno` set: `EINVAL` for NULL pointers, unknown dtype or box-format values, rank above 6, a view whose `data` is not aligned to its element type, an input and output whose memory spans overlap (interleaved views of one buffer included), `InvalidArgument`, or a caught panic ("internal error"); `ENOTSUP` for `Unsupported` and `ForcedTierUnavailable`; `ENOMEM` when copying the level list fails. `ef_kernels_last_error()` returns the thread's message, or NULL after a successful call; the pointer stays valid until that thread's next `plan_new` or `plan_run`. A failed `plan_new` always leaves `*out` NULL when `out` is non-NULL. `plan_free` returns nothing, treats NULL as a no-op, and, like every entry point, absorbs a panic instead of unwinding into C.

`plan_run` allocates nothing on success. `f16` tensors cross as `uint16_t` storage. Before building any slice it computes both views' byte spans (`edgefirst_kernels::ByteSpan`) from the data pointers and the validated layouts and rejects an overlap, because a Rust `&[T]` and `&mut [T]` over shared bytes would be undefined behaviour even where the views address disjoint elements.

**Pools.** `pool.rs` hands the C `run` a trampoline that catches a panicking task instead of unwinding into C. Such a task's block was already claimed and is not rerun, so the adapter records the panic and `plan_run` returns -1 / `EINVAL` / "internal error" rather than 0 with partial output. Worker threads need about 64 KiB of stack headroom for the kernels' staging buffers.

**ABI versioning.** `ef_kernels_abi_version()` returns `1`. Any change to a C type's layout, including a field added to any params struct, bumps it; the governing rules are in [§ C ABI Stability and Versioning](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#c-abi-stability-and-versioning).

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
