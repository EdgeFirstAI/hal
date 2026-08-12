# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Multi-stage image for per-processor JPEG decode benchmarks.
# Builds both arms as native binaries: `hal_cpu` (Rust) and `turbojpeg_bench`
# (C). Default entrypoint runs the decode-only YUV + RGB A/B (no letterbox).
#
# Build from the repository root (optionally after staging coco-smoke/):
#
#   docker build -f benchmarks/docker/Dockerfile -t edgefirst-hal-jpeg-bench .
#
# See benchmarks/README.md § Docker / multi-processor.

FROM rust:bookworm AS build

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY benchmarks ./benchmarks

WORKDIR /src/benchmarks
RUN cargo build --profile profiling -p hal_cpu \
    && cp target/profiling/hal_cpu /usr/local/bin/hal_cpu

# The TurboJPEG arm is a native C binary for the same reason hal_cpu is a native
# Rust one: this image measures two decoding libraries against each other, so
# neither may carry an interpreter inside the timed region. `clean` first,
# because a host build tree in the context would otherwise satisfy make and ship
# a binary linked against the host's glibc.
RUN make -C modules/turbojpeg clean \
    && make -C modules/turbojpeg \
    && cp modules/turbojpeg/build/turbojpeg_bench /usr/local/bin/turbojpeg_bench

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libturbojpeg0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/hal_cpu /usr/local/bin/hal_cpu
COPY --from=build /usr/local/bin/turbojpeg_bench /usr/local/bin/turbojpeg_bench
COPY benchmarks/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# Tiny always-present fixtures + optional 50-image COCO smoke set (gitignored
# host staging under benchmarks/docker/coco-smoke/).
COPY testdata/zidane.jpg testdata/giraffe.jpg testdata/grey.jpg /opt/testdata/
COPY benchmarks/docker/coco-smoke/ /opt/coco-smoke/

RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/hal_cpu \
    /usr/local/bin/turbojpeg_bench

ENV RESULTS_DIR=/results \
    EDGEFIRST_BENCH_COCO=/data/coco \
    LIMIT=50 \
    WARMUP=10 \
    BOARD=unknown \
    TENSOR_MEM=mem \
    MODULES=hal_cpu,turbojpeg

VOLUME ["/results", "/data/coco"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
