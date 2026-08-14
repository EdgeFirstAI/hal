# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Multi-stage image for per-processor JPEG decode benchmarks.
# Builds every CPU arm as a native binary: `hal_cpu` + `rust_jpeg` (Rust) and
# `turbojpeg_bench` + `stb_bench` + `wuffs_bench` (C). Default entrypoint runs
# the decode-only YUV + RGB A/B matrix (no letterbox).
#
# Multi-arch: both stages use official multi-arch bases and everything builds
# natively in-container, so the same file serves x86 and Graviton queues:
#
#   docker build -f benchmarks/docker/Dockerfile -t edgefirst-hal-jpeg-bench .
#   docker buildx build --platform linux/arm64 -f benchmarks/docker/Dockerfile \
#     -t edgefirst-hal-jpeg-bench:arm64 .
#
# See benchmarks/README.md § Docker / multi-processor.

FROM rust:bookworm AS build

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY benchmarks ./benchmarks

WORKDIR /src/benchmarks
RUN cargo build --profile profiling -p hal_cpu -p rust_jpeg \
    && cp target/profiling/hal_cpu /usr/local/bin/hal_cpu \
    && cp target/profiling/rust_jpeg /usr/local/bin/rust_jpeg

# The comparator arms are native C binaries for the same reason hal_cpu is a
# native Rust one: this image measures decoding libraries against each other,
# so no arm may carry an interpreter inside the timed region. `clean` first,
# because a host build tree in the context would otherwise satisfy make and
# ship binaries linked against the host's glibc (stb/wuffs then re-fetch their
# pinned, sha256-checked single-file sources inside the container).
RUN make -C modules/turbojpeg clean \
    && make -C modules/turbojpeg \
    && cp modules/turbojpeg/build/turbojpeg_bench /usr/local/bin/turbojpeg_bench
RUN make -C modules/stb clean \
    && make -C modules/stb \
    && cp modules/stb/build/stb_bench /usr/local/bin/stb_bench
RUN make -C modules/wuffs clean \
    && make -C modules/wuffs \
    && cp modules/wuffs/build/wuffs_bench /usr/local/bin/wuffs_bench

FROM debian:bookworm-slim AS runtime

# awscli: lets a Batch job pull the corpus with its job-role credentials
# (DATASET_S3) with no secret handling; curl/unzip for DATASET_URL archives.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libturbojpeg0 \
        awscli \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/hal_cpu /usr/local/bin/hal_cpu
COPY --from=build /usr/local/bin/rust_jpeg /usr/local/bin/rust_jpeg
COPY --from=build /usr/local/bin/turbojpeg_bench /usr/local/bin/turbojpeg_bench
COPY --from=build /usr/local/bin/stb_bench /usr/local/bin/stb_bench
COPY --from=build /usr/local/bin/wuffs_bench /usr/local/bin/wuffs_bench
COPY benchmarks/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# Tiny always-present fixtures + optional 50-image COCO smoke set (gitignored
# host staging under benchmarks/docker/coco-smoke/).
COPY testdata/zidane.jpg testdata/giraffe.jpg testdata/grey.jpg /opt/testdata/
COPY benchmarks/docker/coco-smoke/ /opt/coco-smoke/

RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/hal_cpu \
    /usr/local/bin/rust_jpeg /usr/local/bin/turbojpeg_bench \
    /usr/local/bin/stb_bench /usr/local/bin/wuffs_bench

ENV RESULTS_DIR=/results \
    EDGEFIRST_BENCH_COCO=/data/coco \
    LIMIT=50 \
    WARMUP=10 \
    BOARD=unknown \
    TENSOR_MEM=mem \
    MODULES=hal_cpu,turbojpeg,zune,image,stb,wuffs

VOLUME ["/results", "/data/coco"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
