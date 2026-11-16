# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Multi-stage image for per-processor JPEG decode benchmarks.
# Builds every CPU arm as a native binary: `hal_cpu` + `rust_jpeg` (Rust) and
# `turbojpeg_bench` + `stb_bench` + `wuffs_bench` (C). Default entrypoint runs
# the decode-only YUV + RGB A/B matrix (no letterbox).
#
# Multi-arch: both stages use official multi-arch bases and everything builds
# natively in-container:
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

# Build the comparator arms as native C binaries, from a clean tree.
RUN make -C modules/turbojpeg clean \
    && make -C modules/turbojpeg \
    && cp modules/turbojpeg/build/turbojpeg_bench /usr/local/bin/turbojpeg_bench
RUN make -C modules/stb clean \
    && make -C modules/stb \
    && cp modules/stb/build/stb_bench /usr/local/bin/stb_bench
RUN make -C modules/wuffs clean \
    && make -C modules/wuffs \
    && cp modules/wuffs/build/wuffs_bench /usr/local/bin/wuffs_bench

# Records the build toolchain versions for the runtime image.
RUN { rustc --version; cargo --version; gcc --version | head -1; } > /opt/build-provenance.txt

# --- Source-built libjpeg-turbo control: an alternate .so in the same image,
# selected per job with EDGEFIRST_TURBOJPEG_LIB. nasm provides the x86 SIMD;
# cmake selects the ASM backend per arch.
#
# The source is fetched at a pinned commit and SHA-256 verified before
# extraction.
FROM build AS turbo-src
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake nasm \
    && rm -rf /var/lib/apt/lists/*
ARG TURBO_SRC_VERSION=3.1.2
ARG TURBO_SRC_COMMIT=4e151a4ad91001b3aa8c2ece2205c15f487ce320
ARG TURBO_SRC_SHA256=de49f4f268ac2845173c0a51055e1a78cf4636b05724c0c31dfc2af8284da962
RUN curl -fSL -o /tmp/turbo-src.tar.gz \
        "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/${TURBO_SRC_COMMIT}.tar.gz" \
    && echo "${TURBO_SRC_SHA256}  /tmp/turbo-src.tar.gz" | sha256sum -c - \
    && mkdir -p /tmp/turbo-src \
    && tar -xzf /tmp/turbo-src.tar.gz -C /tmp/turbo-src --strip-components=1 \
    && cmake -S /tmp/turbo-src -B /tmp/turbo-build \
        -DCMAKE_BUILD_TYPE=Release -DENABLE_STATIC=OFF -DWITH_JPEG8=1 \
    && cmake --build /tmp/turbo-build --parallel \
    && cp /tmp/turbo-build/libturbojpeg.so.0.* /opt/libturbojpeg-src.so \
    && echo "${TURBO_SRC_VERSION}" > /opt/turbo-src-version.txt

FROM debian:bookworm-slim AS runtime

# awscli pulls the corpus for DATASET_S3; curl/unzip handle DATASET_URL archives.
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
COPY --from=build /opt/build-provenance.txt /opt/build-provenance.txt
COPY --from=turbo-src /opt/libturbojpeg-src.so /opt/libturbojpeg-src.so
COPY --from=turbo-src /opt/turbo-src-version.txt /opt/turbo-src-version.txt
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
