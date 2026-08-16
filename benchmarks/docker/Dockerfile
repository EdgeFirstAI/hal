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

# Build/toolchain provenance for the runtime image (see BENCHMARKS.md §1.2/§2.2
# of the v3.12 review): the runtime stage below is debian-slim with no
# compilers installed, so this has to be captured here and carried across.
RUN { rustc --version; cargo --version; gcc --version | head -1; } > /opt/build-provenance.txt

# --- Source-built libjpeg-turbo control (review §1.2/§2.2): baked into the
# same image as an alternate .so, not a separate image variant, so a job
# opts into it via EDGEFIRST_TURBOJPEG_LIB rather than pulling a different
# tag. Settles whether the ARM ~1.16x vs ~1.36x split (packaged turbo 3.x vs
# 2.1.x on the physical boards) is a version confound or coincidence, on
# Graviton. Version pinned to what BENCHMARKS.md's board provenance table
# already asks for ("3.1.2 or later"); nasm needed for x86 SIMD, cmake
# selects the right ASM backend per arch automatically.
#
# Pinned by immutable commit (the tag ref is movable) and SHA-256 verified
# before extraction, same reproducibility bar as the stb/wuffs single-file
# fetches above (Copilot review on PR #145).
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
