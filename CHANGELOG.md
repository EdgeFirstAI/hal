# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-11-29

### Added
- Initial open source release under Apache-2.0 license
- **Core Crates**:
  - `edgefirst` - Main HAL crate re-exporting tensor, image, and decoder functionality
  - `edgefirst-tensor` - Zero-copy memory buffers with DMA-heap, shared memory, and heap allocation support
  - `edgefirst-image` - Hardware-accelerated image conversion and resizing (G2D, OpenGL, CPU fallback)
  - `edgefirst-decoder` - YOLO (v5/v8/v11) detection and segmentation decoder with NMS
  - `edgefirst-tracker` - ByteTrack multi-object tracking with Kalman filtering
  - `g2d-sys` - FFI bindings for NXP i.MX G2D hardware acceleration
- **Python Bindings** (`edgefirst-hal` on PyPI):
  - PyO3-based Python API with numpy integration
  - `TensorImage` for image loading and manipulation
  - `ImageProcessor` for format conversion and resizing
  - `Decoder` for YOLO model output post-processing
  - Support for Python 3.8, 3.9, 3.10, 3.11, 3.12
- **Image Processing Features**:
  - Format conversion: YUYV, NV12, RGB, RGBA, GREY, Planar RGB (8BPS)
  - Resize with various interpolation methods
  - Rotation (0°, 90°, 180°, 270°) and flip operations
  - Normalization modes: signed, unsigned, raw
- **Platform Support**:
  - Linux with i.MX hardware acceleration (G2D, OpenGL)
  - Linux generic with CPU fallback
  - macOS and Windows with heap memory tensors
- **CI/CD Infrastructure**:
  - GitHub Actions workflows for testing, coverage, and releases
  - Multi-platform wheel builds (Linux, macOS, Windows)
  - PyPI trusted publishing with OIDC
  - SBOM generation and license compliance checking
- Comprehensive documentation with architecture diagrams
- Apache-2.0 license with NOTICE file for third-party attributions

### Notes
- SonarCloud and codecov integrations intentionally not implemented per project requirements
- No CLI binaries are built (edgefirst-hal is a library-only project)
- `test_opengl_resize_8bps` test marked as `#[ignore]` pending testdata file
