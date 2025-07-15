#!/bin/sh

bindgen --dynamic-loading g2d --allowlist-function 'g2d_.*' g2d.h > src/ffi.rs