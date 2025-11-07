Tests must be run without "extension-module" feature. This is due to an [known issue](https://pyo3.rs/v0.26.0/faq#i-cant-run-cargo-test-im-having-linker-issues-like-symbol-not-found-or-undefined-reference-to-_pyexc_systemerror) in PyO3 
```
cargo test --no-default-features
```