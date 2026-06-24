use gl_generator::{Api, Fallbacks, GlobalGenerator, Profile, Registry};
use std::env;
use std::fs::File;
use std::path::Path;

fn main() {
    // GLES 3.2 core plus the EGL-image import extensions used for zero-copy
    // DMA-BUF / IOSurface texturing. This matches the configuration the HAL
    // previously consumed from `gls` (its default `gles3` feature).
    let extensions = [
        "GL_OES_EGL_image",
        "GL_OES_EGL_image_external",
        "GL_EXT_YUV_target",
    ];

    let dest = env::var("OUT_DIR").unwrap();
    let mut file = File::create(Path::new(&dest).join("gl_bindings.rs")).unwrap();
    Registry::new(
        Api::Gles2,
        (3, 2),
        Profile::Core,
        Fallbacks::All,
        extensions,
    )
    .write_bindings(GlobalGenerator, &mut file)
    .unwrap();
}
