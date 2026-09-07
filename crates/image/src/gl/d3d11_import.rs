// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `EGL_ANGLE_image_d3d11_texture` attribute assembly. Pure so it is unit
//! tested without a display; the leaf passes the result to `eglCreateImage`.

use edgefirst_egl as egl;

/// `eglCreateImage` target for an `ID3D11Texture2D` client buffer.
pub(super) const EGL_D3D11_TEXTURE_ANGLE: egl::Enum = 0x3484;
pub(super) const EGL_TEXTURE_INTERNAL_FORMAT_ANGLE: egl::Attrib = 0x345D;
/// Plane index for YUV DXGI formats; unused by the HAL's combined-plane R8
/// layout, kept for the native NV12 opt-in.
pub(super) const EGL_D3D11_TEXTURE_PLANE_ANGLE: egl::Attrib = 0x3492;

/// Assemble the `eglCreateImage` attribute list for an `ID3D11Texture2D`
/// client buffer: internal format, an optional plane index, terminated by
/// `EGL_NONE`.
pub(super) fn d3d11_image_attribs(gl_internal_format: u32, plane: Option<u32>) -> Vec<egl::Attrib> {
    let mut a = vec![
        EGL_TEXTURE_INTERNAL_FORMAT_ANGLE,
        gl_internal_format as egl::Attrib,
    ];
    if let Some(p) = plane {
        a.extend([EGL_D3D11_TEXTURE_PLANE_ANGLE, p as egl::Attrib]);
    }
    a.push(edgefirst_egl::NONE as egl::Attrib);
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attribs_carry_the_internal_format_and_end_with_none() {
        let a = d3d11_image_attribs(0x1908, None);
        assert_eq!(
            a,
            vec![
                EGL_TEXTURE_INTERNAL_FORMAT_ANGLE,
                0x1908,
                edgefirst_egl::NONE as egl::Attrib
            ]
        );
        let b = d3d11_image_attribs(0x1903, Some(1));
        assert_eq!(
            b,
            vec![
                EGL_TEXTURE_INTERNAL_FORMAT_ANGLE,
                0x1903,
                EGL_D3D11_TEXTURE_PLANE_ANGLE,
                1,
                edgefirst_egl::NONE as egl::Attrib
            ]
        );
    }
}
