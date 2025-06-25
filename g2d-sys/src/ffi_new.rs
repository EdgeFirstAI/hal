use crate::{g2d_blend_func, g2d_format, g2d_rotation};

// g2d 2.3.0 changed the size of the g2d_phys_addr so we need this for
// compatibility
pub type g2d_phys_addr_t_new = ::std::os::raw::c_ulong;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct g2d_surface_new {
    pub format: g2d_format,
    pub planes: [g2d_phys_addr_t_new; 3usize],
    pub left: ::std::os::raw::c_int,
    pub top: ::std::os::raw::c_int,
    pub right: ::std::os::raw::c_int,
    pub bottom: ::std::os::raw::c_int,
    #[doc = "< buffer stride, in Pixels"]
    pub stride: ::std::os::raw::c_int,
    #[doc = "< surface width, in Pixels"]
    pub width: ::std::os::raw::c_int,
    #[doc = "< surface height, in Pixels"]
    pub height: ::std::os::raw::c_int,
    #[doc = "< alpha blending parameters"]
    pub blendfunc: g2d_blend_func,
    #[doc = "< value is 0 ~ 255"]
    pub global_alpha: ::std::os::raw::c_int,
    pub clrcolor: ::std::os::raw::c_int,
    pub rot: g2d_rotation,
}
