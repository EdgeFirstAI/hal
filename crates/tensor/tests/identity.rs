use edgefirst_tensor::{BufferIdentity, IdentityKind};

#[cfg(target_os = "linux")]
use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait};

/// Allocates a small DMA-BUF tensor, or reports why it could not and returns
/// `None`. A dma-heap is a host resource, not something every CI runner or
/// dev machine provides, so absence must be a loud, visible skip rather than
/// a silently vacuous pass.
#[cfg(target_os = "linux")]
fn dma_tensor_or_skip() -> Option<Tensor<u8>> {
    match Tensor::<u8>::new(&[4096], Some(TensorMemory::DmaBuf), None) {
        Ok(t) => Some(t),
        Err(e) => {
            println!("SKIP: no dma-heap on this host ({e})");
            None
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn a_dup_of_the_same_dma_buf_has_the_same_identity() {
    // `from_fd` on a dup'd fd is what a cross-library import does. If dup
    // changed the identity, every such import would miss the GL cache --
    // the measured blocker this derivation exists to fix.
    let Some(t) = dma_tensor_or_skip() else {
        return;
    };
    let dup_fd = t.clone_fd().expect("dup");
    let dup = Tensor::<u8>::from_fd(dup_fd, t.shape(), None).expect("import dup");
    assert_eq!(
        t.buffer_identity().id(),
        dup.buffer_identity().id(),
        "dup must preserve identity: the same dma_buf is the same buffer"
    );
}

/// Allocates a small IOSurface-backed tensor, or reports why it could not and
/// returns `None` — a loud skip rather than a silently vacuous pass on a host
/// with no IOSurface support.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn iosurface_tensor_or_skip() -> Option<Tensor<u8>> {
    match Tensor::<u8>::new(&[4096], Some(TensorMemory::DmaBuf), None) {
        Ok(t) => Some(t),
        Err(e) => {
            println!("SKIP: no IOSurface support on this host ({e})");
            None
        }
    }
}

/// Re-wrap `t`'s underlying IOSurface as a second, independent tensor — the
/// cross-process/cross-library recovery path (`IOSurfaceLookup`'s in-process
/// analog: wrapping the same `IOSurfaceRef` a second time).
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn wrap_same_surface(t: &Tensor<u8>) -> Tensor<u8> {
    let ptr = t.iosurface_ref().expect("iosurface-backed tensor");
    unsafe { Tensor::<u8>::from_iosurface(ptr, t.shape(), None).expect("wrap same surface") }
}

#[test]
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn two_wraps_of_one_iosurface_share_an_identity() {
    // If two wraps of the same surface minted separate ids, the GL backend's
    // EGLImage import cache would miss (and re-import) on every wrap instead
    // of reusing the one already bound for this surface.
    let Some(t) = iosurface_tensor_or_skip() else {
        return;
    };
    let again = wrap_same_surface(&t);
    assert_eq!(
        t.buffer_identity().id(),
        again.buffer_identity().id(),
        "two wraps of the same IOSurface must share an identity"
    );
}

#[test]
fn the_same_system_key_yields_the_same_identity() {
    // The whole modular design rests on this: two independently-linked copies
    // of this crate must agree on a buffer's identity without sharing state.
    let a = BufferIdentity::derived(IdentityKind::DmaBuf, 4242);
    let b = BufferIdentity::derived(IdentityKind::DmaBuf, 4242);
    assert_eq!(a.id(), b.id(), "same key must give the same id");
}

#[test]
fn different_kinds_with_the_same_key_do_not_collide() {
    let a = BufferIdentity::derived(IdentityKind::DmaBuf, 7);
    let b = BufferIdentity::derived(IdentityKind::IoSurface, 7);
    assert_ne!(a.id(), b.id(), "kind must participate in the id");
}

#[test]
fn no_two_kinds_collapse_to_a_shared_constant() {
    // `derived` is `(kind << 56) ^ key`, so any kind used with a fixed key
    // yields a fixed id. This asserts the API cannot be used that way by
    // accident: distinct keys must give distinct ids for every kind.
    for k in [
        IdentityKind::DmaBuf,
        IdentityKind::Shm,
        IdentityKind::IoSurface,
        IdentityKind::AHardwareBufferId,
        IdentityKind::AHardwareBufferPtr,
        IdentityKind::HostPtr,
        IdentityKind::Pbo,
    ] {
        assert_ne!(
            BufferIdentity::derived(k, 1).id(),
            BufferIdentity::derived(k, 2).id(),
            "{k:?}: different keys must give different ids"
        );
    }
}
