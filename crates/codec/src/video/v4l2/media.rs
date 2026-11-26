// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Whether a video node is a codec, from the media controller graph of its
//! device, without opening the node.
//!
//! A memory-to-memory driver that registers a media controller
//! (`v4l2_m2m_register_media_controller`) publishes a processing entity of
//! function `MEDIA_ENT_F_PROC_VIDEO_ENCODER` or `_DECODER`, data-linked to the
//! I/O entities that the node's interface links to.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

const MEDIA_ENT_F_PROC_VIDEO_ENCODER: u32 = 0x4007;
const MEDIA_ENT_F_PROC_VIDEO_DECODER: u32 = 0x4008;
const MEDIA_LNK_FL_LINK_TYPE: u32 = 0xf << 28;
const MEDIA_LNK_FL_DATA_LINK: u32 = 0;
const MEDIA_LNK_FL_INTERFACE_LINK: u32 = 1 << 28;
/// `_IOWR('|', 0x04, struct media_v2_topology)`.
const MEDIA_IOC_G_TOPOLOGY: libc::c_ulong = 0xc048_7c04;

#[repr(C)]
#[derive(Default)]
struct Topology {
    topology_version: u64,
    num_entities: u32,
    reserved1: u32,
    ptr_entities: u64,
    num_interfaces: u32,
    reserved2: u32,
    ptr_interfaces: u64,
    num_pads: u32,
    reserved3: u32,
    ptr_pads: u64,
    num_links: u32,
    reserved4: u32,
    ptr_links: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawEntity {
    id: u32,
    name: [u8; 64],
    function: u32,
    flags: u32,
    reserved: [u32; 5],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawInterface {
    id: u32,
    intf_type: u32,
    flags: u32,
    reserved: [u32; 9],
    /// `struct media_v2_intf_devnode { major, minor }` in a 16-word union.
    devnode: [u32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawPad {
    id: u32,
    entity_id: u32,
    flags: u32,
    index: u32,
    reserved: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawLink {
    id: u32,
    source_id: u32,
    sink_id: u32,
    flags: u32,
    reserved: [u32; 6],
}

const _: () = {
    assert!(std::mem::size_of::<Topology>() == 72);
    assert!(std::mem::size_of::<RawEntity>() == 96);
    assert!(std::mem::size_of::<RawInterface>() == 112);
    assert!(std::mem::size_of::<RawPad>() == 32);
    assert!(std::mem::size_of::<RawLink>() == 40);
};

/// The parts of a media graph that say what a video node does.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Graph {
    /// `(id, function)` of each entity.
    pub entities: Vec<(u32, u32)>,
    /// `(id, (major, minor))` of each device node interface.
    pub interfaces: Vec<(u32, (u32, u32))>,
    /// `(id, entity)` of each pad.
    pub pads: Vec<(u32, u32)>,
    /// `(source, sink, flags)` of each link: pad ids for a data link, an
    /// interface and an entity for an interface link.
    pub links: Vec<(u32, u32, u32)>,
}

impl Graph {
    /// Whether the device node `devnode` is the interface of a video encoder
    /// or decoder: an entity it links to, or one data-linked to that entity,
    /// has a codec function.
    pub fn is_codec(&self, devnode: (u32, u32)) -> bool {
        let link_type = |flags: u32| flags & MEDIA_LNK_FL_LINK_TYPE;
        let interfaces: Vec<u32> = self
            .interfaces
            .iter()
            .filter(|&&(_, dev)| dev == devnode)
            .map(|&(id, _)| id)
            .collect();
        let io: Vec<u32> = self
            .links
            .iter()
            .filter(|&&(src, _, flags)| {
                link_type(flags) == MEDIA_LNK_FL_INTERFACE_LINK && interfaces.contains(&src)
            })
            .map(|&(_, entity, _)| entity)
            .collect();
        let entity_of = |pad: u32| {
            self.pads
                .iter()
                .find(|&&(id, _)| id == pad)
                .map(|&(_, e)| e)
        };
        let mut near = io.clone();
        for &(src, sink, flags) in &self.links {
            if link_type(flags) != MEDIA_LNK_FL_DATA_LINK {
                continue;
            }
            if let (Some(a), Some(b)) = (entity_of(src), entity_of(sink)) {
                if io.contains(&a) {
                    near.push(b);
                }
                if io.contains(&b) {
                    near.push(a);
                }
            }
        }
        self.entities.iter().any(|&(id, function)| {
            near.contains(&id)
                && matches!(
                    function,
                    MEDIA_ENT_F_PROC_VIDEO_ENCODER | MEDIA_ENT_F_PROC_VIDEO_DECODER
                )
        })
    }
}

/// Reads the graph of the media device at `path`.
fn read_graph(path: &Path) -> std::io::Result<Graph> {
    let file = File::open(path)?;
    let fd = file.as_raw_fd();
    let mut topo = Topology::default();
    // SAFETY: `topo` is a valid media_v2_topology with null arrays, which
    // asks only for the counts.
    if unsafe { libc::ioctl(fd, MEDIA_IOC_G_TOPOLOGY as _, &mut topo) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut entities = vec![
        RawEntity {
            id: 0,
            name: [0; 64],
            function: 0,
            flags: 0,
            reserved: [0; 5],
        };
        topo.num_entities as usize
    ];
    let mut interfaces = vec![
        RawInterface {
            id: 0,
            intf_type: 0,
            flags: 0,
            reserved: [0; 9],
            devnode: [0; 16],
        };
        topo.num_interfaces as usize
    ];
    let mut pads = vec![RawPad::default(); topo.num_pads as usize];
    let mut links = vec![RawLink::default(); topo.num_links as usize];
    topo.ptr_entities = entities.as_mut_ptr() as u64;
    topo.ptr_interfaces = interfaces.as_mut_ptr() as u64;
    topo.ptr_pads = pads.as_mut_ptr() as u64;
    topo.ptr_links = links.as_mut_ptr() as u64;
    // SAFETY: each array holds the count the kernel reported and outlives
    // the call; the kernel writes at most that many elements.
    if unsafe { libc::ioctl(fd, MEDIA_IOC_G_TOPOLOGY as _, &mut topo) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let n = |count: u32, len: usize| (count as usize).min(len);
    Ok(Graph {
        entities: entities[..n(topo.num_entities, entities.len())]
            .iter()
            .map(|e| (e.id, e.function))
            .collect(),
        interfaces: interfaces[..n(topo.num_interfaces, interfaces.len())]
            .iter()
            .map(|i| (i.id, (i.devnode[0], i.devnode[1])))
            .collect(),
        pads: pads[..n(topo.num_pads, pads.len())]
            .iter()
            .map(|p| (p.id, p.entity_id))
            .collect(),
        links: links[..n(topo.num_links, links.len())]
            .iter()
            .map(|l| (l.source_id, l.sink_id, l.flags))
            .collect(),
    })
}

/// `major:minor` from a sysfs `dev` file's text.
fn parse_devnode(text: &str) -> Option<(u32, u32)> {
    let (major, minor) = text.trim().split_once(':')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Whether the video node `path` (`/dev/videoN`) is a codec, as the media
/// controller of its device describes it. `false` when its device has no
/// media controller or the graph cannot be read.
pub(crate) fn is_codec_node(path: &Path) -> bool {
    is_codec_node_in(Path::new("/sys"), Path::new("/dev"), path)
}

fn is_codec_node_in(sys: &Path, dev: &Path, path: &Path) -> bool {
    let Some(name) = path.file_name() else {
        return false;
    };
    let node = sys.join("class/video4linux").join(name);
    let Some(devnode) = std::fs::read_to_string(node.join("dev"))
        .ok()
        .and_then(|t| parse_devnode(&t))
    else {
        return false;
    };
    let Ok(device) = std::fs::canonicalize(node.join("device")) else {
        return false;
    };
    media_devices(sys, &device)
        .into_iter()
        .filter_map(|media| read_graph(&dev.join(media)).ok())
        .any(|graph| graph.is_codec(devnode))
}

/// The names (`mediaN`) of the media devices registered on `device`.
fn media_devices(sys: &Path, device: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(sys.join("bus/media/devices")) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| {
            std::fs::canonicalize(e.path())
                .ok()
                .and_then(|p| p.parent().map(Path::to_path_buf))
                .is_some_and(|parent| parent == device)
        })
        .map(|e| PathBuf::from(e.file_name()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const IO_V4L: u32 = 0x0001_0001;
    const CAM_SENSOR: u32 = 0x0002_0001;

    /// The graph `v4l2_m2m_register_media_controller` builds for one node:
    /// source entity 1, processing entity 3, sink entity 5, and the node's
    /// interface 8 linked to the source and the sink.
    fn m2m(function: u32, devnode: (u32, u32)) -> Graph {
        Graph {
            entities: vec![(1, IO_V4L), (3, function), (5, IO_V4L)],
            interfaces: vec![(8, devnode)],
            pads: vec![(2, 1), (4, 3), (6, 3), (7, 5)],
            links: vec![
                (2, 4, MEDIA_LNK_FL_DATA_LINK | 1),
                (6, 7, MEDIA_LNK_FL_DATA_LINK | 1),
                (8, 1, MEDIA_LNK_FL_INTERFACE_LINK | 1),
                (8, 5, MEDIA_LNK_FL_INTERFACE_LINK | 1),
            ],
        }
    }

    #[test]
    fn a_memory_to_memory_codec_node_is_a_codec() {
        assert!(m2m(MEDIA_ENT_F_PROC_VIDEO_DECODER, (81, 5)).is_codec((81, 5)));
        assert!(m2m(MEDIA_ENT_F_PROC_VIDEO_ENCODER, (81, 6)).is_codec((81, 6)));
    }

    /// The graph `msm_vidc` publishes on the IQ-9075: a decoder at 81:0 and
    /// an encoder at 81:1.
    #[test]
    fn both_msm_vidc_nodes_are_codecs() {
        let g = Graph {
            entities: vec![
                (1, IO_V4L),
                (3, MEDIA_ENT_F_PROC_VIDEO_DECODER),
                (6, IO_V4L),
                (15, IO_V4L),
                (17, MEDIA_ENT_F_PROC_VIDEO_ENCODER),
                (20, IO_V4L),
            ],
            interfaces: vec![(50_331_660, (81, 0)), (50_331_674, (81, 1))],
            pads: vec![
                (16_777_218, 1),
                (16_777_220, 3),
                (16_777_221, 3),
                (16_777_223, 6),
                (16_777_232, 15),
                (16_777_234, 17),
                (16_777_235, 17),
                (16_777_237, 20),
            ],
            links: vec![
                (16_777_218, 16_777_220, 0x3),
                (16_777_221, 16_777_223, 0x3),
                (50_331_660, 1, 0x1000_0003),
                (50_331_660, 6, 0x1000_0003),
                (16_777_232, 16_777_234, 0x3),
                (16_777_235, 16_777_237, 0x3),
                (50_331_674, 15, 0x1000_0003),
                (50_331_674, 20, 0x1000_0003),
            ],
        };
        assert!(g.is_codec((81, 0)));
        assert!(g.is_codec((81, 1)));
        assert!(!g.is_codec((81, 2)));
    }

    #[test]
    fn another_node_of_the_same_device_is_not() {
        assert!(!m2m(MEDIA_ENT_F_PROC_VIDEO_DECODER, (81, 5)).is_codec((81, 7)));
    }

    #[test]
    fn a_camera_or_a_scaler_is_not_a_codec() {
        // A capture node fed by a sensor.
        let camera = Graph {
            entities: vec![(1, CAM_SENSOR), (3, IO_V4L)],
            interfaces: vec![(5, (81, 0))],
            pads: vec![(2, 1), (4, 3)],
            links: vec![
                (2, 4, MEDIA_LNK_FL_DATA_LINK | 1),
                (5, 3, MEDIA_LNK_FL_INTERFACE_LINK | 1),
            ],
        };
        assert!(!camera.is_codec((81, 0)));
        // A memory-to-memory scaler.
        assert!(!m2m(0x4005, (81, 1)).is_codec((81, 1)));
    }

    #[test]
    fn a_codec_entity_two_links_away_does_not_count() {
        let mut g = m2m(IO_V4L, (81, 2));
        g.entities.push((9, MEDIA_ENT_F_PROC_VIDEO_DECODER));
        g.pads.push((10, 9));
        g.links.push((4, 10, MEDIA_LNK_FL_DATA_LINK | 1));
        assert!(!g.is_codec((81, 2)));
    }

    #[test]
    fn sysfs_device_numbers_parse() {
        assert_eq!(parse_devnode("81:16\n"), Some((81, 16)));
        assert_eq!(parse_devnode("81"), None);
        assert_eq!(parse_devnode("a:b"), None);
    }

    #[test]
    fn a_node_without_a_media_device_is_not_a_codec() {
        let dir = std::env::temp_dir().join(format!("ef-media-no-device-{}", std::process::id()));
        let node = dir.join("class/video4linux/video3");
        let device = dir.join("devices/platform/vpu");
        std::fs::create_dir_all(&node).unwrap();
        std::fs::create_dir_all(&device).unwrap();
        std::fs::create_dir_all(dir.join("bus/media/devices")).unwrap();
        std::fs::write(node.join("dev"), "81:3\n").unwrap();
        std::os::unix::fs::symlink(&device, node.join("device")).unwrap();
        assert!(media_devices(&dir, &std::fs::canonicalize(&device).unwrap()).is_empty());
        assert!(!is_codec_node_in(&dir, &dir, Path::new("/dev/video3")));
        assert!(!is_codec_node_in(&dir, &dir, Path::new("/dev/video4")));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
