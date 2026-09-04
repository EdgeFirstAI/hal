// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! DXGI adapter selection: parsing the adapter environment variables,
//! enumerating the DXGI adapters on the host, and resolving a selection to
//! a concrete adapter LUID.
//!
//! Moved from `crates/image/src/gl/platform/windows.rs`, where the same
//! logic picked ANGLE's D3D11 adapter under a single env var. This crate
//! reads two names (see [`read_adapter_env`]) and hands callers a live
//! [`DxgiAdapter`] instead of a bare LUID, since D3D11 device creation needs
//! the `IDXGIAdapter1` itself.

use super::com;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, DXGI_ADAPTER_DESC1,
    DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND,
};

/// Primary adapter-selection env var.
pub const ADAPTER_ENV: &str = "EDGEFIRST_D3D11_ADAPTER";
/// Legacy name from the ANGLE-only code path; still read for compatibility.
pub const ADAPTER_ENV_ALIAS: &str = "EDGEFIRST_ANGLE_ADAPTER";

/// Which D3D11 adapter the tensor crate should create its device on.
///
/// Parsed from [`ADAPTER_ENV`] / [`ADAPTER_ENV_ALIAS`] by
/// [`parse_adapter_selection`]:
///
/// | Value | Meaning |
/// |---|---|
/// | unset / `hardware` | the default hardware adapter (DXGI adapter 0) |
/// | `warp` | Microsoft Basic Render Driver (software) |
/// | `discrete` | the non-software adapter with the most dedicated video memory |
/// | `<high>:<low>` | an explicit adapter LUID (decimal or `0x` hex) |
/// | anything else | case-insensitive substring of the adapter description (e.g. `RTX 3070`) |
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterSelection {
    Hardware,
    Warp,
    Discrete,
    Luid { high: i32, low: u32 },
    Match(String),
}

fn parse_int(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()
    } else {
        s.parse().ok()
    }
}

/// Parse an adapter-selection env var value. Pure.
pub fn parse_adapter_selection(raw: Option<&str>) -> AdapterSelection {
    let raw = raw.map(str::trim).unwrap_or("");
    match raw.to_ascii_lowercase().as_str() {
        "" | "hardware" | "hw" | "default" => AdapterSelection::Hardware,
        "warp" | "software" => AdapterSelection::Warp,
        "discrete" | "dgpu" => AdapterSelection::Discrete,
        _ => {
            if let Some((h, l)) = raw.split_once(':') {
                if let (Some(high), Some(low)) = (parse_int(h), parse_int(l)) {
                    if let (Ok(high), Ok(low)) = (i32::try_from(high), u32::try_from(low)) {
                        return AdapterSelection::Luid { high, low };
                    }
                }
            }
            AdapterSelection::Match(raw.to_string())
        }
    }
}

/// One DXGI adapter as enumerated by `IDXGIFactory1::EnumAdapters1`.
#[derive(Debug, Clone)]
pub struct DxgiAdapter {
    pub description: String,
    pub luid_high: i32,
    pub luid_low: u32,
    pub dedicated_video_memory: u64,
    pub software: bool,
    pub(crate) adapter: IDXGIAdapter1,
}

#[cfg(test)]
impl DxgiAdapter {
    /// Test fixture: fabricated description/LUID/VRAM/software fields hung
    /// off a real `IDXGIAdapter1`, since the struct no longer has a way to
    /// hold a fake COM pointer. Enumeration always yields at least WARP on
    /// Windows; callers skip the test (rather than fail it) when it errors,
    /// matching `dxgi_enumeration_lists_adapters_or_skips` below.
    fn fake(
        description: &str,
        luid_low: u32,
        dedicated_video_memory: u64,
        software: bool,
    ) -> crate::Result<Self> {
        let adapter = enumerate_dxgi_adapters()?
            .into_iter()
            .next()
            .ok_or_else(|| {
                crate::Error::IoError(std::io::Error::other("DXGI enumerated zero adapters"))
            })?
            .adapter;
        Ok(Self {
            description: description.to_string(),
            luid_high: 0,
            luid_low,
            dedicated_video_memory,
            software,
            adapter,
        })
    }
}

/// Enumerate the DXGI adapters (hardware and software) on this host.
/// Errors when the DXGI factory is unavailable; callers degrade to the
/// default adapter.
pub fn enumerate_dxgi_adapters() -> crate::Result<Vec<DxgiAdapter>> {
    // SAFETY: documented factory creation with no preconditions.
    let factory: IDXGIFactory1 = com::hr("CreateDXGIFactory1", unsafe { CreateDXGIFactory1() })?;
    let mut out = Vec::new();
    for index in 0u32.. {
        // SAFETY: `factory` is live.
        let result = unsafe { factory.EnumAdapters1(index) };
        let adapter = match &result {
            Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
            _ => com::hr("IDXGIFactory1::EnumAdapters1", result)?,
        };
        // SAFETY: `adapter` is live.
        let desc: DXGI_ADAPTER_DESC1 =
            com::hr("IDXGIAdapter1::GetDesc1", unsafe { adapter.GetDesc1() })?;
        let len = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        out.push(DxgiAdapter {
            description: String::from_utf16_lossy(&desc.Description[..len]),
            luid_high: desc.AdapterLuid.HighPart,
            luid_low: desc.AdapterLuid.LowPart,
            dedicated_video_memory: desc.DedicatedVideoMemory as u64,
            software: desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
            adapter,
        });
    }
    Ok(out)
}

/// Turn `Discrete` / `Match` into a concrete LUID (or `Hardware` when
/// nothing matches, with a warning). Pure over the adapter list.
pub fn resolve_adapter(sel: AdapterSelection, adapters: &[DxgiAdapter]) -> AdapterSelection {
    let luid = |a: &DxgiAdapter| AdapterSelection::Luid {
        high: a.luid_high,
        low: a.luid_low,
    };
    match sel {
        AdapterSelection::Discrete => match adapters
            .iter()
            .filter(|a| !a.software)
            .max_by_key(|a| a.dedicated_video_memory)
        {
            Some(a) => luid(a),
            None => {
                log::warn!("{ADAPTER_ENV}=discrete: no hardware adapter enumerated; using the default adapter");
                AdapterSelection::Hardware
            }
        },
        AdapterSelection::Match(needle) => {
            let n = needle.to_ascii_lowercase();
            match adapters
                .iter()
                .find(|a| a.description.to_ascii_lowercase().contains(&n))
            {
                Some(a) => luid(a),
                None => {
                    log::warn!(
                        "{ADAPTER_ENV}={needle:?} matches no DXGI adapter description \
                         ({:?}); using the default adapter",
                        adapters
                            .iter()
                            .map(|a| a.description.as_str())
                            .collect::<Vec<_>>()
                    );
                    AdapterSelection::Hardware
                }
            }
        }
        other => other,
    }
}

/// Both [`ADAPTER_ENV`] and [`ADAPTER_ENV_ALIAS`] are read. When both are
/// set and differ, the D3D11 name wins and the disagreement is logged once
/// so a stale deployment variable cannot silently pick the adapter.
pub(crate) fn read_adapter_env() -> Option<String> {
    let primary = std::env::var(ADAPTER_ENV)
        .ok()
        .filter(|s| !s.trim().is_empty());
    let alias = std::env::var(ADAPTER_ENV_ALIAS)
        .ok()
        .filter(|s| !s.trim().is_empty());
    match (primary, alias) {
        (Some(p), Some(a)) if p.trim() != a.trim() => {
            log::warn!(
                "{ADAPTER_ENV}={p:?} and {ADAPTER_ENV_ALIAS}={a:?} disagree; using {ADAPTER_ENV}"
            );
            Some(p)
        }
        (Some(p), _) => Some(p),
        (None, a) => a,
    }
}

/// The result of [`select_adapter`]: the resolved selection, the matching
/// live adapter when one was picked by LUID, and a human label for the
/// caller to log.
pub(crate) struct Selected {
    pub(crate) selection: AdapterSelection,
    pub(crate) adapter: Option<DxgiAdapter>,
    pub(crate) label: String,
}

/// Read both adapter env vars, enumerate DXGI, log the adapters, and
/// resolve the selection to a concrete adapter.
pub(crate) fn select_adapter() -> Selected {
    let raw = read_adapter_env();
    let sel = parse_adapter_selection(raw.as_deref());
    let adapters = match enumerate_dxgi_adapters() {
        Ok(a) => a,
        Err(e) => {
            log::debug!("DXGI adapter enumeration unavailable ({e}); using the default adapter");
            Vec::new()
        }
    };
    for a in &adapters {
        log::debug!(
            "dxgi adapter {:?} luid={:#x}:{:#x} vram={} MiB software={}",
            a.description,
            a.luid_high,
            a.luid_low,
            a.dedicated_video_memory >> 20,
            a.software
        );
    }
    let resolved = resolve_adapter(sel, &adapters);
    let picked = match &resolved {
        AdapterSelection::Luid { high, low } => adapters
            .iter()
            .find(|a| a.luid_high == *high && a.luid_low == *low)
            .cloned(),
        _ => None,
    };
    let label = match &resolved {
        AdapterSelection::Warp => "WARP (Microsoft Basic Render Driver)".to_string(),
        AdapterSelection::Luid { high, low } => picked
            .as_ref()
            .map(|a| a.description.clone())
            .unwrap_or_else(|| format!("LUID {high:#x}:{low:#x}")),
        _ => adapters
            .iter()
            .find(|a| !a.software)
            .map(|a| format!("{} (default)", a.description))
            .unwrap_or_else(|| "default adapter".to_string()),
    };
    if matches!(resolved, AdapterSelection::Hardware)
        && !adapters.is_empty()
        && adapters.iter().all(|a| a.software)
    {
        log::warn!(
            "no hardware D3D11 adapter is enumerated (only {:?}); device creation will run on the software renderer",
            adapters.iter().map(|a| a.description.as_str()).collect::<Vec<_>>()
        );
    }
    log::info!(
        "D3D11 adapter: {label} ({ADAPTER_ENV}={})",
        raw.as_deref().unwrap_or("unset")
    );
    Selected {
        selection: resolved,
        adapter: picked,
        label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_selection_parses_env_values() {
        assert_eq!(parse_adapter_selection(None), AdapterSelection::Hardware);
        assert_eq!(
            parse_adapter_selection(Some("")),
            AdapterSelection::Hardware
        );
        assert_eq!(
            parse_adapter_selection(Some(" Hardware ")),
            AdapterSelection::Hardware
        );
        assert_eq!(
            parse_adapter_selection(Some("WARP")),
            AdapterSelection::Warp
        );
        assert_eq!(
            parse_adapter_selection(Some("discrete")),
            AdapterSelection::Discrete
        );
        assert_eq!(
            parse_adapter_selection(Some("0x1234:0xabcd")),
            AdapterSelection::Luid {
                high: 0x1234,
                low: 0xabcd
            }
        );
        assert_eq!(
            parse_adapter_selection(Some("0:74901")),
            AdapterSelection::Luid {
                high: 0,
                low: 74901
            }
        );
        assert_eq!(
            parse_adapter_selection(Some("RTX 3070")),
            AdapterSelection::Match("RTX 3070".into())
        );
        // A colon that is not a LUID stays a substring match.
        assert_eq!(
            parse_adapter_selection(Some("Intel: Arc")),
            AdapterSelection::Match("Intel: Arc".into())
        );
    }

    #[test]
    fn resolve_adapter_prefers_largest_hardware_adapter_and_matches_substrings() {
        let fixtures = [
            ("Intel(R) UHD Graphics 630", 1u32, 128u64 << 20, false),
            ("NVIDIA GeForce RTX 3070", 2u32, 8u64 << 30, false),
            ("Microsoft Basic Render Driver", 3u32, 0u64, true),
        ];
        let mut adapters = Vec::new();
        for (description, luid_low, vram, software) in fixtures {
            match DxgiAdapter::fake(description, luid_low, vram, software) {
                Ok(a) => adapters.push(a),
                Err(e) => {
                    eprintln!("DXGI enumeration unavailable — skipping: {e}");
                    return;
                }
            }
        }
        assert_eq!(
            resolve_adapter(AdapterSelection::Discrete, &adapters),
            AdapterSelection::Luid { high: 0, low: 2 }
        );
        assert_eq!(
            resolve_adapter(AdapterSelection::Match("intel".into()), &adapters),
            AdapterSelection::Luid { high: 0, low: 1 }
        );
        assert_eq!(
            resolve_adapter(AdapterSelection::Match("no such gpu".into()), &adapters),
            AdapterSelection::Hardware
        );
        assert_eq!(
            resolve_adapter(AdapterSelection::Warp, &adapters),
            AdapterSelection::Warp
        );
        assert_eq!(
            resolve_adapter(AdapterSelection::Discrete, &adapters[2..]),
            AdapterSelection::Hardware
        );
    }

    /// DXGI enumeration is a system facility; on a host where it is
    /// unavailable (Server Core, a sandbox) the function errors and the
    /// caller degrades, so the assertions apply only when enumeration
    /// succeeds.
    #[test]
    fn dxgi_enumeration_lists_adapters_or_skips() {
        match enumerate_dxgi_adapters() {
            Ok(adapters) => {
                assert!(!adapters.is_empty(), "DXGI enumerated zero adapters");
                for a in &adapters {
                    assert!(!a.description.is_empty());
                }
            }
            Err(e) => eprintln!("DXGI enumeration unavailable — skipping: {e}"),
        }
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn both_adapter_env_names_are_read_and_the_d3d11_name_wins() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var(ADAPTER_ENV);
        std::env::set_var(ADAPTER_ENV_ALIAS, "warp");
        assert_eq!(read_adapter_env().as_deref(), Some("warp"));
        std::env::set_var(ADAPTER_ENV, "discrete");
        assert_eq!(read_adapter_env().as_deref(), Some("discrete"));
        std::env::remove_var(ADAPTER_ENV);
        std::env::remove_var(ADAPTER_ENV_ALIAS);
    }
}
