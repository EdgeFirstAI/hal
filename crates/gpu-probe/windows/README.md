# Windows D3D11 texture interop probe

`d3d11_probe.exe` answers the questions a Windows zero-copy tensor design
depends on: how a `ID3D11Texture2D` reaches ANGLE's OpenGL ES engine, what
CPU access through staging costs, how completion is signalled, how CUDA,
D3D12 and DirectML read the same texture, and what works across devices
and processes. It is the Windows counterpart of `../jetson/`.

It links only the inbox D3D libraries; everything else is loaded at run
time. ANGLE (`libEGL.dll`, `libGLESv2.dll`) is **required**: the probe
exits 2 without it, because most of what it measures is the ANGLE import
path. `cudart64_*.dll`, `nvcuda.dll` and `DirectML.dll` are optional -- a
box without them reports SKIP for those sections. The CUDA toolkit
headers are used for types only.

## Build and run

Requirements: Visual Studio 2022 or newer with the C++ toolset, a Windows
SDK with `d3d11_4.h`, `d3d12.h`, `d3d11on12.h` and `DirectML.h`
(10.0.18362 or newer), a CUDA toolkit for its headers (`CUDA_PATH`, or
`-CudaInclude <dir>`), and the ANGLE package fetched by
`bash scripts/fetch-angle.sh`.

```powershell
pwsh crates/gpu-probe/windows/build.ps1                  # build target/d3d11-probe/d3d11_probe.exe
pwsh crates/gpu-probe/windows/build.ps1 -Run             # build and run on the default hardware adapter
pwsh crates/gpu-probe/windows/build.ps1 -Run -Warp       # build and run on the D3D11 WARP adapter
pwsh crates/gpu-probe/windows/build.ps1 -Run -ProbeArgs @("--only","s1,s5","--iters","20")
```

Options of the executable:

| Option | Meaning |
|---|---|
| `--warp` | ANGLE on the WARP software adapter (the CI shape) |
| `--adapter H:L` | pick a hardware adapter by LUID |
| `--angle DIR` | directory with the ANGLE DLLs (default: `EDGEFIRST_ANGLE_PATH`, then `target/angle/windows-x64/bin`) |
| `--only s1,s5` | run only these sections |
| `--iters N` | timing iterations (default 100) |
| `--debug` | D3D11 debug layer on injected devices, ANGLE debug layers |

Every check prints one `[Sx.y] PASS|FAIL|SKIP|INFO` line; timings print
as `TIME` lines with median, mean and minimum. The exit code is non-zero
when any check failed.

## Sections

| Section | What it settles |
|---|---|
| S0 | Display, device query, extension inventory, staging round trip |
| S1 | Texture import through the pbuffer and EGLImage routes for every format ANGLE accepts, as source and as render target; bind/misc flag matrix; import, bind and convert timings |
| S2 | NV12 native planes, the HAL's combined-plane R8 layout for NV12/NV16/NV24 with the HAL's shader, YUYV/VYUY as RG8, EGLStream |
| S3 | Staging read/write cost by size and format against the PBO baseline, `DO_NOT_WAIT`, staging row pitch table |
| S4 | Same-device ordering, D3D11 fence (event, spin, shared, cross-device), two GL threads plus a non-GL consumer with and without `ID3D11Multithread` |
| S5 | CUDA runtime and driver API: register matrix, external memory, ordering with and without the fence as external semaphore, timings against the PBO path |
| S6 | ANGLE on a HAL-created device (hardware and WARP), behaviour diff, device lifetime, separate allocation device with shared handles |
| S7 | Cross-process: named NT handles, fence by name, legacy KMT handle, CUDA in the child |
| S8 | D3D12 open of the texture and fence, linear copy, DirectML operator, ANGLE D3D11on12, injected D3D11On12 device with `UnwrapUnderlyingResource` |

Running the whole thing with `--warp` is the S9 coverage map. The
results that fed the design are recorded with the planning documents
for the Windows D3D11 texture tensor work.
