# FPGA Accelerator Design — ESP / VC707 / Ariane Baremetal

## Project Overview

This workspace contains a compute-intensive workload targeting the **Xilinx VC707 FPGA board** (xc7vx485t) with an **Ariane RISC-V processor**, running in **baremetal** mode (no OS, no libc malloc — uses `aligned_malloc`). Your task is to **analyze the workload in this workspace**, **design hardware accelerators** and their **software drivers** to speed up its execution, then **integrate them into the existing software**. Workloads vary across testcases — some are ML inference (autoencoder, transformer, CNN), others are classical signal / image processing (image fusion, motion imagery, etc.). The exact workload and which operations dominate runtime are **for you to discover** by reading the source files.

## File Structure

### Required Accelerator Directory Layout (strict)

Every accelerator you build MUST live at exactly this path inside the workspace:

```
<name>_workspace/
├── <name>_sysc_catapult/
│   ├── <name>_sysc_catapult.xml
│   ├── src/<name>.cpp
│   ├── inc/
│   │   ├── <name>.hpp
│   │   ├── <name>_specs.hpp
│   │   ├── <name>_conf_info.hpp
│   │   └── <name>_data_types.hpp
│   ├── tb/
│   │   ├── testbench.cpp
│   │   └── testbench.hpp
│   ├── hls/                       ← validator runs `make` here; MUST contain:
│   │   ├── Makefile               ← REQUIRED — defines bev-sim, hls, install-force targets
│   │   ├── build_prj.tcl
│   │   ├── build_prj_top.tcl
│   │   ├── common.tcl
│   │   └── rtl_sim.tcl
│   └── hls-work-virtex7/          (build output — can start empty, populated by `make hls`)
└── <name>_sw_driver/
    ├── <name>_cfg.h
    └── <name>_driver.h
```

The outer `<name>_workspace/` wrapper is **required**. The `hls/` directory must contain the **full set of build scripts** copied from `dummy_workspace/dummy_sysc_catapult/hls/` — an empty `hls/` (even with `.gitkeep`) will satisfy the existence check but fail at `make bev-sim-clean` with "No rule to make target". `check_workspace.sh` reads accelerator names from `esp_xilinx-vc707-xc7vx485t_defconfig` and runs `make bev-sim && make hls && make install-force` inside `<name>_workspace/<name>_sysc_catapult/hls/`. Mirror the reference `dummy_workspace/` exactly — the simplest way is `cp -r dummy_workspace <name>_workspace` followed by a rename pass (see Step 2).

### Software Files (Baremetal C)

| File | Description |
|------|-------------|
| `systest.c` | Main entry point. Loads data, allocates buffers, runs the workload on test samples, measures execution time. Contains the `USE_ACCELERATOR` toggle. |
| `<workload>.h` (and optionally `<workload>.c`) | Core computation invoked from `systest.c`. For ML workloads this is the forward pass; for signal / image-processing workloads it is the main algorithm. Exact filename depends on the workload — inspect the workspace to find it. |
| `nn_ops.h` *(ML workloads only, if present)* | Shared neural-network operation primitives used by the forward pass. Contents depend on the model; inspect to see which ops (linear, conv, normalization, activation, etc.) exist. Non-ML workloads (image processing, etc.) will not have this file; their ops live directly inside `<workload>.c`. |
| `tensor.h` *(ML workloads only, if present)* | Tensor struct definition, model loading from memory-mapped binary, memory allocation utilities. Non-ML workloads may omit this file and load raw input / gold-output buffers directly. |
| `accel_drivers.h` | Accelerator driver registry — single include point for all accelerator drivers. When adding a new accelerator: (1) `#include` its driver header, (2) add init call in `accel_init_all()`, (3) add cleanup call in `accel_cleanup_all()`. |
| `accel_common.h` | Shared accelerator infrastructure: DMA scatter/gather macros, shared memory allocation (`accel_shared_mem`, 512KB), fixed-point conversion (`float2fixed`/`fixed2float`). Included once by `accel_drivers.h`. |

### Accelerator Workspace (`dummy_workspace/`)

This directory contains a **structural reference accelerator** that demonstrates the shape of an ESP accelerator workspace: HLS kernel source, testbench, XML descriptor, and baremetal software driver. The reference kernel is intentionally an **identity pass-through** (it copies input A to output O; input B is loaded but ignored) and performs no useful computation on its own. **You must create your own accelerator workspace(s)** by copying and adapting this reference. The name `dummy` was chosen deliberately so that a plain `sed 's/dummy/myaccel/g'` rename cannot collide with any matchlib or Catapult header path.

| Path | Description |
|------|-------------|
| `dummy_sysc_catapult/` | Reference HLS accelerator (Catapult HLS, SystemC). **Do NOT use this directly** — copy/adapt it. |
| `dummy_sysc_catapult/dummy_sysc_catapult.xml` | Accelerator descriptor: device name, device ID (`200` = `0x200`), register parameters. **Each accelerator must have a unique device_id.** |
| `dummy_sysc_catapult/src/dummy.cpp` | HLS kernel: `config()` → `load()` → `compute()` → `store()` pipeline. `load()` reads A,B via DMA into PLMs; `compute()` is the identity placeholder; `store()` writes O via DMA. |
| `dummy_sysc_catapult/inc/` | Headers: `dummy_specs.hpp` (PLM sizes, DMA config), `dummy_conf_info.hpp` (register params struct), `dummy_data_types.hpp`, `dummy.hpp` (top-level module). |
| `dummy_sysc_catapult/tb/` | SystemC testbench: `testbench.cpp` generates random data, invokes accelerator, verifies against a golden model. |
| `dummy_sysc_catapult/hls-work-virtex7/` | HLS build directory. |
| `dummy_sw_driver/dummy_driver.h` | Baremetal software driver: `dummy_accel_init()` (probe device), `dummy_accel_exec()` (load data → configure regs → start → poll → read results), `dummy_accel_cleanup()`. |
| `dummy_sw_driver/dummy_cfg.h` | Device ID, register offsets (derived from XML), fixed-point bit-width config. |

### Hardware Design Files

| File | Description |
|------|-------------|
| `top.vhd` | VHDL top-level design for the ESP SoC on VC707. |
| `testbench.vhd` | VHDL simulation testbench. |
| `Makefile` | ESP build system for the SoC and software. |
| `esp_xilinx-vc707-xc7vx485t_defconfig` | ESP SoC platform configuration. Defines the NoC grid dimensions, tile layout, and accelerator placement. See **SoC Tile Configuration** below. |

### Data Files

| File | Description |
|------|-------------|
| `<workload>.bin` *(ML workloads)* | Pre-built model binary. Format: magic `0x4145544E` + version + tensors (name, shape, float data). Non-ML workloads may have **no** model binary at all — they load only test input + gold-reference output. |
| `model_files/` *(or `data_files/`)* | Source binaries: model weights (if applicable), test input data, and test labels / gold-reference outputs. Inspect the directory to see the exact files and their sizes. |
| `load_model_ariane.sh` *(or `load_data_ariane.sh`)* | Script to load binary data into FPGA memory via `esplink`. |
| `prepare_data.py` | Python utility to prepare / align the workload's binaries and test data. |

## Accelerator Design Flow
Steps 1–5 in order. Each step depends on the previous one; do not skip ahead or reorder.
### Step 1: Analyze the workload

Start from `systest.c` and trace the execution call graph into the workload-specific header/source (and any shared ops file, if present). Identify the core computations, their sizes, and which operations dominate runtime — those are your acceleration candidates.

### Step 2: Create your accelerator workspace

For each accelerator you want to build, **duplicate the entire `dummy_workspace/` directory** and replace the `dummy`/`DUMMY` name with your accelerator's name (lowercase and UPPERCASE forms). Replace `myaccel` / `MYACCEL` below with your chosen name:

> **Common mistake — DO NOT hand-create the directory tree.** You MUST start with `cp -r dummy_workspace <name>_workspace` (step 1 below). This copies the `hls/` build Makefile and TCL scripts along with all other source files. **An empty `hls/` directory — or one with only `.gitkeep` — will fail** `make bev-sim-clean` with `No rule to make target 'bev-sim-clean'`. Do NOT manually create `<name>_workspace/` or any of its subdirectories, and do NOT place `<name>_sysc_catapult/` directly at the workspace top level without the outer `<name>_workspace/` wrapper. After step 1, verify with: `ls <name>_workspace/<name>_sysc_catapult/hls/Makefile` — if that file is missing, redo the copy.

1. **Copy the workspace and replace `dummy`/`DUMMY` strings inside files** (case-preserving):
```bash
cp -r dummy_workspace myaccel_workspace
cd myaccel_workspace
find . -type f -exec sed -i 's/dummy/myaccel/g; s/DUMMY/MYACCEL/g' {} +
```

2. **Rename files and directories** whose names contain `dummy` — only substitute the basename, not the full path (use `-depth` so inner paths are renamed before their parents):
```bash
find . -depth -name '*dummy*' | while read f; do
  d=$(dirname "$f"); b=$(basename "$f")
  mv "$f" "$d/$(echo "$b" | sed 's/dummy/myaccel/g')"
done
```

3. **Assign a unique device ID** in both the XML file and the driver config header. Pick a value from this project's reserved range `0x201`–`0x27F` (the reference `dummy` already occupies `0x200`). See **Device ID Allocation** below for the full convention.
```bash
sed -i 's/device_id="200"/device_id="201"/' \
  myaccel_sysc_catapult/myaccel_sysc_catapult.xml
sed -i 's/0x200/0x201/g' myaccel_sw_driver/myaccel_cfg.h
```

After these three steps, your new workspace is a functional (but still identity-behavior) clone of the `dummy` reference. You can now redesign the kernel.

### Step 3: Design the accelerator kernel

1. Modify `src/<accel>.cpp` — implement your kernel in the `compute()` function. The `load()` and `store()` functions handle DMA data movement.
2. Modify `inc/<accel>_specs.hpp` — adjust PLM sizes (e.g. `A_PLM_IN_WORD`, `B_PLM_IN_WORD`, `O_PLM_OUT_WORD`) and `MEM_SIZE` to match your data dimensions.
3. If you change the accelerator's configuration parameters, update `inc/<accel>_conf_info.hpp` **and** the XML `<param>` list (register offsets are derived from the XML).
4. Update the testbench (`tb/testbench.cpp`, `tb/testbench.hpp`) to match your accelerator's interface.

### Step 4: Write the software driver

Adapt the driver (from `dummy_sw_driver/dummy_driver.h`) so that it:
1. Probes for the accelerator via `probe(&devs, VENDOR_SLD, DEV_ID, DEV_NAME)`
2. Provides an `exec()` function that:
   - Converts float inputs to fixed-point and writes to `accel_shared_mem`
   - Configures accelerator registers via `iowrite32()`
   - Starts accelerator and polls `STATUS_REG` for completion
   - Converts fixed-point outputs back to float
3. The driver must use the shared memory from `accel_common.h` (do not allocate separate DMA buffers)

### Step 5: Integrate into software

1. Add `#include` for your driver in `accel_drivers.h`
2. Add init/cleanup calls in `accel_init_all()` / `accel_cleanup_all()`
3. Modify the workload's core computation (the file called from `systest.c`) to invoke your accelerator's `exec()` function for the appropriate operations (guarded by `#ifdef USE_ACCELERATOR`)
4. Enable `USE_ACCELERATOR` in `systest.c`
5. Register your accelerator(s) in `esp_xilinx-vc707-xc7vx485t_defconfig` (see **SoC Tile Configuration** below)

## Key ESP Platform Conventions

### SoC Tile Configuration (`esp_defconfig`)

The ESP SoC is organized as a **NoC (Network-on-Chip) grid** of tiles. Each tile can be a CPU, memory, I/O, or accelerator. The tile layout is defined in `esp_xilinx-vc707-xc7vx485t_defconfig`.

The default configuration is a **2×2 grid**:
```
CONFIG_NOC_ROWS = 2
CONFIG_NOC_COLS = 2

TILE_0_0 = 0 mem mem
TILE_0_1 = 1 cpu cpu
TILE_1_0 = 2 empty empty
TILE_1_1 = 3 misc IO
```

To register an accelerator, replace an `empty` tile with an `acc` entry:
```
TILE_1_0 = 2 acc MYACCEL_SYSC_CATAPULT dma64 0 0 sld
```
Format: `TILE_<row>_<col> = <index> acc <ACCEL_NAME_UPPERCASE> dma64 0 0 sld`

If you need **more than one accelerator** and run out of empty tiles, you must **expand the NoC grid** by increasing `CONFIG_NOC_ROWS` and/or `CONFIG_NOC_COLS`, then add new tile entries for the additional slots. For example, a 2×3 grid:
```
CONFIG_NOC_COLS = 3

TILE_0_0 = 0 mem mem
TILE_0_1 = 1 cpu cpu
TILE_0_2 = 2 acc MY_ACCEL_1 dma64 0 0 sld
TILE_1_0 = 3 acc MY_ACCEL_2 dma64 0 0 sld
TILE_1_1 = 4 misc IO
TILE_1_2 = 5 empty empty
```

> **Note**: Expanding the NoC grid increases resource usage. Keep the grid as small as possible while fitting all required tiles.

### Other Conventions

- **Device IDs**: Each accelerator needs a unique ID in its XML file (e.g., `device_id="200"`). The same ID must match in the software driver's `_cfg.h` (as `0x200`). See **Device ID Allocation** below.
- **Register Offsets**: XML `<param>`s are mapped to MMIO register offsets **in reverse order**, starting at `0x40`. For example, if the XML lists three params `[P0, P1, P2]` (in that textual order), then `P2` sits at `0x40`, `P1` at `0x44`, and `P0` at `0x48`. The first-listed param always occupies the highest offset.
- **DMA Layout**: Data is laid out in `accel_shared_mem` as contiguous regions: `[input_A][input_B][output_O]`. Offsets must be aligned to `DMA_WORD_PER_BEAT`.
- **Fixed-point**: The accelerator uses fixed-point arithmetic. `float2fixed(x, WL, IL)` converts float to fixed-point with `WL` total bits and `IL` integer bits (fractional bits = WL - IL). Current config: WL=32, IL=16.
- **BSS is NOT zeroed**: On Ariane baremetal, static variables may contain garbage. Always explicitly initialize. `accel_common_init()` handles this for the shared infra.

### Device ID Allocation

- **Format**: The XML `device_id` attribute is a **hexadecimal** value written **without** the `0x` prefix. `device_id="200"` in XML corresponds to `0x200` in C code. **It is NOT decimal.** For example `device_id="04a"` ↔ `0x04a` in the driver.
- **Valid range**: `0x000` – `0x3FF` (10-bit field; see `devid_t` in the generated `sld_devices.vhd`).
- **Reserved by ESP** (do NOT use): `0x000`–`0x0FF` is densely populated by ESP infrastructure (caches, L2/LLC, TILE_CSR, etc.) and stock accelerators. `0x100` is NVDLA. `0x144` is sinkhorn. `0x222` is svd.
- **This project's reservation**: **`0x200` – `0x27F`**. The reference `dummy` accelerator occupies `0x200`. Assign your new accelerators starting at `0x201`.
- **Collision symptom**: If two accelerators share a device ID, `probe()` may return the wrong device or the SoC build may fail. Keep IDs unique.

## Restrictions

1. You are **NOT allowed** to access (read or write) files **outside the workspace directory**.
2. `dummy_workspace/` is a **reference-only** structural template. You must **not use it as-is** — its kernel is an identity pass-through and cannot perform any useful computation. Create your own accelerator workspace(s) by copying it and redesigning the HLS source, testbench, driver, and configuration to match your design.
3. Each accelerator must have a **unique device ID** in its XML file (`<your_workspace>/<accel_name>/<accel_name>.xml`).

## Evaluation Criteria

1. **Execution time**: Minimize total workload execution time (measured in cycles by `rdcycle64()`).
2. **Resource constraints** (VC707 FPGA):
   - Logic Cells: 485,760
   - Block RAM (Kb): 37,080
   - DSP Slices: 2,800
3. **Correctness**: The accelerated pipeline must **pass** whatever correctness check `systest.c` performs (e.g., output-vs-gold comparison, top-1 match, MSE threshold, etc.). The exact check and its threshold are workload-specific — look at what `systest.c` reports at the end of a run.

## Deliverables

Your job is done when you have completed all of the following:

1. Created accelerator HLS source(s) (new workspace directories with complete `src/`, `inc/`, `tb/`, and XML)
2. Written corresponding software driver(s) (header files with init/exec/cleanup functions)
3. Integrated drivers into `accel_drivers.h` and modified the workload's core computation to use your accelerators
4. Updated `esp_xilinx-vc707-xc7vx485t_defconfig` with accelerator tile entries (expanding NoC grid if needed)
5. Enabled `USE_ACCELERATOR` in `systest.c`
