# Building LLMServingSim Natively on macOS

This guide explains how to build LLMServingSim from source on macOS (Apple Silicon).

## Prerequisites

```bash
# Install Homebrew dependencies
brew install cmake protobuf abseil python@3.13

# Install Python dependencies
pip install pyyaml pyinstrument msgspec
```

## Build Steps

### 1. Clone LLMServingSim

```bash
git clone https://github.com/your-org/LLMServingSim.git
cd LLMServingSim
```

### 2. Initialize Git Submodules

```bash
# Initialize main submodule (astra-sim)
git submodule init
git submodule update

# Initialize nested submodules (chakra, fmt, spdlog, analytical backend)
cd astra-sim
git submodule update --init --recursive
cd ..
```

### 3. Fix C++17 Compatibility Issues

The Chakra code uses deprecated `std::binary_function` (removed in C++17). Patch two files:

**File 1: `astra-sim/extern/graph_frontend/chakra/src/feeder/et_feeder.h`**

```bash
# Around line 40-45, remove std::binary_function inheritance
# Before:
struct CompareNodes : public std::binary_function<
                          std::shared_ptr<ETFeederNode>,
                          std::shared_ptr<ETFeederNode>,
                          bool> {

# After:
struct CompareNodes {
```

**File 2: `astra-sim/extern/graph_frontend/chakra/src/feeder/json_node.h`**

```bash
# Around line 60-65, remove std::binary_function inheritance
# Before:
struct CompareJSONNodesGT
    : public std::binary_function<JSONNode, JSONNode, bool> {

# After:
struct CompareJSONNodesGT {
```

### 4. Fix Abseil Library Linking

The main `astra-sim/CMakeLists.txt` needs to find and link against Abseil libraries (required by Protobuf on macOS).

**Add after line 39 (after Protobuf find_package):**

```cmake
# Find Abseil (required by Protobuf on macOS)
# Homebrew installs Abseil to a versioned directory
if(EXISTS "/opt/homebrew/Cellar/abseil")
    file(GLOB ABSL_ROOT "/opt/homebrew/Cellar/abseil/*/")
    list(GET ABSL_ROOT 0 ABSL_ROOT)  # Get first (and typically only) version
    set(CMAKE_PREFIX_PATH "${ABSL_ROOT}" ${CMAKE_PREFIX_PATH})
endif()
find_package(absl REQUIRED)
```

**Add after line 76 (after Protobuf target_link_libraries):**

```cmake
# Link Abseil libraries (required by Protobuf on macOS)
target_link_libraries(AstraSim PUBLIC
    absl::base
    absl::log
    absl::log_internal_check_op
    absl::log_internal_message
    absl::strings
    absl::synchronization
    absl::status
    absl::statusor
)
```

### 5. Install Chakra

```bash
cd astra-sim/extern/graph_frontend/chakra
pip install .
cd ../../../..
```

### 6. Build ASTRA-Sim Library

```bash
cd astra-sim
mkdir -p build/astra_analytical
cd build/astra_analytical
cmake ../.. -DASTRA_BACKEND=analytical
make -j$(sysctl -n hw.ncpu)
cd ../../..
```

This creates `astra-sim/build/lib/libAstraSim.a`.

### 7. Build Analytical Network Backend

```bash
cd astra-sim/extern/network_backend/analytical
mkdir build
cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
cd ../../../../..
```

This creates two executables:
- `bin/Analytical_Congestion_Unaware`
- `bin/Analytical_Congestion_Aware`

### 8. Copy AnalyticalAstra Binary

LLMServingSim expects the binary at a specific path:

```bash
mkdir -p astra-sim/build/astra_analytical/build/AnalyticalAstra/bin
cp astra-sim/extern/network_backend/analytical/build/bin/Analytical_Congestion_Unaware \
   astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
```

### 9. Verify the Build

```bash
python main.py --help
```

You should see the LLMServingSim help message without errors.

## Running Experiments with the Adapter

From the `sim-to-real-accuracy-validation` repository:

```bash
# Run with native execution (no Docker)
python -m experiment.run --adapters llmservingsim --no-docker --llmservingsim-dir /path/to/LLMServingSim
```

## Troubleshooting

### "Cannot find libprotobuf.dylib"

Ensure Protobuf is installed via Homebrew:
```bash
brew install protobuf
brew link protobuf
```

### "Undefined symbols for absl::log_internal"

Make sure you applied the Abseil linking fix to `astra-sim/CMakeLists.txt` (Step 4) and rebuilt:
```bash
cd astra-sim/build/astra_analytical
cmake ../.. -DASTRA_BACKEND=analytical
make clean
make -j$(sysctl -n hw.ncpu)
```

### "std::binary_function not found"

Apply the C++17 compatibility patches (Step 3) before building.

## Summary of Changes

The native macOS build required three key fixes:
1. **C++17 Compatibility**: Remove deprecated `std::binary_function` from Chakra code
2. **Abseil Linking**: Add CMake configuration to find and link Abseil libraries
3. **Binary Path**: Copy the analytical backend executable to the expected location

These changes enable LLMServingSim to build and run natively on macOS without Docker.
