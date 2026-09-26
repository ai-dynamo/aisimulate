# oneCCL Communication Benchmarking for Intel XPU

This guide explains how to set up and run oneCCL collective communication benchmarks on Intel XPU (GPU) devices using `collector/network/collect_oneccl_xpu.py` or `collector/network/collect_comm.sh --device xpu`.

## Prerequisites

### 1. Intel oneAPI Packages

The following Intel oneAPI components must be installed:

- **oneCCL** (`intel-oneapi-ccl-devel`) — collective communications library and headers
- **Intel MPI** (`intel-oneapi-mpi`) — MPI runtime (`mpirun`)
- **Intel DPC++ Compiler** (`intel-oneapi-compiler-dpcpp-cpp`) — needed to compile the benchmark binary (`icpx`)

Verify installation:

```bash
dpkg -l | grep -i "intel-oneapi-ccl-devel\|intel-oneapi-mpi\|intel-oneapi-compiler"
```

### 2. Intel GPU Devices

At least 2 Intel GPU (XPU) devices must be available. Verify with:

```bash
xpu-smi discovery
```

### 3. Compile the oneCCL Benchmark Binaries

oneCCL distributes the benchmark as source only, with no pre-compiled binary, so it
must be built once. The source lives in the oneCCL repository under `tests/benchmark/`
as one executable per collective (`allreduce_perf`, `allgather_perf`,
`reduce_scatter_perf`).

Some images ship oneCCL and Intel MPI as runtime components only (for example, as
Python wheels) without the DPC++ compiler or MPI headers. When `icpx` or `mpi.h` is
absent, install them from the Intel oneAPI apt repository, matching the compiler to
the image's DPC++ runtime version to avoid an ABI mismatch:

```bash
curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  > /etc/apt/sources.list.d/oneAPI.list
apt-get update
apt-get install -y \
  "intel-oneapi-compiler-dpcpp-cpp=$(pip show dpcpp-cpp-rt | sed -n 's/^Version: //p')-*" \
  intel-oneapi-mpi-devel
```

Fetch the benchmark source at the tag matching the installed oneCCL library, then
build one binary per collective, linking the oneCCL and MPI installed in the image:

```bash
CCL_ROOT=$(python -c 'import sys; print(sys.prefix)')       # oneCCL install prefix
MPI_ROOT=/opt/intel/oneapi/mpi/latest
CCL_VER=$(pip show oneccl | sed -n 's/^Version: //p')

curl -sSL "https://github.com/uxlfoundation/oneCCL/archive/refs/tags/${CCL_VER}.tar.gz" | tar xz
cd "oneCCL-${CCL_VER}/tests/benchmark"
source /opt/intel/oneapi/compiler/latest/env/vars.sh

# The DPC++ compiler provides its own SYCL headers; placing the oneCCL include
# directory directly on the include path would shadow them, so expose only oneapi/.
mkdir -p /tmp/cclinc && ln -sfn "${CCL_ROOT}/include/oneapi" /tmp/cclinc/oneapi

for op in all_reduce:allreduce_perf allgather:allgather_perf reduce_scatter:reduce_scatter_perf; do
  icpx -std=c++17 -fsycl -I/tmp/cclinc -I"${MPI_ROOT}/include" \
    "${op%%:*}.cpp" common.cpp timer.cpp \
    -L"${CCL_ROOT}/lib" -lccl -L"${MPI_ROOT}/lib" -lmpi -lpthread \
    -o "/usr/local/bin/${op##*:}"
done
```

Verify (each binary accepts `-b minbytes -e maxbytes -g ngpus --iters n --warmup_iters n`):

```bash
allreduce_perf --help
```

### 4. Environment Variables

The following environment variables must be set at runtime (the script sets them automatically):

| Variable | Value | Purpose |
|----------|-------|---------|
| `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK` | `0` | Avoids topology detection errors on PCIe-connected GPUs |
| `FI_PROVIDER` | `tcp` | Uses TCP fabric provider (avoids UCX assertion failures) |
| `I_MPI_OFI_PROVIDER` | `tcp` | Forces Intel MPI to use TCP OFI provider |

## Usage

### Option A: Run All Benchmarks via `collect_comm.sh`

This runs the collective operations (`all_gather`, `reduce_scatter`, `all_reduce`) with both `half` and `int8` data types across all detected GPU counts:

```bash
cd collector/network/
bash collect_comm.sh --device xpu
```

### Option B: Run Individual Operations via `collect_oneccl_xpu.py`

```bash
cd collector/network/

# all_gather with 2 GPUs, half precision, default range (512B to 512MB)
python collect_oneccl_xpu.py --oneccl_op all_gather --dtype half --num_gpus 2

# reduce_scatter with 2 GPUs
python collect_oneccl_xpu.py --oneccl_op reduce_scatter --dtype half --num_gpus 2

# all_reduce with 4 GPUs, custom range
python collect_oneccl_xpu.py --oneccl_op all_reduce --dtype half --num_gpus 4 --range "1024,268435456,2"
```

### CLI Options for `collect_oneccl_xpu.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--oneccl_op`, `-O` | `all_gather` | Collective operation: `all_gather`, `reduce_scatter`, `all_reduce` |
| `--dtype`, `-t` | `half` | Data type: `half` (bf16, 2 bytes), `int8` (1 byte) |
| `--range`, `-r` | `512,536870913,2` | `min_bytes,max_bytes,multiplicative_ratio` |
| `--num_gpus`, `-n` | `2` | Number of GPUs (MPI ranks) |
| `--iters`, `-i` | `100` | Benchmark iterations per message size |
| `--warmup_iters`, `-w` | `20` | Warmup iterations per message size |
