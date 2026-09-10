// Standalone CUDA Runtime helper. Protocol and provenance checks live in native.py.
// Build for AGX Orin: nvcc -O2 -std=c++17 -arch=sm_87 inflate.cu -o inflate_cuda
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kMaxCells = 16384;
constexpr int kMaxOffsets = 17 * 17;
constexpr std::size_t kMaxRequestBytes = 128 * 1024;
static_assert(sizeof(int) == 4 && sizeof(int2) == 8, "Protocol requires int32");

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T> class DeviceBuffer {
public:
    T* data = nullptr;
    const std::size_t bytes;
    explicit DeviceBuffer(std::size_t count) : bytes(count * sizeof(T)) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

class Event {
public:
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreate(&value), "cudaEventCreate"); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

struct Request {
    int width, height, offset_count, device, repeats;
    std::vector<int> cells, border;
    std::vector<int2> offsets;
};

int integer(std::istream& input, long long low, long long high, const char* field) {
    long long value;
    if (!(input >> value) || value < low || value > high) {
        throw std::runtime_error(std::string("Invalid or missing ") + field);
    }
    return static_cast<int>(value);
}

Request read_request() {
    std::string text;
    char ch;
    while (std::cin.get(ch)) {
        if (text.size() >= kMaxRequestBytes) throw std::runtime_error("Request exceeds 128 KiB");
        text.push_back(ch);
    }
    if (!std::cin.eof()) throw std::runtime_error("Failed to read stdin");
    std::istringstream input(text);
    input.imbue(std::locale::classic());
    std::string protocol;
    if (!(input >> protocol) || protocol != "MARS_INFLATE_V1") {
        throw std::runtime_error("Expected MARS_INFLATE_V1 protocol");
    }
    Request r;
    r.width = integer(input, 1, kMaxCells, "width");
    r.height = integer(input, 1, kMaxCells, "height");
    const long long count = static_cast<long long>(r.width) * r.height;
    if (count > kMaxCells) throw std::runtime_error("width * height exceeds 16384");
    r.offset_count = integer(input, 0, kMaxOffsets, "offset count");
    r.device = integer(input, 0, INT32_MAX, "device ordinal");
    r.repeats = integer(input, 1, 20, "repeats");
    r.cells.resize(count);
    r.border.resize(count);
    r.offsets.resize(r.offset_count);
    for (int& value : r.cells) value = integer(input, -1, 1, "cell (-1/0/1)");
    for (int2& offset : r.offsets) {
        offset.x = integer(input, -8, 8, "offset dx");
        offset.y = integer(input, -8, 8, "offset dy");
    }
    for (int& value : r.border) value = integer(input, 0, 1, "border flag");
    input >> std::ws;
    if (!input.eof()) throw std::runtime_error("Unexpected trailing protocol data");
    return r;
}

__global__ void inflate_kernel(const int* cells, const int2* offsets,
                               const int* border, int* blocked,
                               int width, int height, int offset_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) return;
    const int x = index % width;
    const int y = index / width;
    int value = border[index];
    for (int j = 0; !value && j < offset_count; ++j) {
        const int nx = x + offsets[j].x;
        const int ny = y + offsets[j].y;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height && cells[ny * width + nx] != 0) {
            value = 1;
        }
    }
    blocked[index] = value;
}

// Independent scatter reference is verification only; it never supplies output.
// The CUDA kernel gathers neighbors; this projects nonzero source cells backwards.
std::vector<int> reference(const Request& r) {
    std::vector<int> expected = r.border;
    for (int source = 0; source < static_cast<int>(r.cells.size()); ++source) {
        if (r.cells[source] == 0) continue;
        for (const int2& offset : r.offsets) {
            const int tx = source % r.width - offset.x;
            const int ty = source / r.width - offset.y;
            if (tx >= 0 && tx < r.width && ty >= 0 && ty < r.height) expected[ty * r.width + tx] = 1;
        }
    }
    return expected;
}

std::string quoted(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\' << ch;
        else if (ch < 0x20 || ch >= 0x7f) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
        } else out << ch;
    }
    out << '"';
    return out.str();
}

template <typename T> void array(std::ostream& out, const std::vector<T>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void run(const Request& r) {
    int device_count = 0, runtime_version = 0, driver_version = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount (check NVIDIA driver/device access)");
    if (device_count <= r.device) throw std::runtime_error("Requested CUDA device is unavailable; no CPU fallback");
    check(cudaSetDevice(r.device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, r.device), "cudaGetDeviceProperties");
    check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

    const auto count = r.cells.size();
    DeviceBuffer<int> cells(count), border(count), blocked(count);
    // cudaMalloc(0) is avoided for a valid empty neighborhood (border-only output).
    DeviceBuffer<int2> offsets(r.offset_count ? r.offset_count : 1);
    check(cudaMemcpy(cells.data, r.cells.data(), cells.bytes, cudaMemcpyHostToDevice), "copy cells H2D");
    check(cudaMemcpy(border.data, r.border.data(), border.bytes, cudaMemcpyHostToDevice), "copy border H2D");
    if (r.offset_count) {
        check(cudaMemcpy(offsets.data, r.offsets.data(), r.offset_count * sizeof(int2), cudaMemcpyHostToDevice), "copy offsets H2D");
    }
    const auto expected = reference(r);
    std::vector<int> output(count, -1);
    const auto launch = [&]() {
        inflate_kernel<<<(count + 255) / 256, 256>>>(cells.data, offsets.data, border.data,
                                                   blocked.data, r.width, r.height, r.offset_count);
        check(cudaGetLastError(), "inflate_kernel launch");
    };
    const auto copy_and_verify = [&]() {
        check(cudaMemcpy(output.data(), blocked.data, blocked.bytes, cudaMemcpyDeviceToHost), "copy blocked D2H");
        if (output != expected) throw std::runtime_error("CUDA inflation output mismatches complete independent reference");
    };
    // Exactly one unmeasured execution, using the same kernel and allocations.
    check(cudaMemset(blocked.data, 0xff, blocked.bytes), "poison warmup output");
    launch();
    check(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize");
    copy_and_verify();

    Event start, stop;
    std::vector<double> event_ms, wall_ms;
    for (int repeat = 0; repeat < r.repeats; ++repeat) {
        check(cudaMemset(blocked.data, 0xff, blocked.bytes), "poison output");
        check(cudaDeviceSynchronize(), "pre-timing cudaDeviceSynchronize");
        const auto wall_start = std::chrono::steady_clock::now();
        check(cudaEventRecord(start.value), "cudaEventRecord start");
        launch();
        check(cudaEventRecord(stop.value), "cudaEventRecord stop");
        check(cudaEventSynchronize(stop.value), "cudaEventSynchronize stop");
        check(cudaDeviceSynchronize(), "timed cudaDeviceSynchronize");
        const auto wall_stop = std::chrono::steady_clock::now();
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, start.value, stop.value), "cudaEventElapsedTime");
        const double wall = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
        if (!std::isfinite(elapsed) || elapsed <= 0 || !std::isfinite(wall) || wall <= 0) {
            throw std::runtime_error("CUDA timing evidence is non-positive or non-finite; refusing fabricated timing");
        }
        event_ms.push_back(elapsed);
        wall_ms.push_back(wall);
        // Copies, poisoning, allocations and reference checks are outside both timings.
        copy_and_verify();
    }

    const std::string device = "cuda:" + std::to_string(r.device);
    std::ostringstream identity;
    identity << "\"backend\":\"cuda_runtime\",\"device\":" << quoted(device)
             << ",\"device_name\":" << quoted(properties.name)
             << ",\"compute_capability\":[" << properties.major << ',' << properties.minor << ']'
             << ",\"cuda_runtime_version\":" << runtime_version
             << ",\"cuda_driver_version\":" << driver_version;
    // Emit only one complete JSON object, after every CUDA operation/check succeeded.
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << "{\"blocked\":";
    array(out, output);
    out << ",\"gpu_info\":{" << identity.str()
        << ",\"available\":true,\"kernel_execution_verified\":true,\"device_count\":" << device_count
        << "},\"measurement\":{" << identity.str() << ",\"cuda_event_ms\":";
    array(out, event_ms);
    out << ",\"synchronized_wall_ms\":";
    array(out, wall_ms);
    out << ",\"allocated_device_bytes\":" << cells.bytes + border.bytes + blocked.bytes + offsets.bytes
        << ",\"repeats\":" << r.repeats
        << ",\"warmup\":1,\"timing_scope\":\"occupancy_inflation_kernel_only\""
        << ",\"input_device\":" << quoted(device) << ",\"output_device\":" << quoted(device) << "}}";
    std::cout << out.str() << '\n';
}
} // namespace

int main(int argc, char**) {
    try {
        if (argc != 1) throw std::runtime_error("This helper takes a bounded stdin request, no command-line arguments");
        run(read_request());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Native CUDA inflation failed: " << error.what() << '\n';
        return 1;
    }
}
