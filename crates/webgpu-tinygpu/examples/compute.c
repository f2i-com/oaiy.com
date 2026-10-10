// examples/compute.c: a C program against webgpu.h, as any is written, its compute on the TinyGPU card (or the
// server's emulator, with TINYGPU_SOCKET at it):
//
//     cargo build --release -p webgpu-tinygpu
//     cc -I crates/webgpu-tinygpu/include crates/webgpu-tinygpu/examples/compute.c -L target/release -lwebgpu_tinygpu -o compute
//     DYLD_LIBRARY_PATH=target/release ./compute
//
// The adapter, the device and the map are waited for with futures (wgpuInstanceWaitAny), as Dawn's own programs do.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "webgpu.h"

static void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* out, void* unused) {
    (void)unused;
    if (status != WGPURequestAdapterStatus_Success) fprintf(stderr, "no adapter: %.*s\n", (int)message.length, message.data);
    *(WGPUAdapter*)out = adapter;
}

static void on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* out, void* unused) {
    (void)unused;
    if (status != WGPURequestDeviceStatus_Success) fprintf(stderr, "no device: %.*s\n", (int)message.length, message.data);
    *(WGPUDevice*)out = device;
}

static void on_map(WGPUMapAsyncStatus status, WGPUStringView message, void* out, void* unused) {
    (void)message;
    (void)unused;
    *(int*)out = status == WGPUMapAsyncStatus_Success;
}

static void wait_for(WGPUInstance instance, WGPUFuture f) {
    WGPUFutureWaitInfo info = {f, 0};
    wgpuInstanceWaitAny(instance, 1, &info, UINT64_MAX);
}

static const char* SHADER =
    "@group(0) @binding(0) var<storage, read_write> v: array<u32>;\n"
    "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  var x = v[id.x]; var steps = 0u;\n"
    "  while (x != 1u && steps < 1000u) { x = select(x / 2u, 3u * x + 1u, x % 2u == 1u); steps += 1u; }\n"
    "  v[id.x] = steps;\n"
    "}\n";

int main(void) {
    WGPUInstance instance = wgpuCreateInstance(NULL);
    WGPUAdapter adapter = NULL;
    wait_for(instance, wgpuInstanceRequestAdapter(instance, NULL, (WGPURequestAdapterCallbackInfo){.mode = WGPUCallbackMode_WaitAnyOnly, .callback = on_adapter, .userdata1 = &adapter}));
    if (!adapter) return 1;
    WGPUAdapterInfo info = {0};
    wgpuAdapterGetInfo(adapter, &info);
    printf("adapter: %.*s\n", (int)info.device.length, info.device.data);
    wgpuAdapterInfoFreeMembers(info);
    WGPUDevice device = NULL;
    wait_for(instance, wgpuAdapterRequestDevice(adapter, NULL, (WGPURequestDeviceCallbackInfo){.mode = WGPUCallbackMode_WaitAnyOnly, .callback = on_device, .userdata1 = &device}));
    if (!device) return 1;
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    enum { N = 4096 };
    uint32_t* start = malloc(N * 4);
    for (uint32_t i = 0; i < N; i++) start[i] = i + 1;
    WGPUBuffer v = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor){.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, .size = N * 4});
    WGPUBuffer back = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor){.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, .size = N * 4});
    wgpuQueueWriteBuffer(queue, v, 0, start, N * 4);

    WGPUShaderSourceWGSL wgsl = {.chain = {.sType = WGPUSType_ShaderSourceWGSL}, .code = {SHADER, WGPU_STRLEN}};
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &(WGPUShaderModuleDescriptor){.nextInChain = &wgsl.chain});
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &(WGPUComputePipelineDescriptor){.compute = {.module = module, .entryPoint = {"main", WGPU_STRLEN}}});
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    WGPUBindGroupEntry entry = {.binding = 0, .buffer = v, .offset = 0, .size = WGPU_WHOLE_SIZE};
    WGPUBindGroup group = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){.layout = layout, .entryCount = 1, .entries = &entry});

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, N / 64, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, v, 0, back, 0, N * 4);
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(queue, 1, &commands);

    int mapped = 0;
    wait_for(instance, wgpuBufferMapAsync(back, WGPUMapMode_Read, 0, N * 4, (WGPUBufferMapCallbackInfo){.mode = WGPUCallbackMode_WaitAnyOnly, .callback = on_map, .userdata1 = &mapped}));
    const uint32_t* steps = wgpuBufferGetConstMappedRange(back, 0, N * 4);
    int wrong = !mapped || !steps;
    for (uint32_t i = 0; !wrong && i < N; i++) {   // the Collatz steps of i + 1, on the CPU
        uint32_t x = i + 1, s = 0;
        while (x != 1 && s < 1000) x = x % 2 ? 3 * x + 1 : x / 2, s++;
        wrong += steps[i] != s;
    }
    printf("%d Collatz counts: %s (27 takes %u steps)\n", N, wrong ? "WRONG" : "OK", steps ? steps[26] : 0);
    wgpuBufferUnmap(back);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(group);
    wgpuBindGroupLayoutRelease(layout);
    wgpuComputePipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
    wgpuBufferRelease(back);
    wgpuBufferRelease(v);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    free(start);
    return wrong != 0;
}
