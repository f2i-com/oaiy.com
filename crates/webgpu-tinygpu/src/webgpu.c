// webgpu.c: webgpu.h's functions (WebGPU's C API, with wgpu.h's extensions, as wgpu-native's library has them) over
// lib.rs (include/tgw.h). The header's structs are read here, so the compiler lays them out from the header itself.
//
// Each object is counted (AddRef, Release) and holds its device; lib.rs's object goes when the count does. Every
// operation is done as it is asked (the TinyGPU adapter's are), so a future is complete when it is made: its callback
// fires at once for AllowSpontaneous, at the next wgpuInstanceProcessEvents or wgpuDevicePoll for AllowProcessEvents,
// and in wgpuInstanceWaitAny for its future. An error goes to the device's innermost error scope of its kind, else to
// its uncaptured-error callback (else stderr). Compute alone: what is not compute is not here, and a program that
// calls it finds no symbol.
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "webgpu.h"
#include "wgpu.h"
#include "tgw.h"

#define API __attribute__((visibility("default")))
#define OBJECT atomic_uint refs

// the version wgpu-native's library gives, as Python's wgpu (0.32) expects it: 29.0.1.1
#define VERSION ((29u << 24) | (0u << 16) | (1u << 8) | 1u)

// ---- strings ----

static size_t sv_len(WGPUStringView s) {
    if (!s.data) return 0;
    return s.length == WGPU_STRLEN ? strlen(s.data) : s.length;
}

static WGPUStringView sv(const char* s) {
    WGPUStringView v = {s, s ? strlen(s) : 0};
    return v;
}

static char* dup_of(const char* s) {
    size_t n = strlen(s);
    char* d = malloc(n + 1);
    memcpy(d, s, n + 1);
    return d;
}

// ---- objects ----

struct WGPUInstanceImpl { OBJECT; };
struct WGPUAdapterImpl { OBJECT; TgwAdapter* rs; };
struct WGPUDeviceImpl {
    OBJECT;
    TgwDevice* rs;
    WGPUAdapter adapter;
    WGPUQueue queue;   // (not counted: the queue counts the device)
    WGPUUncapturedErrorCallbackInfo uncaptured;
    WGPUDeviceLostCallbackInfo lost;
    int lost_told;
};
struct WGPUQueueImpl { OBJECT; WGPUDevice device; };
struct WGPUBufferImpl {
    OBJECT;
    WGPUDevice device;
    TgwBuffer* rs;
    uint64_t size;
    WGPUBufferUsage usage;
    WGPUBufferMapState state;
    int write;   // mapped for writing (or at creation)
};
struct WGPUShaderModuleImpl { OBJECT; WGPUDevice device; TgwShader* rs; };
struct WGPUBindGroupLayoutImpl { OBJECT; WGPUDevice device; TgwBgl* rs; };
struct WGPUPipelineLayoutImpl { OBJECT; WGPUDevice device; TgwPipelineLayout* rs; };
struct WGPUComputePipelineImpl { OBJECT; WGPUDevice device; TgwPipeline* rs; };
struct WGPUBindGroupImpl { OBJECT; WGPUDevice device; TgwBindGroup* rs; };
struct WGPUCommandEncoderImpl { OBJECT; WGPUDevice device; TgwEncoder* rs; };
struct WGPUComputePassEncoderImpl { OBJECT; WGPUDevice device; TgwPass* rs; };
struct WGPUCommandBufferImpl { OBJECT; WGPUDevice device; TgwCommandBuffer* rs; };

#define NEW(T) ((T)calloc(1, sizeof(*(T)0)))

#define COUNTED(T, FINI)                                                              \
    API void wgpu##T##AddRef(WGPU##T o) {                                            \
        if (o) atomic_fetch_add(&o->refs, 1);                                        \
    }                                                                                 \
    API void wgpu##T##Release(WGPU##T o) {                                           \
        if (o && atomic_fetch_sub(&o->refs, 1) == 1) {                               \
            FINI(o);                                                                  \
            free(o);                                                                  \
        }                                                                             \
    }                                                                                 \
    API void wgpu##T##SetLabel(WGPU##T o, WGPUStringView label) {                    \
        (void)o;                                                                      \
        (void)label;                                                                  \
    }

static void instance_fini(WGPUInstance o) { (void)o; }
static void adapter_fini(WGPUAdapter o) { tgw_adapter_free(o->rs); }
static void device_fini(WGPUDevice o) {
    tgw_device_free(o->rs);
    wgpuAdapterRelease(o->adapter);
}
static void queue_fini(WGPUQueue o) {
    o->device->queue = NULL;
    wgpuDeviceRelease(o->device);
}
#define CHILD_FINI(NAME, T, FREE)  \
    static void NAME(T o) {        \
        FREE(o->rs);               \
        wgpuDeviceRelease(o->device); \
    }
CHILD_FINI(buffer_fini, WGPUBuffer, tgw_buffer_free)
CHILD_FINI(shader_fini, WGPUShaderModule, tgw_shader_free)
CHILD_FINI(bgl_fini, WGPUBindGroupLayout, tgw_bgl_free)
CHILD_FINI(pipeline_layout_fini, WGPUPipelineLayout, tgw_pipeline_layout_free)
CHILD_FINI(pipeline_fini, WGPUComputePipeline, tgw_pipeline_free)
CHILD_FINI(bind_group_fini, WGPUBindGroup, tgw_bind_group_free)
CHILD_FINI(encoder_fini, WGPUCommandEncoder, tgw_encoder_free)
CHILD_FINI(pass_fini, WGPUComputePassEncoder, tgw_pass_free)
CHILD_FINI(command_buffer_fini, WGPUCommandBuffer, tgw_command_buffer_free)

COUNTED(Instance, instance_fini)
COUNTED(Adapter, adapter_fini)
COUNTED(Device, device_fini)
COUNTED(Queue, queue_fini)
COUNTED(Buffer, buffer_fini)
COUNTED(ShaderModule, shader_fini)
COUNTED(BindGroupLayout, bgl_fini)
COUNTED(PipelineLayout, pipeline_layout_fini)
COUNTED(ComputePipeline, pipeline_fini)
COUNTED(BindGroup, bind_group_fini)
COUNTED(CommandEncoder, encoder_fini)
COUNTED(ComputePassEncoder, pass_fini)
COUNTED(CommandBuffer, command_buffer_fini)

// a device's child, holding it
#define CHILD(T, d, RS)                 \
    ({                                  \
        T o_ = NEW(T);                  \
        o_->refs = 1;                   \
        o_->device = (d);               \
        wgpuDeviceAddRef(d);            \
        o_->rs = (RS);                  \
        o_;                             \
    })

// ---- errors ----

static WGPUErrorType error_type(uint32_t kind) {
    return kind == TGW_OUT_OF_MEMORY ? WGPUErrorType_OutOfMemory : kind == TGW_INTERNAL ? WGPUErrorType_Internal : WGPUErrorType_Validation;
}

// the errors no scope took, to the device's callback
static void tell(WGPUDevice d) {
    uint32_t kind = 0;
    char* m;
    while ((m = tgw_device_take_error(d->rs, &kind))) {
        if (d->uncaptured.callback) {
            WGPUDevice self = d;
            d->uncaptured.callback(&self, error_type(kind), sv(m), d->uncaptured.userdata1, d->uncaptured.userdata2);
        } else {
            fprintf(stderr, "webgpu-tinygpu: %s\n", m);
        }
        tgw_string_free(m);
    }
}

// ---- futures ----

typedef enum { EV_ADAPTER, EV_DEVICE, EV_MAP, EV_WORK_DONE, EV_POP_SCOPE, EV_PIPELINE, EV_COMPILATION, EV_LOST } EventKind;

typedef struct Event {
    struct Event* next;
    uint64_t future;
    WGPUCallbackMode mode;
    EventKind kind;
    void (*callback)(void);
    void* userdata1;
    void* userdata2;
    uint32_t status;
    uint32_t type;
    void* object;
    char* message;   // (the event's, freed once told)
} Event;

static pthread_mutex_t events_lock = PTHREAD_MUTEX_INITIALIZER;
static Event* events = NULL;
static Event** events_end = &events;
static _Atomic uint64_t next_future = 1;

static void fire(Event* e) {
    WGPUStringView m = e->message ? sv(e->message) : (WGPUStringView){NULL, 0};
    void *u1 = e->userdata1, *u2 = e->userdata2;
    switch (e->kind) {
    case EV_ADAPTER: ((WGPURequestAdapterCallback)e->callback)(e->status, e->object, m, u1, u2); break;
    case EV_DEVICE: ((WGPURequestDeviceCallback)e->callback)(e->status, e->object, m, u1, u2); break;
    case EV_MAP: ((WGPUBufferMapCallback)e->callback)(e->status, m, u1, u2); break;
    case EV_WORK_DONE: ((WGPUQueueWorkDoneCallback)e->callback)(e->status, m, u1, u2); break;
    case EV_POP_SCOPE: ((WGPUPopErrorScopeCallback)e->callback)(e->status, e->type, m, u1, u2); break;
    case EV_PIPELINE: ((WGPUCreateComputePipelineAsyncCallback)e->callback)(e->status, e->object, m, u1, u2); break;
    case EV_COMPILATION: {
        WGPUCompilationInfo info = {0};
        ((WGPUCompilationInfoCallback)e->callback)(e->status, &info, u1, u2);
        break;
    }
    case EV_LOST: {
        WGPUDevice d = e->object;
        ((WGPUDeviceLostCallback)e->callback)(&d, e->status, m, u1, u2);
        break;
    }
    }
    free(e->message);
    free(e);
}

// a completed operation's callback: told at once, or kept for the instance's events or a wait
static WGPUFuture post(EventKind kind, WGPUCallbackMode mode, void (*callback)(void), void* u1, void* u2, uint32_t status, uint32_t type, void* object, char* message) {
    WGPUFuture f = {atomic_fetch_add(&next_future, 1)};
    if (!callback) {
        free(message);
        return f;
    }
    Event* e = calloc(1, sizeof *e);
    *e = (Event){NULL, f.id, mode, kind, callback, u1, u2, status, type, object, message};
    if (mode == WGPUCallbackMode_AllowSpontaneous) {
        fire(e);
        return f;
    }
    pthread_mutex_lock(&events_lock);
    *events_end = e;
    events_end = &e->next;
    pthread_mutex_unlock(&events_lock);
    return f;
}

// the events that `pick` takes, out of the list in order
static Event* take_events(int (*pick)(Event*, void*), void* arg) {
    Event *taken = NULL, **taken_end = &taken;
    pthread_mutex_lock(&events_lock);
    Event** p = &events;
    while (*p) {
        Event* e = *p;
        if (pick(e, arg)) {
            *p = e->next;
            e->next = NULL;
            *taken_end = e;
            taken_end = &e->next;
        } else {
            p = &e->next;
        }
    }
    events_end = p;
    pthread_mutex_unlock(&events_lock);
    return taken;
}

static int not_wait_only(Event* e, void* arg) {
    (void)arg;
    return e->mode != WGPUCallbackMode_WaitAnyOnly;
}

static int of_future(Event* e, void* arg) { return e->future == *(uint64_t*)arg; }

static int fire_all(Event* e) {
    int n = 0;
    while (e) {
        Event* next = e->next;
        fire(e);
        e = next;
        n++;
    }
    return n;
}

static void process_events(void) { fire_all(take_events(not_wait_only, NULL)); }

// ---- the instance ----

API WGPUInstance wgpuCreateInstance(WGPUInstanceDescriptor const* descriptor) {
    (void)descriptor;
    WGPUInstance i = NEW(WGPUInstance);
    i->refs = 1;
    return i;
}

API void wgpuInstanceProcessEvents(WGPUInstance instance) {
    (void)instance;
    process_events();
}

API WGPUWaitStatus wgpuInstanceWaitAny(WGPUInstance instance, size_t futureCount, WGPUFutureWaitInfo* futures, uint64_t timeoutNS) {
    (void)instance;
    (void)timeoutNS;
    for (size_t i = 0; i < futureCount; i++) {
        uint64_t id = futures[i].future.id;
        fire_all(take_events(of_future, &id));
        futures[i].completed = 1;   // (every operation is done as it is asked)
    }
    return WGPUWaitStatus_Success;
}

static WGPUAdapter open_adapter(char** error) {
    TgwAdapter* rs = tgw_adapter_open(error);
    if (!rs) return NULL;
    WGPUAdapter a = NEW(WGPUAdapter);
    a->refs = 1;
    a->rs = rs;
    return a;
}

API WGPUFuture wgpuInstanceRequestAdapter(WGPUInstance instance, WGPURequestAdapterOptions const* options, WGPURequestAdapterCallbackInfo callbackInfo) {
    (void)instance;
    (void)options;
    char* error = NULL;
    WGPUAdapter a = open_adapter(&error);
    char* message = NULL;
    if (error) {
        message = dup_of(error);
        tgw_string_free(error);
    }
    return post(EV_ADAPTER, callbackInfo.mode, (void (*)(void))callbackInfo.callback, callbackInfo.userdata1, callbackInfo.userdata2,
                a ? WGPURequestAdapterStatus_Success : WGPURequestAdapterStatus_Unavailable, 0, a, message);
}

API size_t wgpuInstanceEnumerateAdapters(WGPUInstance instance, WGPUInstanceEnumerateAdapterOptions const* options, WGPUAdapter* adapters) {
    (void)instance;
    (void)options;
    char* error = NULL;
    WGPUAdapter a = open_adapter(&error);
    tgw_string_free(error);
    if (!a) return 0;
    if (adapters) adapters[0] = a;
    else wgpuAdapterRelease(a);
    return 1;
}

API uint32_t wgpuGetVersion(void) { return VERSION; }
API void wgpuSetLogCallback(WGPULogCallback callback, void* userdata) {
    (void)callback;
    (void)userdata;
}
API void wgpuSetLogLevel(WGPULogLevel level) { (void)level; }
API void wgpuGenerateReport(WGPUInstance instance, WGPUGlobalReport* report) {
    (void)instance;
    memset(report, 0, sizeof *report);
}

// ---- adapter ----

static WGPUStatus info_of(TgwAdapter* rs, WGPUAdapterInfo* info) {
    TgwInfo t;
    memset(&t, 0, sizeof t);
    tgw_adapter_info(rs, &t);
    info->vendor = sv(dup_of(t.vendor));
    info->architecture = sv(dup_of(t.backend == 0 ? "tinygpu" : ""));
    info->device = sv(dup_of(t.name));
    info->description = sv(dup_of(t.driver));
    info->backendType = t.backend == 1 ? WGPUBackendType_Metal : t.backend == 2 ? WGPUBackendType_Vulkan : WGPUBackendType_Null;
    info->adapterType = t.discrete ? WGPUAdapterType_DiscreteGPU : t.backend == 0 ? WGPUAdapterType_CPU : WGPUAdapterType_IntegratedGPU;
    info->vendorID = t.vendor_id;
    info->deviceID = t.device_id;
    info->subgroupMinSize = t.subgroup_min;
    info->subgroupMaxSize = t.subgroup_max;
    return WGPUStatus_Success;
}

API WGPUStatus wgpuAdapterGetInfo(WGPUAdapter adapter, WGPUAdapterInfo* info) { return info_of(adapter->rs, info); }

API void wgpuAdapterInfoFreeMembers(WGPUAdapterInfo adapterInfo) {
    free((void*)adapterInfo.vendor.data);
    free((void*)adapterInfo.architecture.data);
    free((void*)adapterInfo.device.data);
    free((void*)adapterInfo.description.data);
}

static void limits_into(const TgwLimits* t, WGPULimits* l) {
    l->maxTextureDimension1D = (uint32_t)t->max_texture_dimension_1d;
    l->maxTextureDimension2D = (uint32_t)t->max_texture_dimension_2d;
    l->maxTextureDimension3D = (uint32_t)t->max_texture_dimension_3d;
    l->maxTextureArrayLayers = (uint32_t)t->max_texture_array_layers;
    l->maxBindGroups = (uint32_t)t->max_bind_groups;
    l->maxBindGroupsPlusVertexBuffers = (uint32_t)(t->max_bind_groups + t->max_vertex_buffers);
    l->maxBindingsPerBindGroup = (uint32_t)t->max_bindings_per_bind_group;
    l->maxDynamicUniformBuffersPerPipelineLayout = (uint32_t)t->max_dynamic_uniform_buffers_per_pipeline_layout;
    l->maxDynamicStorageBuffersPerPipelineLayout = (uint32_t)t->max_dynamic_storage_buffers_per_pipeline_layout;
    l->maxSampledTexturesPerShaderStage = (uint32_t)t->max_sampled_textures_per_shader_stage;
    l->maxSamplersPerShaderStage = (uint32_t)t->max_samplers_per_shader_stage;
    l->maxStorageBuffersPerShaderStage = (uint32_t)t->max_storage_buffers_per_shader_stage;
    l->maxStorageTexturesPerShaderStage = (uint32_t)t->max_storage_textures_per_shader_stage;
    l->maxUniformBuffersPerShaderStage = (uint32_t)t->max_uniform_buffers_per_shader_stage;
    l->maxUniformBufferBindingSize = t->max_uniform_buffer_binding_size;
    l->maxStorageBufferBindingSize = t->max_storage_buffer_binding_size;
    l->minUniformBufferOffsetAlignment = (uint32_t)t->min_uniform_buffer_offset_alignment;
    l->minStorageBufferOffsetAlignment = (uint32_t)t->min_storage_buffer_offset_alignment;
    l->maxVertexBuffers = (uint32_t)t->max_vertex_buffers;
    l->maxBufferSize = t->max_buffer_size;
    l->maxVertexAttributes = (uint32_t)t->max_vertex_attributes;
    l->maxVertexBufferArrayStride = (uint32_t)t->max_vertex_buffer_array_stride;
    l->maxInterStageShaderVariables = (uint32_t)t->max_inter_stage_shader_variables;
    l->maxColorAttachments = (uint32_t)t->max_color_attachments;
    l->maxColorAttachmentBytesPerSample = (uint32_t)t->max_color_attachment_bytes_per_sample;
    l->maxComputeWorkgroupStorageSize = (uint32_t)t->max_compute_workgroup_storage_size;
    l->maxComputeInvocationsPerWorkgroup = (uint32_t)t->max_compute_invocations_per_workgroup;
    l->maxComputeWorkgroupSizeX = (uint32_t)t->max_compute_workgroup_size_x;
    l->maxComputeWorkgroupSizeY = (uint32_t)t->max_compute_workgroup_size_y;
    l->maxComputeWorkgroupSizeZ = (uint32_t)t->max_compute_workgroup_size_z;
    l->maxComputeWorkgroupsPerDimension = (uint32_t)t->max_compute_workgroups_per_dimension;
    l->maxImmediateSize = (uint32_t)t->max_immediate_size;
    // wgpu.h's native limits, where they are chained
    for (WGPUChainedStruct* c = l->nextInChain; c; c = c->next) {
        if (c->sType == (WGPUSType)WGPUSType_NativeLimits) {
            WGPUNativeLimits* n = (WGPUNativeLimits*)c;
            n->maxNonSamplerBindings = 1000000;
            n->maxBindingArrayElementsPerShaderStage = 0;
            n->maxBindingArraySamplerElementsPerShaderStage = 0;
            n->maxMultiviewViewCount = 0;
        }
    }
}

API WGPUStatus wgpuAdapterGetLimits(WGPUAdapter adapter, WGPULimits* limits) {
    TgwLimits t;
    tgw_adapter_limits(adapter->rs, &t);
    limits_into(&t, limits);
    return WGPUStatus_Success;
}

static const WGPUFeatureName FEATURES[] = {WGPUFeatureName_ShaderF16, WGPUFeatureName_TimestampQuery};
static const uint64_t FEATURE_BITS[] = {TGW_SHADER_F16, TGW_TIMESTAMP_QUERY};
#define N_FEATURES (sizeof FEATURES / sizeof FEATURES[0])

static uint64_t bit_of(WGPUFeatureName f) {
    for (size_t i = 0; i < N_FEATURES; i++)
        if (FEATURES[i] == f) return FEATURE_BITS[i];
    return 0;
}

static void features_into(uint64_t bits, WGPUSupportedFeatures* out) {
    WGPUFeatureName* list = calloc(N_FEATURES, sizeof *list);
    size_t n = 0;
    for (size_t i = 0; i < N_FEATURES; i++)
        if (bits & FEATURE_BITS[i]) list[n++] = FEATURES[i];
    out->featureCount = n;
    out->features = list;
}

API void wgpuAdapterGetFeatures(WGPUAdapter adapter, WGPUSupportedFeatures* features) { features_into(tgw_adapter_features(adapter->rs), features); }
API void wgpuSupportedFeaturesFreeMembers(WGPUSupportedFeatures supportedFeatures) { free((void*)supportedFeatures.features); }
API WGPUBool wgpuAdapterHasFeature(WGPUAdapter adapter, WGPUFeatureName feature) { return (tgw_adapter_features(adapter->rs) & bit_of(feature)) != 0; }

API WGPUFuture wgpuAdapterRequestDevice(WGPUAdapter adapter, WGPUDeviceDescriptor const* descriptor, WGPURequestDeviceCallbackInfo callbackInfo) {
    uint64_t want = 0;
    if (descriptor)
        for (size_t i = 0; i < descriptor->requiredFeatureCount; i++) want |= bit_of(descriptor->requiredFeatures[i]);
    char* error = NULL;
    TgwDevice* rs = tgw_device_request(adapter->rs, want, &error);
    WGPUDevice d = NULL;
    char* message = NULL;
    if (rs) {
        d = NEW(WGPUDevice);
        d->refs = 1;
        d->rs = rs;
        d->adapter = adapter;
        wgpuAdapterAddRef(adapter);
        if (descriptor) {
            d->uncaptured = descriptor->uncapturedErrorCallbackInfo;
            d->lost = descriptor->deviceLostCallbackInfo;
        }
    } else if (error) {
        message = dup_of(error);
        tgw_string_free(error);
    }
    return post(EV_DEVICE, callbackInfo.mode, (void (*)(void))callbackInfo.callback, callbackInfo.userdata1, callbackInfo.userdata2,
                d ? WGPURequestDeviceStatus_Success : WGPURequestDeviceStatus_Error, 0, d, message);
}

// ---- device ----

API WGPUQueue wgpuDeviceGetQueue(WGPUDevice device) {
    if (device->queue) {
        wgpuQueueAddRef(device->queue);
        return device->queue;
    }
    WGPUQueue q = NEW(WGPUQueue);
    q->refs = 1;
    q->device = device;
    wgpuDeviceAddRef(device);
    device->queue = q;
    return q;
}

API WGPUStatus wgpuDeviceGetLimits(WGPUDevice device, WGPULimits* limits) {
    TgwLimits t;
    tgw_device_limits(device->rs, &t);
    limits_into(&t, limits);
    return WGPUStatus_Success;
}

API void wgpuDeviceGetFeatures(WGPUDevice device, WGPUSupportedFeatures* features) { features_into(tgw_device_features(device->rs), features); }
API WGPUBool wgpuDeviceHasFeature(WGPUDevice device, WGPUFeatureName feature) { return (tgw_device_features(device->rs) & bit_of(feature)) != 0; }
API WGPUStatus wgpuDeviceGetAdapterInfo(WGPUDevice device, WGPUAdapterInfo* adapterInfo) { return info_of(device->adapter->rs, adapterInfo); }

API void wgpuDevicePushErrorScope(WGPUDevice device, WGPUErrorFilter filter) {
    tgw_device_push_scope(device->rs, filter == WGPUErrorFilter_OutOfMemory ? TGW_OUT_OF_MEMORY : filter == WGPUErrorFilter_Internal ? TGW_INTERNAL : TGW_VALIDATION);
}

API WGPUFuture wgpuDevicePopErrorScope(WGPUDevice device, WGPUPopErrorScopeCallbackInfo callbackInfo) {
    uint32_t kind = 0;
    char* m = NULL;
    int popped = tgw_device_pop_scope(device->rs, &kind, &m);
    char* message = NULL;
    if (m) {
        message = dup_of(m);
        tgw_string_free(m);
    } else if (popped < 0) {
        message = dup_of("no error scope to pop");
    }
    return post(EV_POP_SCOPE, callbackInfo.mode, (void (*)(void))callbackInfo.callback, callbackInfo.userdata1, callbackInfo.userdata2,
                popped < 0 ? WGPUPopErrorScopeStatus_Error : WGPUPopErrorScopeStatus_Success, kind ? error_type(kind) : WGPUErrorType_NoError, NULL, message);
}

API WGPUBool wgpuDevicePoll(WGPUDevice device, WGPUBool wait, WGPUSubmissionIndex const* submissionIndex) {
    (void)submissionIndex;
    int empty = tgw_device_poll(device->rs, wait);
    tell(device);
    process_events();
    return empty;
}

API void wgpuDeviceDestroy(WGPUDevice device) {
    tgw_device_destroy(device->rs);
    if (!device->lost_told && device->lost.callback) {
        device->lost_told = 1;
        post(EV_LOST, device->lost.mode, (void (*)(void))device->lost.callback, device->lost.userdata1, device->lost.userdata2, WGPUDeviceLostReason_Destroyed, 0, device,
             dup_of("the device was destroyed"));
    }
}

// ---- buffers ----

API WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice device, WGPUBufferDescriptor const* descriptor) {
    TgwBuffer* rs = tgw_buffer_create(device->rs, descriptor->size, (uint32_t)descriptor->usage, descriptor->mappedAtCreation);
    tell(device);
    if (!rs) return NULL;
    WGPUBuffer b = CHILD(WGPUBuffer, device, rs);
    b->size = descriptor->size;
    b->usage = descriptor->usage;
    b->state = descriptor->mappedAtCreation ? WGPUBufferMapState_Mapped : WGPUBufferMapState_Unmapped;
    b->write = descriptor->mappedAtCreation != 0;
    return b;
}

API WGPUFuture wgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo callbackInfo) {
    uint64_t n = size == WGPU_WHOLE_MAP_SIZE ? buffer->size - offset : size;
    int write = (mode & WGPUMapMode_Write) != 0;
    int failed = tgw_buffer_map(buffer->device->rs, buffer->rs, write, offset, n);
    tell(buffer->device);
    if (!failed) {
        buffer->state = WGPUBufferMapState_Mapped;
        buffer->write = write;
    }
    return post(EV_MAP, callbackInfo.mode, (void (*)(void))callbackInfo.callback, callbackInfo.userdata1, callbackInfo.userdata2,
                failed ? WGPUMapAsyncStatus_Error : WGPUMapAsyncStatus_Success, 0, NULL, failed ? dup_of("the buffer could not be mapped") : NULL);
}

static void* mapped_range(WGPUBuffer buffer, size_t offset, size_t size) {
    if (buffer->state != WGPUBufferMapState_Mapped) return NULL;
    uint64_t n = size == WGPU_WHOLE_MAP_SIZE ? buffer->size - offset : size;
    void* p = tgw_buffer_range(buffer->device->rs, buffer->rs, offset, n, buffer->write);
    tell(buffer->device);
    return p;
}

API void* wgpuBufferGetMappedRange(WGPUBuffer buffer, size_t offset, size_t size) { return mapped_range(buffer, offset, size); }
API void const* wgpuBufferGetConstMappedRange(WGPUBuffer buffer, size_t offset, size_t size) { return mapped_range(buffer, offset, size); }

API WGPUStatus wgpuBufferReadMappedRange(WGPUBuffer buffer, size_t offset, void* data, size_t size) {
    void* p = mapped_range(buffer, offset, size);
    if (!p) return WGPUStatus_Error;
    memcpy(data, p, size);
    return WGPUStatus_Success;
}

API WGPUStatus wgpuBufferWriteMappedRange(WGPUBuffer buffer, size_t offset, void const* data, size_t size) {
    void* p = mapped_range(buffer, offset, size);
    if (!p) return WGPUStatus_Error;
    memcpy(p, data, size);
    return WGPUStatus_Success;
}

API void wgpuBufferUnmap(WGPUBuffer buffer) {
    tgw_buffer_unmap(buffer->rs);
    tell(buffer->device);
    buffer->state = WGPUBufferMapState_Unmapped;
    buffer->write = 0;
}

API WGPUBufferMapState wgpuBufferGetMapState(WGPUBuffer buffer) { return buffer->state; }
API uint64_t wgpuBufferGetSize(WGPUBuffer buffer) { return buffer->size; }
API WGPUBufferUsage wgpuBufferGetUsage(WGPUBuffer buffer) { return buffer->usage; }

API void wgpuBufferDestroy(WGPUBuffer buffer) {
    tgw_buffer_destroy(buffer->rs);
    buffer->state = WGPUBufferMapState_Unmapped;
}

// ---- modules, layouts, pipelines, bind groups ----

API WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice device, WGPUShaderModuleDescriptor const* descriptor) {
    const WGPUShaderSourceWGSL* wgsl = NULL;
    for (const WGPUChainedStruct* c = descriptor->nextInChain; c; c = c->next)
        if (c->sType == WGPUSType_ShaderSourceWGSL) wgsl = (const WGPUShaderSourceWGSL*)c;
    if (!wgsl) {
        fprintf(stderr, "webgpu-tinygpu: a shader module is taken as WGSL\n");
        return NULL;
    }
    TgwShader* rs = tgw_shader_create(device->rs, wgsl->code.data, sv_len(wgsl->code));
    tell(device);
    return rs ? CHILD(WGPUShaderModule, device, rs) : NULL;
}

API WGPUFuture wgpuShaderModuleGetCompilationInfo(WGPUShaderModule shaderModule, WGPUCompilationInfoCallbackInfo callbackInfo) {
    (void)shaderModule;
    return post(EV_COMPILATION, callbackInfo.mode, (void (*)(void))callbackInfo.callback, callbackInfo.userdata1, callbackInfo.userdata2,
                WGPUCompilationInfoRequestStatus_Success, 0, NULL, NULL);
}

API WGPUBindGroupLayout wgpuDeviceCreateBindGroupLayout(WGPUDevice device, WGPUBindGroupLayoutDescriptor const* descriptor) {
    size_t n = descriptor->entryCount;
    TgwLayoutEntry* entries = calloc(n ? n : 1, sizeof *entries);
    for (size_t i = 0; i < n; i++) {
        const WGPUBindGroupLayoutEntry* e = &descriptor->entries[i];
        if (e->buffer.type == WGPUBufferBindingType_BindingNotUsed) {
            free(entries);
            fprintf(stderr, "webgpu-tinygpu: a binding other than a buffer's (binding %u): compute's buffers alone are taken\n", e->binding);
            return NULL;
        }
        entries[i] = (TgwLayoutEntry){e->binding,
                                      e->buffer.type == WGPUBufferBindingType_Uniform ? TGW_UNIFORM : e->buffer.type == WGPUBufferBindingType_ReadOnlyStorage ? TGW_READ_ONLY_STORAGE : TGW_STORAGE,
                                      (uint32_t)e->buffer.hasDynamicOffset, e->buffer.minBindingSize};
    }
    TgwBgl* rs = tgw_bgl_create(device->rs, entries, n);
    free(entries);
    tell(device);
    return rs ? CHILD(WGPUBindGroupLayout, device, rs) : NULL;
}

API WGPUPipelineLayout wgpuDeviceCreatePipelineLayout(WGPUDevice device, WGPUPipelineLayoutDescriptor const* descriptor) {
    size_t n = descriptor->bindGroupLayoutCount;
    TgwBgl** groups = calloc(n ? n : 1, sizeof *groups);
    for (size_t i = 0; i < n; i++) groups[i] = descriptor->bindGroupLayouts[i] ? descriptor->bindGroupLayouts[i]->rs : NULL;
    TgwPipelineLayout* rs = tgw_pipeline_layout_create(device->rs, groups, n);
    free(groups);
    tell(device);
    return rs ? CHILD(WGPUPipelineLayout, device, rs) : NULL;
}

static WGPUComputePipeline create_pipeline(WGPUDevice device, WGPUComputePipelineDescriptor const* descriptor) {
    const WGPUComputeState* c = &descriptor->compute;
    if (!c->module) return NULL;
    size_t n = c->constantCount;
    const char** keys = calloc(n ? n : 1, sizeof *keys);
    size_t* key_lens = calloc(n ? n : 1, sizeof *key_lens);
    double* values = calloc(n ? n : 1, sizeof *values);
    for (size_t i = 0; i < n; i++) {
        keys[i] = c->constants[i].key.data;
        key_lens[i] = sv_len(c->constants[i].key);
        values[i] = c->constants[i].value;
    }
    TgwPipeline* rs = tgw_pipeline_create(device->rs, descriptor->layout ? descriptor->layout->rs : NULL, c->module->rs, c->entryPoint.data, sv_len(c->entryPoint), keys,
                                          key_lens, values, n);
    free(keys);
    free(key_lens);
    free(values);
    return rs ? CHILD(WGPUComputePipeline, device, rs) : NULL;
}

API WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice device, WGPUComputePipelineDescriptor const* descriptor) {
    WGPUComputePipeline p = create_pipeline(device, descriptor);
    tell(device);
    return p;
}

API WGPUFuture wgpuDeviceCreateComputePipelineAsync(WGPUDevice device, WGPUComputePipelineDescriptor const* descriptor, WGPUCreateComputePipelineAsyncCallbackInfo callbackInfo) {
    // (an error of the pipeline's is its callback's, not the device's)
    tgw_device_push_scope(device->rs, TGW_VALIDATION);
    WGPUComputePipeline p = create_pipeline(device, descriptor);
    uint32_t kind = 0;
    char* m = NULL;
    tgw_device_pop_scope(device->rs, &kind, &m);
    char* message = m ? dup_of(m) : NULL;
    tgw_string_free(m);
    tell(device);
    return post(EV_PIPELINE, callbackInfo.mode, (void (*)(void))callbackInfo.callback, callbackInfo.userdata1, callbackInfo.userdata2,
                p ? WGPUCreatePipelineAsyncStatus_Success : WGPUCreatePipelineAsyncStatus_ValidationError, 0, p, message);
}

API WGPUBindGroupLayout wgpuComputePipelineGetBindGroupLayout(WGPUComputePipeline computePipeline, uint32_t groupIndex) {
    TgwBgl* rs = tgw_pipeline_bgl(computePipeline->device->rs, computePipeline->rs, groupIndex);
    tell(computePipeline->device);
    return rs ? CHILD(WGPUBindGroupLayout, computePipeline->device, rs) : NULL;
}

API WGPUBindGroup wgpuDeviceCreateBindGroup(WGPUDevice device, WGPUBindGroupDescriptor const* descriptor) {
    size_t n = descriptor->entryCount;
    TgwGroupEntry* entries = calloc(n ? n : 1, sizeof *entries);
    for (size_t i = 0; i < n; i++) {
        const WGPUBindGroupEntry* e = &descriptor->entries[i];
        if (!e->buffer) {
            free(entries);
            fprintf(stderr, "webgpu-tinygpu: a bind group entry other than a buffer (binding %u)\n", e->binding);
            return NULL;
        }
        entries[i] = (TgwGroupEntry){e->binding, e->buffer->rs, e->offset, e->size == WGPU_WHOLE_SIZE ? UINT64_MAX : e->size};
    }
    TgwBindGroup* rs = descriptor->layout ? tgw_bind_group_create(device->rs, descriptor->layout->rs, entries, n) : NULL;
    free(entries);
    tell(device);
    return rs ? CHILD(WGPUBindGroup, device, rs) : NULL;
}

// ---- commands ----

API WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(WGPUDevice device, WGPUCommandEncoderDescriptor const* descriptor) {
    (void)descriptor;
    TgwEncoder* rs = tgw_encoder_create(device->rs);
    tell(device);
    return rs ? CHILD(WGPUCommandEncoder, device, rs) : NULL;
}

API void wgpuCommandEncoderCopyBufferToBuffer(WGPUCommandEncoder commandEncoder, WGPUBuffer source, uint64_t sourceOffset, WGPUBuffer destination, uint64_t destinationOffset,
                                              uint64_t size) {
    uint64_t n = size == WGPU_WHOLE_SIZE ? source->size - sourceOffset : size;
    tgw_encoder_copy(commandEncoder->device->rs, commandEncoder->rs, source->rs, sourceOffset, destination->rs, destinationOffset, n);
    tell(commandEncoder->device);
}

API void wgpuCommandEncoderClearBuffer(WGPUCommandEncoder commandEncoder, WGPUBuffer buffer, uint64_t offset, uint64_t size) {
    tgw_encoder_clear(commandEncoder->device->rs, commandEncoder->rs, buffer->rs, offset, size == WGPU_WHOLE_SIZE ? UINT64_MAX : size);
    tell(commandEncoder->device);
}

API WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder commandEncoder, WGPUComputePassDescriptor const* descriptor) {
    (void)descriptor;
    TgwPass* rs = tgw_pass_begin(commandEncoder->device->rs, commandEncoder->rs);
    tell(commandEncoder->device);
    return rs ? CHILD(WGPUComputePassEncoder, commandEncoder->device, rs) : NULL;
}

API WGPUCommandBuffer wgpuCommandEncoderFinish(WGPUCommandEncoder commandEncoder, WGPUCommandBufferDescriptor const* descriptor) {
    (void)descriptor;
    TgwCommandBuffer* rs = tgw_encoder_finish(commandEncoder->device->rs, commandEncoder->rs);
    tell(commandEncoder->device);
    return rs ? CHILD(WGPUCommandBuffer, commandEncoder->device, rs) : NULL;
}

API void wgpuCommandEncoderInsertDebugMarker(WGPUCommandEncoder commandEncoder, WGPUStringView markerLabel) {
    (void)commandEncoder;
    (void)markerLabel;
}
API void wgpuCommandEncoderPushDebugGroup(WGPUCommandEncoder commandEncoder, WGPUStringView groupLabel) {
    (void)commandEncoder;
    (void)groupLabel;
}
API void wgpuCommandEncoderPopDebugGroup(WGPUCommandEncoder commandEncoder) { (void)commandEncoder; }

API void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder computePassEncoder, WGPUComputePipeline pipeline) {
    tgw_pass_set_pipeline(computePassEncoder->device->rs, computePassEncoder->rs, pipeline ? pipeline->rs : NULL);
    tell(computePassEncoder->device);
}

API void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder computePassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount,
                                            uint32_t const* dynamicOffsets) {
    tgw_pass_set_bind_group(computePassEncoder->device->rs, computePassEncoder->rs, groupIndex, group ? group->rs : NULL, dynamicOffsets, dynamicOffsetCount);
    tell(computePassEncoder->device);
}

API void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder computePassEncoder, uint32_t workgroupCountX, uint32_t workgroupCountY, uint32_t workgroupCountZ) {
    tgw_pass_dispatch(computePassEncoder->device->rs, computePassEncoder->rs, workgroupCountX, workgroupCountY, workgroupCountZ);
    tell(computePassEncoder->device);
}

API void wgpuComputePassEncoderDispatchWorkgroupsIndirect(WGPUComputePassEncoder computePassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset) {
    tgw_pass_dispatch_indirect(computePassEncoder->device->rs, computePassEncoder->rs, indirectBuffer->rs, indirectOffset);
    tell(computePassEncoder->device);
}

API void wgpuComputePassEncoderEnd(WGPUComputePassEncoder computePassEncoder) {
    tgw_pass_end(computePassEncoder->rs);
    tell(computePassEncoder->device);
}

API void wgpuComputePassEncoderInsertDebugMarker(WGPUComputePassEncoder computePassEncoder, WGPUStringView markerLabel) {
    (void)computePassEncoder;
    (void)markerLabel;
}
API void wgpuComputePassEncoderPushDebugGroup(WGPUComputePassEncoder computePassEncoder, WGPUStringView groupLabel) {
    (void)computePassEncoder;
    (void)groupLabel;
}
API void wgpuComputePassEncoderPopDebugGroup(WGPUComputePassEncoder computePassEncoder) { (void)computePassEncoder; }

// ---- the queue ----

API void wgpuQueueSubmit(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const* commands) {
    TgwCommandBuffer** list = calloc(commandCount ? commandCount : 1, sizeof *list);
    for (size_t i = 0; i < commandCount; i++) list[i] = commands[i] ? commands[i]->rs : NULL;
    tgw_queue_submit(queue->device->rs, list, commandCount);
    free(list);
    tell(queue->device);
}

API void wgpuQueueWriteBuffer(WGPUQueue queue, WGPUBuffer buffer, uint64_t bufferOffset, void const* data, size_t size) {
    tgw_queue_write(queue->device->rs, buffer->rs, bufferOffset, data, size);
    tell(queue->device);
}

API WGPUFuture wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo callbackInfo) {
    tgw_device_poll(queue->device->rs, 1);
    tell(queue->device);
    return post(EV_WORK_DONE, callbackInfo.mode, (void (*)(void))callbackInfo.callback, callbackInfo.userdata1, callbackInfo.userdata2, WGPUQueueWorkDoneStatus_Success, 0, NULL,
                NULL);
}
