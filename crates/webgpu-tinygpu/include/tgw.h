// tgw.h: between webgpu.c (webgpu.h's functions, which read the API's structs) and lib.rs (wgpu's objects): plain
// arguments only, so no struct of webgpu.h is laid out twice. Every object is lib.rs's, owned by the C object that
// holds it and freed by its tgw_*_free. A call that fails reports the error to its device (lib.rs keeps the error
// scopes) and returns NULL or nonzero; what reaches no scope webgpu.c takes with tgw_device_take_error.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct TgwAdapter TgwAdapter;
typedef struct TgwDevice TgwDevice;
typedef struct TgwBuffer TgwBuffer;
typedef struct TgwShader TgwShader;
typedef struct TgwBgl TgwBgl;
typedef struct TgwPipelineLayout TgwPipelineLayout;
typedef struct TgwPipeline TgwPipeline;
typedef struct TgwBindGroup TgwBindGroup;
typedef struct TgwEncoder TgwEncoder;
typedef struct TgwPass TgwPass;
typedef struct TgwCommandBuffer TgwCommandBuffer;

// an error's kind, as an error scope's filter names it
enum { TGW_VALIDATION = 1, TGW_OUT_OF_MEMORY = 2, TGW_INTERNAL = 3 };

// features, as bits
enum { TGW_SHADER_F16 = 1, TGW_TIMESTAMP_QUERY = 2 };

// a binding's buffer, in a bind group layout
enum { TGW_UNIFORM = 1, TGW_STORAGE = 2, TGW_READ_ONLY_STORAGE = 3 };

typedef struct TgwLimits {
    uint64_t max_bind_groups, max_bindings_per_bind_group, max_dynamic_uniform_buffers_per_pipeline_layout,
        max_dynamic_storage_buffers_per_pipeline_layout, max_storage_buffers_per_shader_stage,
        max_uniform_buffers_per_shader_stage, max_uniform_buffer_binding_size, max_storage_buffer_binding_size,
        min_uniform_buffer_offset_alignment, min_storage_buffer_offset_alignment, max_buffer_size,
        max_compute_workgroup_storage_size, max_compute_invocations_per_workgroup, max_compute_workgroup_size_x,
        max_compute_workgroup_size_y, max_compute_workgroup_size_z, max_compute_workgroups_per_dimension,
        max_texture_dimension_1d, max_texture_dimension_2d, max_texture_dimension_3d, max_texture_array_layers,
        max_sampled_textures_per_shader_stage, max_samplers_per_shader_stage, max_storage_textures_per_shader_stage,
        max_vertex_buffers, max_vertex_attributes, max_vertex_buffer_array_stride, max_inter_stage_shader_variables,
        max_color_attachments, max_color_attachment_bytes_per_sample, max_immediate_size;
} TgwLimits;

typedef struct TgwInfo {
    char name[256];
    char driver[256];
    char vendor[64];
    uint32_t vendor_id, device_id;
    uint32_t backend;   // 0 the TinyGPU server's card (or its emulator); else wgpu's own: 1 Metal, 2 Vulkan, 3 other
    uint32_t discrete;
    uint32_t subgroup_min, subgroup_max;
} TgwInfo;

typedef struct TgwLayoutEntry {
    uint32_t binding;
    uint32_t kind;   // TGW_UNIFORM, ...
    uint32_t has_dynamic_offset;
    uint64_t min_binding_size;
} TgwLayoutEntry;

typedef struct TgwGroupEntry {
    uint32_t binding;
    TgwBuffer* buffer;
    uint64_t offset;
    uint64_t size;   // UINT64_MAX: to the buffer's end
} TgwGroupEntry;

// strings come as pointer and length; an error comes back as a string of lib.rs's, let go by tgw_string_free
void tgw_string_free(char* s);

TgwAdapter* tgw_adapter_open(char** error);
void tgw_adapter_info(TgwAdapter* a, TgwInfo* out);
void tgw_adapter_limits(TgwAdapter* a, TgwLimits* out);
uint64_t tgw_adapter_features(TgwAdapter* a);
void tgw_adapter_free(TgwAdapter* a);

TgwDevice* tgw_device_request(TgwAdapter* a, uint64_t features, char** error);
void tgw_device_limits(TgwDevice* d, TgwLimits* out);
uint64_t tgw_device_features(TgwDevice* d);
void tgw_device_push_scope(TgwDevice* d, uint32_t filter);
// the popped scope's first error (its kind, else 0) and message (NULL with none); -1 with no scope to pop
int tgw_device_pop_scope(TgwDevice* d, uint32_t* kind, char** message);
// an error no scope took, else NULL
char* tgw_device_take_error(TgwDevice* d, uint32_t* kind);
int tgw_device_poll(TgwDevice* d, int wait);
void tgw_device_destroy(TgwDevice* d);
void tgw_device_free(TgwDevice* d);

TgwBuffer* tgw_buffer_create(TgwDevice* d, uint64_t size, uint32_t usage, int mapped_at_creation);
int tgw_buffer_map(TgwDevice* d, TgwBuffer* b, int write, uint64_t offset, uint64_t size);
void* tgw_buffer_range(TgwDevice* d, TgwBuffer* b, uint64_t offset, uint64_t size, int write);
void tgw_buffer_unmap(TgwBuffer* b);
void tgw_buffer_destroy(TgwBuffer* b);
void tgw_buffer_free(TgwBuffer* b);

TgwShader* tgw_shader_create(TgwDevice* d, const char* wgsl, size_t len);
void tgw_shader_free(TgwShader* s);

TgwBgl* tgw_bgl_create(TgwDevice* d, const TgwLayoutEntry* entries, size_t n);
void tgw_bgl_free(TgwBgl* l);
TgwPipelineLayout* tgw_pipeline_layout_create(TgwDevice* d, TgwBgl* const* groups, size_t n);
void tgw_pipeline_layout_free(TgwPipelineLayout* l);
// keys and values of the pipeline's overrides: n of each
TgwPipeline* tgw_pipeline_create(TgwDevice* d, TgwPipelineLayout* layout, TgwShader* s, const char* entry, size_t entry_len,
                                 const char* const* keys, const size_t* key_lens, const double* values, size_t n);
TgwBgl* tgw_pipeline_bgl(TgwDevice* d, TgwPipeline* p, uint32_t index);
void tgw_pipeline_free(TgwPipeline* p);
TgwBindGroup* tgw_bind_group_create(TgwDevice* d, TgwBgl* layout, const TgwGroupEntry* entries, size_t n);
void tgw_bind_group_free(TgwBindGroup* g);

TgwEncoder* tgw_encoder_create(TgwDevice* d);
void tgw_encoder_copy(TgwDevice* d, TgwEncoder* e, TgwBuffer* src, uint64_t src_offset, TgwBuffer* dst, uint64_t dst_offset, uint64_t size);
void tgw_encoder_clear(TgwDevice* d, TgwEncoder* e, TgwBuffer* b, uint64_t offset, uint64_t size);
TgwCommandBuffer* tgw_encoder_finish(TgwDevice* d, TgwEncoder* e);
void tgw_encoder_free(TgwEncoder* e);
TgwPass* tgw_pass_begin(TgwDevice* d, TgwEncoder* e);
void tgw_pass_set_pipeline(TgwDevice* d, TgwPass* p, TgwPipeline* pipeline);
void tgw_pass_set_bind_group(TgwDevice* d, TgwPass* p, uint32_t index, TgwBindGroup* g, const uint32_t* offsets, size_t n);
void tgw_pass_dispatch(TgwDevice* d, TgwPass* p, uint32_t x, uint32_t y, uint32_t z);
void tgw_pass_dispatch_indirect(TgwDevice* d, TgwPass* p, TgwBuffer* b, uint64_t offset);
void tgw_pass_end(TgwPass* p);
void tgw_pass_free(TgwPass* p);
void tgw_command_buffer_free(TgwCommandBuffer* c);

void tgw_queue_submit(TgwDevice* d, TgwCommandBuffer* const* buffers, size_t n);
void tgw_queue_write(TgwDevice* d, TgwBuffer* b, uint64_t offset, const void* data, size_t size);
