// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    bool inputs_allocated[GGML_SCHED_MAX_SPLIT_INPUTS];
    bool inputs_added[GGML_SCHED_MAX_SPLIT_INPUTS];
    bool input_prefetched[GGML_SCHED_MAX_SPLIT_INPUTS];
    bool has_prefetched_inputs;
    bool input_transient[GGML_SCHED_MAX_SPLIT_INPUTS];
    bool input_full_moe_prefetch[GGML_SCHED_MAX_SPLIT_INPUTS];
    ggml_backend_buffer_t transient_buffers[GGML_SCHED_MAX_SPLIT_INPUTS];
    size_t transient_sizes[GGML_SCHED_MAX_SPLIT_INPUTS];
    bool input_resident[GGML_SCHED_MAX_SPLIT_INPUTS];
    bool input_resident_hit[GGML_SCHED_MAX_SPLIT_INPUTS];
    bool input_compact_moe[GGML_SCHED_MAX_SPLIT_INPUTS];
    struct ggml_tensor * compact_node[GGML_SCHED_MAX_SPLIT_INPUTS];
    struct ggml_tensor * compact_original_ids[GGML_SCHED_MAX_SPLIT_INPUTS];
    struct ggml_tensor * compact_ids_copy[GGML_SCHED_MAX_SPLIT_INPUTS];
    ggml_backend_buffer_t compact_ids_buffer[GGML_SCHED_MAX_SPLIT_INPUTS];
    int64_t compact_original_ne2[GGML_SCHED_MAX_SPLIT_INPUTS];
    std::vector<int32_t> * compact_experts[GGML_SCHED_MAX_SPLIT_INPUTS];
    std::vector<int32_t> * compact_remapped_ids[GGML_SCHED_MAX_SPLIT_INPUTS];
    std::vector<int32_t> * compact_slots[GGML_SCHED_MAX_SPLIT_INPUTS];
    std::vector<uint8_t> * compact_misses[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    enum ggml_backend_sched_split_reason reason;
    size_t weight_bytes;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched_resident {
    const struct ggml_tensor * source;
    ggml_backend_buffer_t source_buffer;
    const void * source_data;
    size_t logical_size;
    int backend_id;
    struct ggml_tensor * copy;
    ggml_backend_buffer_t buffer;
    size_t allocation_size;
    uint64_t completed_use;
    uint64_t frequency;
    bool executing;
    bool expert_tier;
    std::vector<int32_t> experts;
    std::vector<int32_t> expert_slots;
    std::vector<uint64_t> expert_frequency;
    std::vector<uint64_t> expert_completed_use;
};

struct ggml_backend_sched_resident_key {
    const struct ggml_tensor * source;
    ggml_backend_buffer_t source_buffer;
    const void * source_data;
    int backend_id;
    enum ggml_type type;
    bool expert_slab;
    std::array<int64_t, GGML_MAX_DIMS> ne;
    std::array<size_t, GGML_MAX_DIMS> nb;
    std::vector<int32_t> experts;

    bool operator==(const ggml_backend_sched_resident_key & other) const {
        return source == other.source && source_buffer == other.source_buffer && source_data == other.source_data &&
            backend_id == other.backend_id && type == other.type && expert_slab == other.expert_slab &&
            ne == other.ne && nb == other.nb && experts == other.experts;
    }
};

struct ggml_backend_sched_resident_key_hash {
    size_t operator()(const ggml_backend_sched_resident_key & key) const {
        size_t h = std::hash<const void *>{}(key.source) ^ (std::hash<const void *>{}(key.source_data) << 1);
        h ^= std::hash<const void *>{}(key.source_buffer) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.backend_id) + std::hash<int>{}((int) key.type);
        h ^= std::hash<bool>{}(key.expert_slab) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            h ^= std::hash<int64_t>{}(key.ne[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<size_t>{}(key.nb[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        for (int32_t expert : key.experts) {
            h ^= std::hash<int32_t>{}(expert) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

using ggml_backend_sched_resident_map = std::unordered_map<
    ggml_backend_sched_resident_key, ggml_backend_sched_resident, ggml_backend_sched_resident_key_hash>;

static constexpr int GGML_BACKEND_SCHED_STAGING_SLOTS = 4;
static constexpr size_t GGML_BACKEND_SCHED_STORAGE_PADDING = (size_t) 2 * 1024 * 1024;

struct ggml_backend_sched_staging {
    ggml_backend_buffer_t buffers[GGML_BACKEND_SCHED_STAGING_SLOTS];
    ggml_backend_event_t events[GGML_BACKEND_SCHED_STAGING_SLOTS];
    size_t capacities[GGML_BACKEND_SCHED_STAGING_SLOTS];
    bool pending[GGML_BACKEND_SCHED_STAGING_SLOTS];
    bool reserved[GGML_BACKEND_SCHED_STAGING_SLOTS];
    int next;
};

struct ggml_backend_sched_storage_prefetch {
    std::thread worker;
    const uint8_t * logical_src;
    size_t size;
    size_t data_offset;
    int split_id;
    int input_id;
    bool active;
    bool success;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;
    bool force_weight_offload;
    bool async_weight_prefetch;
    bool prefetch_full_moe;
    ggml_backend_sched_weight_read_callback weight_read_callback;
    ggml_backend_sched_weight_read_padded_callback weight_read_padded_callback;
    void * weight_read_callback_user_data;
    bool force_weight_offload_split_configured;
    bool force_weight_offload_token_generation;
    float force_weight_offload_split[GGML_SCHED_MAX_BACKENDS];
    int force_weight_offload_max_layer;

    // sequential load: maximum total weight bytes per GPU split; 0 = unlimited
    size_t max_weight_bytes_per_split[GGML_SCHED_MAX_BACKENDS];
    bool weight_window_configured[GGML_SCHED_MAX_BACKENDS];
    bool weight_window_memory_valid[GGML_SCHED_MAX_BACKENDS];
    size_t weight_window_limit[GGML_SCHED_MAX_BACKENDS];
    size_t weight_window_safety_reserve[GGML_SCHED_MAX_BACKENDS];
    size_t transient_bytes;
    size_t transient_count;
    struct ggml_backend_sched_transient_metrics transient_metrics;
    std::unordered_set<const struct ggml_tensor *> * transient_sources_seen;
    bool residency_enabled[GGML_SCHED_MAX_BACKENDS];
    bool persistent_weight_residency;
    uint64_t residency_use_clock;
    ggml_backend_sched_resident_map * residents;

    ggml_backend_t prefetch_backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_event_t prefetch_events[GGML_SCHED_MAX_BACKENDS][2];
    struct ggml_backend_sched_staging staging[GGML_SCHED_MAX_BACKENDS];
    struct ggml_backend_sched_storage_prefetch
        storage_prefetch[GGML_SCHED_MAX_BACKENDS][GGML_BACKEND_SCHED_STAGING_SLOTS];

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static bool ggml_backend_sched_input_is_transient(
        ggml_backend_sched_t sched, const struct ggml_tensor * input, int backend_id) {
    return sched->force_weight_offload && sched->n_copies == 1 &&
        input->view_src == NULL && input->buffer != NULL && input->data != NULL &&
        ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(input->buffer) && !ggml_backend_buft_is_host(sched->bufts[backend_id]);
}

static bool ggml_backend_sched_input_is_full_moe_prefetch(
        ggml_backend_sched_t sched, const struct ggml_tensor * node,
        const struct ggml_tensor * input, int backend_id) {
    if (!sched->prefetch_full_moe || sched->force_weight_offload || sched->n_copies != 1 ||
            node == NULL || node->op != GGML_OP_MUL_MAT_ID || node->src[0] != input || node->src[2] == NULL ||
            input == NULL || input->view_src != NULL || input->buffer == NULL || input->data == NULL ||
            ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_backend_buffer_is_host(input->buffer) || ggml_backend_buft_is_host(sched->bufts[backend_id]) ||
            input->ne[2] <= 1) {
        return false;
    }

    // At large prefill batches the router selects nearly every expert.  Waiting
    // for the IDs and then issuing sparse copies only serializes PCIe behind the
    // router.  The reference optimization used this conservative density test:
    // at least twice as many routed IDs as experts.
    const struct ggml_tensor * ids = node->src[2];
    if (ids->ne[0] <= 0 || ids->ne[1] <= 0) {
        return false;
    }
    const uint64_t n0 = (uint64_t) ids->ne[0];
    const uint64_t n1 = (uint64_t) ids->ne[1];
    const uint64_t n_ids = n0 > UINT64_MAX / n1 ? UINT64_MAX : n0 * n1;
    const uint64_t n_expert = (uint64_t) input->ne[2];
    return n_expert <= UINT64_MAX / 2 && n_ids >= 2 * n_expert;
}

static ggml_backend_sched_resident_key ggml_backend_sched_resident_key_make(
        const struct ggml_tensor * source, int backend_id, const std::vector<int32_t> & experts) {
    ggml_backend_sched_resident_key key{};
    key.source = source;
    key.source_buffer = source->buffer;
    key.source_data = source->data;
    key.backend_id = backend_id;
    key.type = source->type;
    key.expert_slab = false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        key.ne[i] = source->ne[i];
        key.nb[i] = source->nb[i];
    }
    key.experts = experts;
    return key;
}

static ggml_backend_sched_resident_key ggml_backend_sched_expert_slab_key_make(
        const struct ggml_tensor * source, int backend_id) {
    auto key = ggml_backend_sched_resident_key_make(source, backend_id, {});
    key.expert_slab = true;
    return key;
}

static bool ggml_backend_sched_compact_moe_layout_supported(
        const struct ggml_tensor * weights, const struct ggml_tensor * ids) {
    return weights != NULL && ids != NULL && weights->view_src == NULL && weights->ne[2] > 1 && weights->ne[3] == 1 &&
        weights->nb[0] == ggml_type_size(weights->type) && weights->nb[2] >= weights->nb[1] * (size_t) weights->ne[1] &&
        ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1 && ids->nb[0] == sizeof(int32_t) &&
        ids->nb[1] >= ids->nb[0] * (size_t) ids->ne[0];
}

static void ggml_backend_sched_compact_reset_input(
        struct ggml_backend_sched_split * split, int input_id, bool restore_node) {
    if (restore_node && split->compact_node[input_id] != NULL && split->compact_ids_copy[input_id] != NULL &&
            split->compact_node[input_id]->src[2] == split->compact_ids_copy[input_id]) {
        split->compact_node[input_id]->src[2] = split->compact_original_ids[input_id];
    }
    if (split->input_compact_moe[input_id]) {
        struct ggml_tensor * input_cpy = split->compact_node[input_id] != NULL ? split->compact_node[input_id]->src[0] : NULL;
        if (input_cpy != NULL && split->compact_original_ne2[input_id] > 0) {
            input_cpy->ne[2] = split->compact_original_ne2[input_id];
        }
    }
    ggml_backend_buffer_free(split->compact_ids_buffer[input_id]);
    split->compact_ids_buffer[input_id] = NULL;
    split->compact_ids_copy[input_id] = NULL;
    split->compact_original_ids[input_id] = NULL;
    split->compact_node[input_id] = NULL;
    split->input_compact_moe[input_id] = false;
    split->compact_original_ne2[input_id] = 0;
    delete split->compact_experts[input_id];
    delete split->compact_remapped_ids[input_id];
    delete split->compact_slots[input_id];
    delete split->compact_misses[input_id];
    split->compact_experts[input_id] = NULL;
    split->compact_remapped_ids[input_id] = NULL;
    split->compact_slots[input_id] = NULL;
    split->compact_misses[input_id] = NULL;
}

static void ggml_backend_sched_counter_add(ggml_backend_sched_t sched, uint64_t * counter, uint64_t value) {
    if (UINT64_MAX - *counter < value) {
        *counter = UINT64_MAX;
        if (sched->transient_metrics.counter_overflow_count != UINT64_MAX) {
            sched->transient_metrics.counter_overflow_count++;
        }
    } else {
        *counter += value;
    }
}

static uint64_t ggml_backend_sched_elapsed_us(int64_t start_us) {
    const int64_t end_us = ggml_time_us();
    return end_us >= start_us ? (uint64_t) (end_us - start_us) : 0;
}

static int ggml_backend_sched_histogram_bucket(uint64_t value, uint64_t first_limit) {
    int bucket = 0;
    uint64_t limit = first_limit;
    while (bucket + 1 < GGML_BACKEND_SCHED_SPLIT_HISTOGRAM_BUCKETS && value > limit) {
        limit = limit > UINT64_MAX / 4 ? UINT64_MAX : limit * 4;
        ++bucket;
    }
    return bucket;
}

struct ggml_backend_sched_fault_sample {
    uint64_t minor;
    uint64_t major;
};

static ggml_backend_sched_fault_sample ggml_backend_sched_faults() {
#if defined(__linux__)
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return { (uint64_t) usage.ru_minflt, (uint64_t) usage.ru_majflt };
    }
#endif
    return {};
}

static void ggml_backend_sched_readahead(
        ggml_backend_sched_t sched, int backend_id, const void * data, size_t size) {
#if defined(__linux__)
    if (data == NULL || size == 0) {
        return;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return;
    }
    const uintptr_t begin = (uintptr_t) data;
    const uintptr_t aligned = begin & ~((uintptr_t) page_size - 1);
    const size_t prefix = begin - aligned;
    if (size > SIZE_MAX - prefix) {
        return;
    }
    const int64_t start = ggml_time_us();
    if (madvise((void *) aligned, size + prefix, MADV_WILLNEED) == 0) {
        auto & row = sched->transient_metrics.backends[backend_id];
        ggml_backend_sched_counter_add(sched, &row.mmap_readahead_calls, 1);
        ggml_backend_sched_counter_add(sched, &row.mmap_readahead_bytes, size);
        ggml_backend_sched_counter_add(sched, &row.mmap_readahead_time_us, ggml_backend_sched_elapsed_us(start));
    }
#else
    GGML_UNUSED(sched);
    GGML_UNUSED(backend_id);
    GGML_UNUSED(data);
    GGML_UNUSED(size);
#endif
}

static constexpr size_t GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_DEFAULT = (size_t) 64 * 1024 * 1024;
static constexpr size_t GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MIN     = (size_t) 16 * 1024 * 1024;
static constexpr size_t GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MAX     = (size_t) 256 * 1024 * 1024;
static constexpr size_t GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_THRESHOLD = (size_t) 1024 * 1024 * 1024;

static size_t ggml_backend_sched_transfer_chunk_size(
        ggml_backend_sched_t sched, int backend_id, size_t total_size) {
    if (total_size == 0) {
        return GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MIN;
    }

    // Explicit override is useful for storage/controller-specific tuning. Keep a
    // bounded range so four pinned ring slots cannot accidentally consume many GiB.
    const char * env = getenv("GGML_SEQUENTIAL_CHUNK_MB");
    if (env != NULL && env[0] != '\0') {
        char * end = NULL;
        const unsigned long long mb = strtoull(env, &end, 10);
        if (end != env && mb > 0) {
            const size_t requested = mb > SIZE_MAX/(1024ull*1024ull) ? SIZE_MAX : (size_t) mb*1024ull*1024ull;
            return std::min(total_size, std::max(GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MIN,
                std::min(requested, GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MAX)));
        }
    }

    // Automatic mode is workload- and memory-aware. Small transfers use smaller
    // chunks to reduce first-byte latency; very large sequential weights use wider
    // chunks to amortize O_DIRECT and PCIe submission overhead when live VRAM allows.
    size_t chunk = GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_DEFAULT;
    if (total_size < (size_t) 256*1024*1024) {
        chunk = (size_t) 32*1024*1024;
    } else if (total_size >= (size_t) 8*1024*1024*1024ull) {
        chunk = GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MAX;
    } else if (total_size >= (size_t) 2*1024*1024*1024ull) {
        chunk = (size_t) 128*1024*1024;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(sched->backends[backend_id]), &free_bytes, &total_bytes);
    if (free_bytes > 0 && total_bytes > 0 && free_bytes <= total_bytes) {
        if (free_bytes < (size_t) 2*1024*1024*1024ull) {
            chunk = std::min(chunk, (size_t) 32*1024*1024);
        } else if (free_bytes < (size_t) 4*1024*1024*1024ull) {
            chunk = std::min(chunk, GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_DEFAULT);
        }
    }

    return std::min(total_size, std::max(chunk, GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MIN));
}

static int ggml_backend_sched_transfer_prefetch_depth(
        ggml_backend_sched_t sched, int backend_id, size_t chunk_size, size_t total_size) {
    const int max_depth = GGML_BACKEND_SCHED_STAGING_SLOTS - 1;
    const char * env = getenv("GGML_SEQUENTIAL_PREFETCH_DEPTH");
    if (env != NULL && env[0] != '\0') {
        char * end = NULL;
        const long requested = strtol(env, &end, 10);
        if (end != env && requested > 0) {
            return std::max(1, std::min(max_depth, (int) requested));
        }
    }

    int depth = chunk_size >= GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MAX ? 1 :
        (chunk_size >= (size_t) 128*1024*1024 ? 2 : max_depth);

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(sched->backends[backend_id]), &free_bytes, &total_bytes);
    if (free_bytes > 0 && total_bytes > 0 && free_bytes <= total_bytes) {
        if (free_bytes < (size_t) 2*1024*1024*1024ull) {
            depth = 1;
        } else if (free_bytes < (size_t) 4*1024*1024*1024ull) {
            depth = std::min(depth, 2);
        }
    }

    if (chunk_size > 0) {
        const size_t n_chunks = total_size/chunk_size + (total_size % chunk_size != 0);
        depth = std::min(depth, (int) std::min<size_t>(n_chunks, max_depth));
    }
    return std::max(1, depth);
}

static void ggml_backend_sched_storage_prefetch_finish(ggml_backend_sched_storage_prefetch & prefetch) {
    if (prefetch.worker.joinable()) {
        prefetch.worker.join();
    }
}

static void ggml_backend_sched_storage_prefetch_release(
        ggml_backend_sched_t sched, int backend_id, int slot) {
    auto & task = sched->storage_prefetch[backend_id][slot];
    if (!task.active && !task.worker.joinable()) {
        return;
    }
    ggml_backend_sched_storage_prefetch_finish(task);
    // A storage worker waits for the previous H2D event before touching the slot.
    // Once joined, that old transfer is complete even when the read itself failed.
    sched->staging[backend_id].pending[slot] = false;
    sched->staging[backend_id].reserved[slot] = false;
    task.active = false;
    task.success = false;
    task.logical_src = NULL;
    task.size = 0;
    task.data_offset = 0;
    task.split_id = -1;
    task.input_id = -1;
}

static size_t ggml_backend_sched_staging_required_capacity(ggml_backend_sched_t sched, size_t size) {
    if (sched->weight_read_padded_callback == NULL) {
        return size;
    }
    return size <= SIZE_MAX - GGML_BACKEND_SCHED_STORAGE_PADDING ?
        size + GGML_BACKEND_SCHED_STORAGE_PADDING : SIZE_MAX;
}

static void ggml_backend_sched_staging_metrics_update(ggml_backend_sched_t sched, int backend_id) {
    size_t bytes = 0;
    for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
        bytes += sched->staging[backend_id].capacities[slot];
    }
    sched->transient_metrics.backends[backend_id].staging_buffer_bytes = bytes;
}

static bool ggml_backend_sched_staging_prepare(
        ggml_backend_sched_t sched, int backend_id, int slot, size_t size, bool wait_pending) {
    auto & staging = sched->staging[backend_id];
    ggml_backend_t backend = sched->backends[backend_id];
    if (staging.reserved[slot] || backend->iface.event_record == NULL || staging.events[slot] == NULL) {
        return false;
    }
    const size_t required = ggml_backend_sched_staging_required_capacity(sched, size);
    if (required == SIZE_MAX) {
        return false;
    }
    if (staging.pending[slot]) {
        // Prefetch workers can reserve an already-large-enough slot and wait for its
        // previous H2D event themselves. Reallocation, however, must happen here only
        // after the prior transfer is complete.
        if (!wait_pending && staging.capacities[slot] >= required) {
            return true;
        }
        if (!wait_pending) {
            return false;
        }
        ggml_backend_event_synchronize(staging.events[slot]);
        staging.pending[slot] = false;
    }
    if (staging.capacities[slot] >= required) {
        return true;
    }
    ggml_backend_buffer_free(staging.buffers[slot]);
    staging.buffers[slot] = NULL;
    staging.capacities[slot] = 0;
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    if (host_buft == NULL) {
        return false;
    }
    staging.buffers[slot] = ggml_backend_buft_alloc_buffer(host_buft, required);
    if (staging.buffers[slot] == NULL || !ggml_backend_buffer_is_host(staging.buffers[slot])) {
        ggml_backend_buffer_free(staging.buffers[slot]);
        staging.buffers[slot] = NULL;
        return false;
    }
    staging.capacities[slot] = ggml_backend_buffer_get_size(staging.buffers[slot]);
    ggml_backend_sched_staging_metrics_update(sched, backend_id);
    return staging.capacities[slot] >= required;
}

static int ggml_backend_sched_staging_acquire(
        ggml_backend_sched_t sched, int backend_id, size_t size, bool for_prefetch) {
    auto & staging = sched->staging[backend_id];
    for (int pass = 0; pass < (for_prefetch ? 1 : 2); ++pass) {
        for (int n = 0; n < GGML_BACKEND_SCHED_STAGING_SLOTS; ++n) {
            const int slot = (staging.next + n) % GGML_BACKEND_SCHED_STAGING_SLOTS;
            if (staging.reserved[slot]) {
                continue;
            }
            if (pass == 0 && staging.pending[slot] && !for_prefetch) {
                continue;
            }
            if (!ggml_backend_sched_staging_prepare(sched, backend_id, slot, size, !for_prefetch)) {
                continue;
            }
            staging.reserved[slot] = true;
            staging.next = (slot + 1) % GGML_BACKEND_SCHED_STAGING_SLOTS;
            return slot;
        }
    }
    return -1;
}

static bool ggml_backend_sched_storage_task_exists(
        ggml_backend_sched_t sched, int backend_id, const uint8_t * logical_src, size_t size) {
    for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
        const auto & task = sched->storage_prefetch[backend_id][slot];
        if (task.active && task.logical_src == logical_src && task.size == size) {
            return true;
        }
    }
    return false;
}

static int ggml_backend_sched_storage_prefetch_consume(
        ggml_backend_sched_t sched, int backend_id, const uint8_t * logical_src, size_t size,
        uint8_t ** data) {
    for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
        auto & task = sched->storage_prefetch[backend_id][slot];
        if (!task.active || task.logical_src != logical_src || task.size != size) {
            continue;
        }
        ggml_backend_sched_storage_prefetch_finish(task);
        auto & staging = sched->staging[backend_id];
        // The worker waits on the old upload before reading, so the slot is no
        // longer pending at this point. Keep it reserved until the new H2D is queued.
        staging.pending[slot] = false;
        if (!task.success || task.data_offset > staging.capacities[slot] ||
                size > staging.capacities[slot] - task.data_offset) {
            staging.reserved[slot] = false;
            task.active = false;
            task.success = false;
            return -1;
        }
        *data = (uint8_t *) ggml_backend_buffer_get_base(staging.buffers[slot]) + task.data_offset;
        task.active = false;
        task.success = false;
        return slot;
    }
    return -1;
}

static bool ggml_backend_sched_read_storage_into_slot(
        ggml_backend_sched_t sched, int backend_id, int slot,
        const uint8_t * logical_src, size_t size, uint8_t ** data) {
    auto & staging = sched->staging[backend_id];
    void * base = ggml_backend_buffer_get_base(staging.buffers[slot]);
    size_t data_offset = 0;
    bool ok = false;
    if (sched->weight_read_padded_callback != NULL) {
        ok = sched->weight_read_padded_callback(sched->weight_read_callback_user_data,
            logical_src, base, staging.capacities[slot], size, &data_offset);
    } else if (sched->weight_read_callback != NULL) {
        ok = sched->weight_read_callback(sched->weight_read_callback_user_data,
            logical_src, base, size);
    }
    if (!ok || data_offset > staging.capacities[slot] || size > staging.capacities[slot] - data_offset) {
        return false;
    }
    *data = (uint8_t *) base + data_offset;
    return true;
}

static void ggml_backend_sched_weight_upload_chunked(
        ggml_backend_sched_t sched,
        ggml_backend_t backend,
        int backend_id,
        struct ggml_tensor * dst,
        const void * src,
        size_t offset,
        size_t size,
        bool instrument) {
    const uint8_t * src_bytes = (const uint8_t *) src;
    size_t copied = 0;
    const int64_t upload_start_us = instrument ? ggml_time_us() : 0;
    const ggml_backend_sched_fault_sample faults_before = instrument ? ggml_backend_sched_faults() : ggml_backend_sched_fault_sample{};
    const bool storage_read = sched->weight_read_callback != NULL || sched->weight_read_padded_callback != NULL;
    if (instrument && !storage_read) {
        ggml_backend_sched_readahead(sched, backend_id, src, size);
    }
    const size_t tuned_chunk_size = instrument || storage_read ?
        ggml_backend_sched_transfer_chunk_size(sched, backend_id, size) :
        (size > GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_THRESHOLD ? GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_DEFAULT : size);
    while (copied < size) {
        const size_t chunk = std::min(tuned_chunk_size, size - copied);
        uint8_t * staged_data = NULL;
        int slot = storage_read ? ggml_backend_sched_storage_prefetch_consume(
            sched, backend_id, src_bytes + copied, chunk, &staged_data) : -1;
        if (slot < 0 && (instrument || storage_read)) {
            slot = ggml_backend_sched_staging_acquire(sched, backend_id, chunk, false);
            if (slot >= 0) {
                auto & staging = sched->staging[backend_id];
                if (storage_read) {
                    if (!ggml_backend_sched_read_storage_into_slot(
                            sched, backend_id, slot, src_bytes + copied, chunk, &staged_data)) {
                        staging.reserved[slot] = false;
                        GGML_ASSERT(false && "storage-backed weight callback rejected its logical source");
                    }
                } else {
                    staged_data = (uint8_t *) ggml_backend_buffer_get_base(staging.buffers[slot]);
                    memcpy(staged_data, src_bytes + copied, chunk);
                }
            }
        }
        if (slot >= 0) {
            auto & staging = sched->staging[backend_id];
            ggml_backend_tensor_set_async(backend, dst, staged_data, offset + copied, chunk);
            ggml_backend_event_record(staging.events[slot], backend);
            staging.pending[slot] = true;
            staging.reserved[slot] = false;
            if (instrument) {
                auto & metrics = sched->transient_metrics.backends[backend_id];
                ggml_backend_sched_counter_add(sched, &metrics.staged_upload_chunk_count, 1);
                ggml_backend_sched_counter_add(sched, &metrics.staged_upload_bytes, chunk);
            }
        } else {
            GGML_ASSERT(!storage_read && "storage-backed weights require a scheduler staging slot");
            ggml_backend_tensor_set_async(backend, dst, src_bytes + copied, offset + copied, chunk);
        }
        if (instrument) {
            auto & metrics = sched->transient_metrics.backends[backend_id];
            ggml_backend_sched_counter_add(sched, &metrics.upload_chunk_count, 1);
            ggml_backend_sched_counter_add(sched, &metrics.uploaded_logical_bytes, chunk);
            metrics.max_upload_chunk_bytes = std::max(metrics.max_upload_chunk_bytes, chunk);
        }
        copied += chunk;
    }
    if (instrument) {
        const ggml_backend_sched_fault_sample faults_after = ggml_backend_sched_faults();
        auto & metrics = sched->transient_metrics.backends[backend_id];
        ggml_backend_sched_counter_add(sched, &metrics.mmap_minor_faults,
            faults_after.minor >= faults_before.minor ? faults_after.minor - faults_before.minor : 0);
        ggml_backend_sched_counter_add(sched, &metrics.mmap_major_faults,
            faults_after.major >= faults_before.major ? faults_after.major - faults_before.major : 0);
        ggml_backend_sched_counter_add(sched,
            &metrics.upload_submission_time_us,
            ggml_backend_sched_elapsed_us(upload_start_us));
    }
}

static size_t ggml_backend_sched_weight_window_safety_reserve(size_t total_bytes) {
    const size_t reserve_floor = (size_t) 512 * 1024 * 1024;
    const size_t reserve_tenth = total_bytes / 10 + (total_bytes % 10 != 0);
    return std::max(reserve_floor, reserve_tenth);
}

static bool ggml_backend_sched_weight_window_admit(
        ggml_backend_sched_t sched, int backend_id, size_t request_bytes,
        bool * unknown_memory, bool * live_guard_rejected) {
    *unknown_memory = false;
    *live_guard_rejected = false;
    if (!sched->weight_window_configured[backend_id]) {
        return true;
    }
    if (!sched->weight_window_memory_valid[backend_id]) {
        *unknown_memory = true;
        return false;
    }

    const auto & row = sched->transient_metrics.backends[backend_id];
    const size_t limit = sched->weight_window_limit[backend_id];
    const size_t owned = row.current_transient_bytes + row.current_resident_bytes;
    if (owned < row.current_transient_bytes || request_bytes > limit || owned > limit - request_bytes) {
        return false;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(sched->backends[backend_id]), &free_bytes, &total_bytes);
    const size_t reserve = sched->weight_window_safety_reserve[backend_id];
    if (free_bytes == 0 || total_bytes == 0 || free_bytes > total_bytes || total_bytes != row.weight_window_total_bytes ||
            free_bytes <= reserve || request_bytes > free_bytes - reserve) {
        *live_guard_rejected = true;
        return false;
    }
    return true;
}

static void ggml_backend_sched_resident_metrics_update(ggml_backend_sched_t sched, int backend_id) {
    auto & row = sched->transient_metrics.backends[backend_id];
    size_t current_bytes = 0;
    size_t current_records = 0;
    for (int i = 0; i < sched->n_backends; ++i) {
        current_bytes += sched->transient_metrics.backends[i].current_resident_bytes;
        current_records += sched->transient_metrics.backends[i].current_resident_records;
    }
    sched->transient_metrics.current_resident_bytes = current_bytes;
    sched->transient_metrics.current_resident_records = current_records;
    row.peak_resident_bytes = std::max(row.peak_resident_bytes, row.current_resident_bytes);
    row.peak_resident_records = std::max(row.peak_resident_records, row.current_resident_records);
    row.peak_manually_owned_bytes = std::max(row.peak_manually_owned_bytes,
        row.current_resident_bytes + row.current_transient_bytes);
    sched->transient_metrics.peak_resident_bytes = std::max(sched->transient_metrics.peak_resident_bytes, current_bytes);
    sched->transient_metrics.peak_resident_records = std::max(sched->transient_metrics.peak_resident_records, current_records);
}

static void ggml_backend_sched_evict_resident(
        ggml_backend_sched_t sched,
        ggml_backend_sched_resident_map::iterator it) {
    ggml_backend_sched_resident resident = it->second;
    GGML_ASSERT(!resident.executing);
    ggml_backend_synchronize(sched->backends[resident.backend_id]);
    if (resident.copy != NULL && resident.copy->buffer == resident.buffer) {
        resident.copy->buffer = NULL;
        resident.copy->data = NULL;
    }
    auto & row = sched->transient_metrics.backends[resident.backend_id];
    GGML_ASSERT(row.current_resident_records > 0 && row.current_resident_bytes >= resident.allocation_size);
    row.current_resident_records--;
    row.current_resident_bytes -= resident.allocation_size;
    sched->residents->erase(it);
    ggml_backend_buffer_free(resident.buffer);
    ggml_backend_sched_counter_add(sched, &row.residency_eviction_count, 1);
    ggml_backend_sched_counter_add(sched, &row.compact_expert_eviction_count,
        std::count_if(resident.experts.begin(), resident.experts.end(), [](int32_t expert) { return expert >= 0; }));
    ggml_backend_sched_resident_metrics_update(sched, resident.backend_id);
}

static void ggml_backend_sched_drain_residents(ggml_backend_sched_t sched) {
    while (sched->residents != NULL && !sched->residents->empty()) {
        auto it = sched->residents->begin();
        it->second.executing = false;
        const int backend_id = it->second.backend_id;
        ggml_backend_sched_evict_resident(sched, it);
        ggml_backend_sched_counter_add(sched, &sched->transient_metrics.backends[backend_id].residency_drain_count, 1);
    }
}

static bool ggml_backend_sched_make_resident_space(
        ggml_backend_sched_t sched, int backend_id, size_t request) {
    const size_t window = sched->weight_window_limit[backend_id];
    if (!sched->weight_window_configured[backend_id] || !sched->weight_window_memory_valid[backend_id] ||
            request == 0 || request > window) {
        return false;
    }
    while (true) {
        bool unknown = false;
        bool live_rejected = false;
        if (ggml_backend_sched_weight_window_admit(
                sched, backend_id, request, &unknown, &live_rejected)) {
            return true;
        }
        if (unknown) {
            return false;
        }

        // Dense weights and expert slabs share one global budget. Frequency first
        // makes repeated dense weights scan-resistant, while completed-use breaks
        // ties in LRU order for the changing MoE active set. Executing entries are
        // part of the current working set and are never eviction candidates.
        auto victim = sched->residents->end();
        for (auto it = sched->residents->begin(); it != sched->residents->end(); ++it) {
            if (it->second.backend_id != backend_id || it->second.executing) {
                continue;
            }
            if (victim == sched->residents->end() || it->second.frequency < victim->second.frequency ||
                    (it->second.frequency == victim->second.frequency &&
                     it->second.completed_use < victim->second.completed_use)) {
                victim = it;
            }
        }
        if (victim == sched->residents->end()) {
            return false;
        }
        ggml_backend_sched_evict_resident(sched, victim);
    }
}

static bool ggml_backend_sched_grow_expert_slab(
        ggml_backend_sched_t sched,
        int backend_id,
        struct ggml_tensor * layout,
        ggml_backend_sched_resident_map::iterator slab_it,
        size_t active_slots,
        size_t max_slots) {
    auto & slab = slab_it->second;
    if (slab.executing || slab.experts.empty() || active_slots <= slab.experts.size() || max_slots == 0) {
        return active_slots <= slab.experts.size();
    }

    const size_t old_slots = slab.experts.size();
    size_t target_slots = std::max(active_slots, old_slots <= SIZE_MAX / 2 ? old_slots * 2 : max_slots);
    target_slots = std::min(target_slots, max_slots);
    if (target_slots <= old_slots || target_slots > (size_t) INT64_MAX) {
        return false;
    }

    const int64_t saved_ne2 = layout->ne[2];
    layout->ne[2] = (int64_t) target_slots;
    const size_t new_alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[backend_id], layout);
    layout->ne[2] = saved_ne2;
    if (new_alloc_size == 0 || new_alloc_size <= slab.allocation_size) {
        return false;
    }

    slab.executing = true;
    if (!ggml_backend_sched_make_resident_space(sched, backend_id, new_alloc_size)) {
        slab.executing = false;
        return false;
    }

    ggml_backend_buffer_t new_buffer = ggml_backend_buft_alloc_buffer(sched->bufts[backend_id], new_alloc_size);
    if (new_buffer == NULL) {
        slab.executing = false;
        return false;
    }

    // Copy only the initialized prefix corresponding to the old compact slots.
    // Both temporary descriptors have exactly the same layout, so the backend can
    // use its native D2D path. They do not own either backend buffer.
    layout->ne[2] = (int64_t) old_slots;
    struct ggml_tensor * old_view = ggml_dup_tensor_layout(sched->ctx, layout);
    struct ggml_tensor * new_view = ggml_dup_tensor_layout(sched->ctx, layout);
    layout->ne[2] = saved_ne2;
    old_view->flags = (enum ggml_tensor_flag) (old_view->flags | GGML_TENSOR_FLAG_NO_ALLOC);
    new_view->flags = (enum ggml_tensor_flag) (new_view->flags | GGML_TENSOR_FLAG_NO_ALLOC);

    const enum ggml_status old_ec = ggml_backend_tensor_alloc(
        slab.buffer, old_view, ggml_backend_buffer_get_base(slab.buffer));
    const enum ggml_status new_ec = ggml_backend_tensor_alloc(
        new_buffer, new_view, ggml_backend_buffer_get_base(new_buffer));
    if (old_ec != GGML_STATUS_SUCCESS || new_ec != GGML_STATUS_SUCCESS) {
        old_view->buffer = NULL;
        old_view->data = NULL;
        new_view->buffer = NULL;
        new_view->data = NULL;
        ggml_backend_buffer_free(new_buffer);
        slab.executing = false;
        return false;
    }

    ggml_backend_t backend = sched->backends[backend_id];
    ggml_backend_tensor_copy_async(backend, backend, old_view, new_view);
    ggml_backend_synchronize(backend);

    old_view->buffer = NULL;
    old_view->data = NULL;
    new_view->buffer = NULL;
    new_view->data = NULL;

    auto & row = sched->transient_metrics.backends[backend_id];
    GGML_ASSERT(row.current_resident_bytes >= slab.allocation_size);
    row.current_resident_bytes -= slab.allocation_size;
    row.current_resident_bytes += new_alloc_size;

    ggml_backend_buffer_t old_buffer = slab.buffer;
    slab.buffer = new_buffer;
    slab.allocation_size = new_alloc_size;
    slab.copy = NULL;
    slab.experts.resize(target_slots, -1);
    slab.expert_slots.resize(target_slots, -1);
    slab.expert_frequency.resize(target_slots, 0);
    slab.expert_completed_use.resize(target_slots, 0);
    slab.executing = false;
    slab.completed_use = ++sched->residency_use_clock;
    ggml_backend_buffer_free(old_buffer);
    ggml_backend_sched_resident_metrics_update(sched, backend_id);

    GGML_LOG_DEBUG("sequential compact slab grow: backend=%s old_slots=%zu new_slots=%zu old_bytes=%zu new_bytes=%zu\n",
        ggml_backend_name(backend), old_slots, target_slots,
        old_view ? ggml_nbytes(old_view) : 0, new_alloc_size);
    return true;
}

static bool ggml_backend_sched_ledger_valid(ggml_backend_sched_t sched, int backend_id) {
    size_t bytes = 0;
    size_t records = 0;
    for (int i = 0; i < sched->n_backends; ++i) {
        const auto & row = sched->transient_metrics.backends[i];
        if (SIZE_MAX - bytes < row.current_transient_bytes || SIZE_MAX - records < row.current_transient_records) {
            return false;
        }
        bytes += row.current_transient_bytes;
        records += row.current_transient_records;
    }
    const auto & row = sched->transient_metrics.backends[backend_id];
    return bytes == sched->transient_bytes && records == sched->transient_count &&
        sched->transient_metrics.current_transient_bytes == sched->transient_bytes &&
        sched->transient_metrics.current_transient_records == sched->transient_count &&
        row.current_transient_bytes <= sched->transient_bytes && row.current_transient_records <= sched->transient_count;
}

static void ggml_backend_sched_ledger_assert(ggml_backend_sched_t sched, int backend_id, bool condition) {
    if (!condition || !ggml_backend_sched_ledger_valid(sched, backend_id)) {
        ggml_backend_sched_counter_add(sched, &sched->transient_metrics.ledger_mismatch_count, 1);
        GGML_ASSERT(false && "scheduler transient ledger mismatch");
    }
}

static void ggml_backend_sched_ledger_enter(ggml_backend_sched_t sched, int backend_id, size_t size) {
    auto & row = sched->transient_metrics.backends[backend_id];
    const bool valid = size > 0 && SIZE_MAX - sched->transient_bytes >= size && sched->transient_count < SIZE_MAX &&
        SIZE_MAX - row.current_transient_bytes >= size && row.current_transient_records < SIZE_MAX;
    if (!valid) {
        ggml_backend_sched_counter_add(sched, &sched->transient_metrics.ledger_mismatch_count, 1);
        GGML_ASSERT(false && "scheduler transient ledger overflow");
    }
    sched->transient_bytes += size;
    sched->transient_count++;
    row.current_transient_bytes += size;
    row.current_transient_records++;
    sched->transient_metrics.current_transient_bytes = sched->transient_bytes;
    sched->transient_metrics.current_transient_records = sched->transient_count;
    row.peak_transient_bytes = std::max(row.peak_transient_bytes, row.current_transient_bytes);
    row.peak_transient_records = std::max(row.peak_transient_records, row.current_transient_records);
    row.peak_manually_owned_bytes = std::max(row.peak_manually_owned_bytes,
        row.current_resident_bytes + row.current_transient_bytes);
    sched->transient_metrics.peak_transient_bytes = std::max(sched->transient_metrics.peak_transient_bytes, sched->transient_bytes);
    sched->transient_metrics.peak_transient_records = std::max(sched->transient_metrics.peak_transient_records, sched->transient_count);
    ggml_backend_sched_ledger_assert(sched, backend_id, true);
}

static void ggml_backend_sched_ledger_leave(ggml_backend_sched_t sched, int backend_id, size_t size) {
    auto & row = sched->transient_metrics.backends[backend_id];
    ggml_backend_sched_ledger_assert(sched, backend_id,
        sched->transient_count > 0 && sched->transient_bytes >= size &&
        row.current_transient_records > 0 && row.current_transient_bytes >= size);
    sched->transient_count--;
    sched->transient_bytes -= size;
    row.current_transient_records--;
    row.current_transient_bytes -= size;
    sched->transient_metrics.current_transient_bytes = sched->transient_bytes;
    sched->transient_metrics.current_transient_records = sched->transient_count;
    ggml_backend_sched_ledger_assert(sched, backend_id, true);
}

static void ggml_backend_sched_release_transients(
        ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, bool synchronize,
        enum ggml_backend_sched_transient_drain_reason reason, bool compute_submitted) {
    bool has_live = false;
    bool has_compact = false;
    for (int i = 0; i < split->n_inputs; ++i) {
        has_live = has_live || split->transient_buffers[i] != NULL;
        has_compact = has_compact || split->input_compact_moe[i] || split->compact_ids_buffer[i] != NULL;
    }
    if (!has_live && !has_compact) {
        return;
    }
    auto & row = sched->transient_metrics.backends[split->backend_id];
    const int64_t drain_start_us = ggml_time_us();
    if (synchronize) {
        const int64_t wait_start_us = ggml_time_us();
        ggml_backend_synchronize(sched->backends[split->backend_id]);
        if (compute_submitted) {
            ggml_backend_sched_counter_add(sched, &row.compute_completion_wait_count, 1);
            ggml_backend_sched_counter_add(sched, &row.compute_completion_wait_us, ggml_backend_sched_elapsed_us(wait_start_us));
        }
    }
    for (int i = split->n_inputs - 1; i >= 0; --i) {
        if (split->input_resident[i]) {
            continue;
        }
        ggml_backend_buffer_t buffer = split->transient_buffers[i];
        if (buffer == NULL) {
            continue;
        }
        struct ggml_tensor * copy = tensor_copy(split->inputs[i], split->backend_id, 0);
        GGML_ASSERT(copy != NULL && (copy->buffer == NULL || copy->buffer == buffer));
        if (copy->buffer == buffer) {
            copy->buffer = NULL;
            copy->data = NULL;
        } else {
            GGML_ASSERT(copy->data == NULL);
        }
        split->transient_buffers[i] = NULL;
        ggml_backend_sched_ledger_leave(sched, split->backend_id, split->transient_sizes[i]);
        split->transient_sizes[i] = 0;
        ggml_backend_buffer_free(buffer);
    }
    for (int i = 0; i < split->n_inputs; ++i) {
        ggml_backend_sched_compact_reset_input(split, i, true);
    }
    ggml_backend_sched_counter_add(sched, &row.drain_count[reason], 1);
    ggml_backend_sched_counter_add(sched, &row.drain_time_us[reason], ggml_backend_sched_elapsed_us(drain_start_us));
}

static void ggml_backend_sched_drain_transients(
        ggml_backend_sched_t sched, enum ggml_backend_sched_transient_drain_reason reason) {
    for (int i = 0; i < sched->n_splits; ++i) {
        ggml_backend_sched_release_transients(sched, &sched->splits[i], true, reason, false);
    }
    if (sched->transient_count != 0 || sched->transient_bytes != 0) {
        ggml_backend_sched_counter_add(sched, &sched->transient_metrics.ledger_mismatch_count, 1);
        GGML_ASSERT(false && "scheduler transient ledger not empty after drain");
    }
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

static int ggml_backend_sched_tensor_layer(const struct ggml_tensor * tensor) {
    const char * marker = strstr(tensor->name, "blk.");
    if (marker == NULL) {
        return -1;
    }
    marker += 4;
    if (*marker < '0' || *marker > '9') {
        return -1;
    }
    int layer = 0;
    while (*marker >= '0' && *marker <= '9') {
        if (layer > (INT_MAX - (*marker - '0')) / 10) {
            return -1;
        }
        layer = layer * 10 + (*marker++ - '0');
    }
    return layer;
}

static int ggml_backend_sched_op_layer(const struct ggml_tensor * op) {
    int layer = ggml_backend_sched_tensor_layer(op);
    for (int i = 0; i < GGML_MAX_SRC && layer < 0; ++i) {
        if (op->src[i] != NULL) {
            layer = ggml_backend_sched_tensor_layer(op->src[i]);
        }
    }
    return layer;
}

static int ggml_backend_sched_forced_weight_backend(
        ggml_backend_sched_t sched, const struct ggml_tensor * weight, const struct ggml_tensor * op, int cpu_backend_id) {
    int preferred = 0;
    const int layer = ggml_backend_sched_tensor_layer(weight);
    // Keep attention and dense activation chains on the primary GPU. At large
    // contexts, moving a complete layer range duplicates the full prompt
    // compute arena on a smaller secondary GPU. MoE expert matmuls have a much
    // smaller activation footprint and dominate streamed weight traffic, so
    // distribute only those operations by layer range.
    if ((op->op == GGML_OP_MUL_MAT_ID || sched->force_weight_offload_token_generation) &&
            sched->force_weight_offload_split_configured &&
            layer >= 0 && sched->force_weight_offload_max_layer >= 0) {
        // Interleave the weighted assignment instead of using one contiguous
        // range per GPU. Transformer layers are sequentially dependent, so a
        // contiguous range leaves every secondary GPU idle for most of each
        // token. The irrational stride spreads each device's configured share
        // uniformly through the layer stack without changing that share.
        constexpr float golden_ratio_conjugate = 0.61803398875f;
        const float position = fmodf((layer + 0.5f) * golden_ratio_conjugate, 1.0f);
        while (preferred + 1 < cpu_backend_id && position > sched->force_weight_offload_split[preferred]) {
            preferred++;
        }
    }

    if (preferred < cpu_backend_id && ggml_backend_supports_op(sched->backends[preferred], op)) {
        return preferred;
    }
    for (int b = 0; b < cpu_backend_id; ++b) {
        if (ggml_backend_supports_op(sched->backends[b], op)) {
            return b;
        }
    }
    return cpu_backend_id;
}

static void ggml_backend_sched_log_forced_split_summary(ggml_backend_sched_t sched) {
    if (!sched->force_weight_offload_split_configured || sched->force_weight_offload_max_layer < 0) {
        return;
    }
    int counts[GGML_SCHED_MAX_BACKENDS] = {};
    for (int layer = 0; layer <= sched->force_weight_offload_max_layer; ++layer) {
        constexpr float golden_ratio_conjugate = 0.61803398875f;
        const float position = fmodf((layer + 0.5f) * golden_ratio_conjugate, 1.0f);
        int backend_id = 0;
        while (backend_id + 1 < sched->n_backends - 1 && position > sched->force_weight_offload_split[backend_id]) {
            backend_id++;
        }
        counts[backend_id]++;
    }
    GGML_LOG_INFO("sequential MoE layer assignment:");
    for (int i = 0; i < sched->n_backends - 1; ++i) {
        GGML_LOG_INFO(" %s=%d", ggml_backend_name(sched->backends[i]), counts[i]);
    }
    GGML_LOG_INFO("\n");
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                if (sched->force_weight_offload) {
                    const int backend_id = ggml_backend_sched_forced_weight_backend(sched, src, tensor, src_backend_id);
                    if (backend_id != src_backend_id) {
                        SET_CAUSE(tensor, "1.off");
                        return backend_id;
                    }
                } else {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (sched->op_offload && ggml_backend_offload_op(sched->backends[b], tensor) &&
                                ggml_backend_supports_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

static bool ggml_backend_sched_split_input_is_moe(const struct ggml_backend_sched_split * split, int input_id, const struct ggml_tensor * input_cpy) {
    if (input_id != 0 || split->graph.n_nodes == 0) {
        return false;
    }

    const ggml_tensor * node = split->graph.nodes[0];
    return node->op == GGML_OP_MUL_MAT_ID && node->src[0] == input_cpy;
}

static bool ggml_backend_sched_split_input_can_prefetch(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split, int input_id) {
    const int split_backend_id = split->backend_id;
    const ggml_backend_t split_backend = sched->backends[split_backend_id];
    const ggml_backend_t prefetch_backend = sched->prefetch_backends[split_backend_id];
    const ggml_backend_event_t prefetch_event = sched->prefetch_events[split_backend_id][0];
    const ggml_backend_event_t prefetch_event_alt = sched->prefetch_events[split_backend_id][1];

    if (!sched->async_weight_prefetch || sched->weight_read_callback != NULL || sched->weight_read_padded_callback != NULL ||
            prefetch_backend == NULL || prefetch_event == NULL || prefetch_event_alt == NULL ||
            split_backend->iface.event_wait == NULL || split->inputs_allocated[input_id]) {
        return false;
    }

    struct ggml_tensor * input = split->inputs[input_id];
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

    return input->buffer != NULL && input_cpy != NULL &&
        ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(input->buffer) && !ggml_backend_buft_is_host(sched->bufts[split_backend_id]) &&
        !ggml_backend_sched_split_input_is_moe(split, input_id, input_cpy);
}

static bool ggml_backend_sched_split_input_was_prefetched(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split, int input_id) {
    GGML_UNUSED(sched);
    return split->input_prefetched[input_id];
}

static void ggml_backend_sched_init_prefetch_backend(ggml_backend_sched_t sched, int backend_id) {
    if (!sched->async_weight_prefetch || sched->prefetch_backends[backend_id] != NULL) {
        return;
    }

    ggml_backend_t backend = sched->backends[backend_id];
    if (ggml_backend_dev_type(backend->device) == GGML_BACKEND_DEVICE_TYPE_CPU || backend->iface.set_tensor_async == NULL) {
        return;
    }

    ggml_backend_t prefetch_backend = ggml_backend_dev_init(backend->device, NULL);
    if (prefetch_backend == NULL) {
        return;
    }
    if (prefetch_backend->iface.set_tensor_async == NULL || prefetch_backend->iface.event_record == NULL) {
        ggml_backend_free(prefetch_backend);
        return;
    }

    ggml_backend_event_t event_0 = ggml_backend_event_new(backend->device);
    ggml_backend_event_t event_1 = ggml_backend_event_new(backend->device);
    if (event_0 == NULL || event_1 == NULL) {
        ggml_backend_event_free(event_0);
        ggml_backend_event_free(event_1);
        ggml_backend_free(prefetch_backend);
        return;
    }

    sched->prefetch_backends[backend_id] = prefetch_backend;
    sched->prefetch_events[backend_id][0] = event_0;
    sched->prefetch_events[backend_id][1] = event_1;
}

static bool ggml_backend_sched_prefetch_resident_transient_input(
        ggml_backend_sched_t sched,
        struct ggml_backend_sched_split * split,
        int input_id,
        ggml_backend_t prefetch_backend) {
    const int backend_id = split->backend_id;
    if (!split->input_transient[input_id] || split->input_prefetched[input_id] ||
            split->transient_buffers[input_id] != NULL) {
        return false;
    }

    struct ggml_tensor * input = split->inputs[input_id];
    struct ggml_tensor * input_cpy = tensor_copy(input, backend_id, sched->cur_copy);
    const bool full_moe_prefetch = split->input_full_moe_prefetch[input_id];
    const bool is_moe = input_cpy != NULL && ggml_backend_sched_split_input_is_moe(split, input_id, input_cpy);
    if (input == NULL || input_cpy == NULL || input->data == NULL || input->view_src != NULL ||
            input_cpy->buffer != NULL || input_cpy->data != NULL ||
            (is_moe && !full_moe_prefetch) ||
            input->buffer == NULL || ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_backend_buffer_is_host(input->buffer) || ggml_backend_buft_is_host(sched->bufts[backend_id])) {
        return false;
    }

    // Full-layer MoE prefetch is intentionally ephemeral.  Caching the entire
    // expert tensor would compete with the compact hot-expert cache used by
    // sequential mode and can consume several GiB per layer.
    const bool cache_eligible = sched->residency_enabled[backend_id] && !full_moe_prefetch;
    const std::vector<int32_t> empty_experts;
    const auto resident_key = ggml_backend_sched_resident_key_make(input, backend_id, empty_experts);
    if (cache_eligible && sched->residents->find(resident_key) != sched->residents->end()) {
        return false;
    }

    const size_t alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[backend_id], input_cpy);
    if (alloc_size == 0) {
        return false;
    }
    const size_t split_limit = sched->max_weight_bytes_per_split[backend_id];
    size_t split_bytes = 0;
    for (int i = 0; i < split->n_inputs; ++i) {
        if (SIZE_MAX - split_bytes < split->transient_sizes[i]) {
            return false;
        }
        split_bytes += split->transient_sizes[i];
    }
    if (split_limit > 0 && (alloc_size > split_limit || split_bytes > split_limit - alloc_size)) {
        return false;
    }

    bool unknown_memory = false;
    bool live_guard_rejected = false;
    // Never evict from the early-prefetch path. Eviction synchronizes the
    // compute stream and would turn the intended overlap back into serialization.
    if (!ggml_backend_sched_weight_window_admit(
            sched, backend_id, alloc_size, &unknown_memory, &live_guard_rejected)) {
        return false;
    }

    auto & metrics = sched->transient_metrics.backends[backend_id];
    const int64_t allocation_start_us = ggml_time_us();
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(sched->bufts[backend_id], alloc_size);
    ggml_backend_sched_counter_add(sched, &metrics.allocation_time_us,
        ggml_backend_sched_elapsed_us(allocation_start_us));
    if (buffer == NULL) {
        return false;
    }
    if (ggml_backend_tensor_alloc(buffer, input_cpy, ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        input_cpy->buffer = NULL;
        input_cpy->data = NULL;
        return false;
    }

    split->transient_buffers[input_id] = buffer;
    split->transient_sizes[input_id] = alloc_size;
    split->input_resident[input_id] = cache_eligible;
    split->input_resident_hit[input_id] = false;

    ggml_backend_sched_counter_add(sched, &metrics.allocation_requested_bytes, alloc_size);
    ggml_backend_sched_counter_add(sched, &metrics.allocation_admitted_bytes, alloc_size);
    ggml_backend_sched_counter_add(sched, &metrics.allocation_count, 1);

    if (cache_eligible) {
        ggml_backend_sched_resident resident{};
        resident.source = input;
        resident.source_buffer = input->buffer;
        resident.source_data = input->data;
        resident.logical_size = ggml_nbytes(input);
        resident.backend_id = backend_id;
        resident.copy = input_cpy;
        resident.buffer = buffer;
        resident.allocation_size = alloc_size;
        resident.completed_use = ++sched->residency_use_clock;
        resident.frequency = 1;
        resident.executing = true;
        resident.expert_tier = false;
        sched->residents->emplace(resident_key, resident);
        metrics.current_resident_bytes += alloc_size;
        metrics.current_resident_records++;
        ggml_backend_sched_counter_add(sched, &metrics.residency_miss_count, 1);
        ggml_backend_sched_resident_metrics_update(sched, backend_id);
    } else {
        ggml_backend_sched_ledger_enter(sched, backend_id, alloc_size);
    }

    // For mmap-backed Fable prefetch, submit directly from the registered mmap
    // pages. Sequential storage-backed weights retain the pinned staging path.
    ggml_backend_sched_weight_upload_chunked(sched, prefetch_backend, backend_id,
        input_cpy, input->data, 0, ggml_nbytes(input_cpy), !full_moe_prefetch);
    ggml_backend_sched_counter_add(sched, &metrics.upload_count, 1);
    ggml_backend_sched_counter_add(sched, &metrics.uploaded_backend_bytes, alloc_size);
    if (cache_eligible) {
        ggml_backend_sched_counter_add(sched, &metrics.residency_upload_count, 1);
    }
    if (!sched->transient_sources_seen->insert(input).second) {
        ggml_backend_sched_counter_add(sched, &metrics.shared_reload_count, 1);
    }

    split->input_prefetched[input_id] = true;
    return true;
}

static void ggml_backend_sched_prefetch_split_inputs(ggml_backend_sched_t sched, int split_id) {
    if (!sched->async_weight_prefetch || split_id <= 0 || split_id >= sched->n_splits) {
        return;
    }

    struct ggml_backend_sched_split * split = &sched->splits[split_id];
    const int split_backend_id = split->backend_id;
    if (split->has_prefetched_inputs) {
        return;
    }

    ggml_backend_sched_init_prefetch_backend(sched, split_backend_id);

    ggml_backend_t prefetch_backend = sched->prefetch_backends[split_backend_id];
    ggml_backend_event_t prefetch_event = sched->prefetch_events[split_backend_id][split_id & 1];
    if (prefetch_backend == NULL || prefetch_event == NULL) {
        return;
    }

    bool prefetched = false;
    for (int input_id = 0; input_id < split->n_inputs; input_id++) {
        if (split->input_transient[input_id]) {
            prefetched = ggml_backend_sched_prefetch_resident_transient_input(
                sched, split, input_id, prefetch_backend) || prefetched;
            continue;
        }

        if (!ggml_backend_sched_split_input_can_prefetch(sched, split, input_id)) {
            continue;
        }

        struct ggml_tensor * input = split->inputs[input_id];
        if (input->view_src != NULL || tensor_backend_id(input) != sched->n_backends - 1) {
            continue;
        }

        struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
        if (input->data == NULL || input_cpy == NULL || input_cpy->buffer == NULL || ggml_backend_buffer_is_host(input_cpy->buffer)) {
            continue;
        }

        ggml_backend_tensor_set_async(prefetch_backend, input_cpy, input->data, 0, ggml_nbytes(input_cpy));
        split->input_prefetched[input_id] = true;
        prefetched = true;
    }

    if (prefetched) {
        ggml_backend_event_record(prefetch_event, prefetch_backend);
        split->has_prefetched_inputs = true;
    }
}

static int ggml_backend_sched_storage_prefetch_active_count(ggml_backend_sched_t sched, int backend_id) {
    int count = 0;
    for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
        count += sched->storage_prefetch[backend_id][slot].active ? 1 : 0;
    }
    return count;
}

static void ggml_backend_sched_prefetch_storage_inputs(ggml_backend_sched_t sched, int start_split_id) {
    if (!sched->async_weight_prefetch ||
            (sched->weight_read_callback == NULL && sched->weight_read_padded_callback == NULL) ||
            start_split_id < 0 || start_split_id >= sched->n_splits) {
        return;
    }

    for (int split_id = start_split_id; split_id < sched->n_splits; ++split_id) {
        auto * split = &sched->splits[split_id];
        const int backend_id = split->backend_id;
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            ggml_tensor * input = split->inputs[input_id];
            ggml_tensor * input_cpy = tensor_copy(input, backend_id, 0);
            if (!split->input_transient[input_id] ||
                    ggml_backend_sched_split_input_is_moe(split, input_id, input_cpy) ||
                    input == NULL || input->data == NULL || input_cpy == NULL) {
                continue;
            }

            const size_t total = ggml_nbytes(input_cpy);
            const size_t chunk_size = ggml_backend_sched_transfer_chunk_size(sched, backend_id, total);
            const int max_active = ggml_backend_sched_transfer_prefetch_depth(
                sched, backend_id, chunk_size, total);
            if (ggml_backend_sched_storage_prefetch_active_count(sched, backend_id) >= max_active) {
                continue;
            }

            for (size_t copied = 0; copied < total; copied += chunk_size) {
                if (ggml_backend_sched_storage_prefetch_active_count(sched, backend_id) >= max_active) {
                    break;
                }
                const size_t size = std::min(chunk_size, total - copied);
                const uint8_t * logical_src = (const uint8_t *) input->data + copied;
                if (ggml_backend_sched_storage_task_exists(sched, backend_id, logical_src, size)) {
                    continue;
                }
                const int slot = ggml_backend_sched_staging_acquire(sched, backend_id, size, true);
                if (slot < 0) {
                    break;
                }
                auto & task = sched->storage_prefetch[backend_id][slot];
                task.logical_src = logical_src;
                task.size = size;
                task.data_offset = 0;
                task.split_id = split_id;
                task.input_id = input_id;
                task.active = true;
                task.success = false;
                task.worker = std::thread([sched, backend_id, slot]() {
                    auto & staging = sched->staging[backend_id];
                    auto & task = sched->storage_prefetch[backend_id][slot];
                    if (staging.pending[slot]) {
                        ggml_backend_event_synchronize(staging.events[slot]);
                    }
                    void * base = ggml_backend_buffer_get_base(staging.buffers[slot]);
                    size_t data_offset = 0;
                    if (sched->weight_read_padded_callback != NULL) {
                        task.success = sched->weight_read_padded_callback(
                            sched->weight_read_callback_user_data, task.logical_src,
                            base, staging.capacities[slot], task.size, &data_offset);
                    } else {
                        task.success = sched->weight_read_callback(
                            sched->weight_read_callback_user_data, task.logical_src, base, task.size);
                    }
                    task.data_offset = data_offset;
                });
            }
        }
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    ggml_backend_sched_drain_transients(sched, GGML_BACKEND_SCHED_TRANSIENT_DRAIN_GRAPH_REBUILD);
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        for (int input_id = 0; input_id < sched->splits[split_id].n_inputs; ++input_id) {
            ggml_backend_sched_compact_reset_input(&sched->splits[split_id], input_id, true);
        }
    }
    // Copy descriptors are graph-context objects, but resident payloads are
    // backend buffers owned by the scheduler. Detach descriptors before the
    // graph context is rebuilt and reattach matching payloads below. This is
    // essential for token generation, whose graph descriptors are rebuilt on
    // every decode even though the mmap-backed source weights are unchanged.
    if (sched->persistent_weight_residency) {
        for (auto & entry : *sched->residents) {
            entry.second.copy = NULL;
            entry.second.executing = false;
        }
    } else {
        ggml_backend_sched_drain_residents(sched);
    }
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    sched->force_weight_offload_max_layer = -1;
    sched->force_weight_offload_token_generation = false;
    if (sched->force_weight_offload && sched->force_weight_offload_split_configured) {
        // Generation graphs have one row of activations per sequence. Their
        // per-layer activation arena is small enough to place complete layers
        // on a secondary GPU, unlike the full prompt-processing graph.
        bool has_layer_activation = false;
        sched->force_weight_offload_token_generation = true;
        for (int i = 0; i < graph->n_nodes; ++i) {
            const struct ggml_tensor * node = graph->nodes[i];
            if (node->op == GGML_OP_NONE || ggml_backend_sched_op_layer(node) < 0) {
                continue;
            }
            has_layer_activation = true;
            if (node->ne[1] > 1) {
                sched->force_weight_offload_token_generation = false;
                break;
            }
        }
        sched->force_weight_offload_token_generation &= has_layer_activation;
        GGML_LOG_DEBUG("sequential forced offload graph mode: %s\n",
            sched->force_weight_offload_token_generation ? "token-generation full-layer" : "prompt MoE-only");
        for (int i = 0; i < graph->n_leafs; ++i) {
            sched->force_weight_offload_max_layer = std::max(
                sched->force_weight_offload_max_layer, ggml_backend_sched_tensor_layer(graph->leafs[i]));
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (graph->nodes[i]->src[j] != NULL) {
                    sched->force_weight_offload_max_layer = std::max(
                        sched->force_weight_offload_max_layer, ggml_backend_sched_tensor_layer(graph->nodes[i]->src[j]));
                }
            }
        }
        ggml_backend_sched_log_forced_split_summary(sched);
    }

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    if (sched->force_weight_offload) {
        const int cpu_backend_id = sched->n_backends - 1;
        for (int i = 0; i < graph->n_nodes; ++i) {
            struct ggml_tensor * node = graph->nodes[i];
            if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] == NULL || node->src[2] == NULL) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id < 0 || *node_backend_id == cpu_backend_id) {
                continue;
            }
            const ggml_tensor * weights = node->src[0];
            const ggml_tensor * ids = node->src[2];
            if (weights->buffer == NULL || !ggml_backend_buffer_is_host(weights->buffer) ||
                    weights->ne[2] <= 1 || weights->nb[2] == 0 || ids->ne[0] <= 0 || ids->ne[1] <= 0) {
                continue;
            }
            const uint64_t n_ids = (uint64_t) ids->ne[0] > UINT64_MAX / (uint64_t) ids->ne[1] ?
                UINT64_MAX : (uint64_t) ids->ne[0] * (uint64_t) ids->ne[1];
            uint64_t n_active = std::min<uint64_t>((uint64_t) weights->ne[2], n_ids);
            if (ids->data != NULL && (ids->buffer == NULL || ggml_backend_buffer_is_host(ids->buffer))) {
                std::vector<bool> active((size_t) weights->ne[2], false);
                uint64_t unique = 0;
                bool valid = true;
                for (int64_t i1 = 0; valid && i1 < ids->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
                        const int32_t id = *(const int32_t *) ((const char *) ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
                        if (id < 0 || id >= weights->ne[2]) {
                            valid = false;
                            break;
                        }
                        if (!active[(size_t) id]) {
                            active[(size_t) id] = true;
                            unique++;
                        }
                    }
                }
                if (valid) {
                    n_active = unique;
                }
            }
            const uint64_t estimate = n_active > UINT64_MAX / weights->nb[2] ?
                UINT64_MAX : n_active * weights->nb[2];
            auto backend_limit = [&](int backend_id) -> size_t {
                return sched->weight_window_configured[backend_id] ?
                    sched->weight_window_limit[backend_id] : sched->max_weight_bytes_per_split[backend_id];
            };
            const size_t selected_limit = backend_limit(*node_backend_id);
            if (selected_limit == 0 || estimate <= selected_limit) {
                continue;
            }

            int overflow_backend = -1;
            for (int candidate = 0; candidate < cpu_backend_id; ++candidate) {
                if (candidate == *node_backend_id || !ggml_backend_supports_op(sched->backends[candidate], node)) {
                    continue;
                }
                const size_t limit = backend_limit(candidate);
                if (limit > 0 && estimate <= limit) {
                    overflow_backend = candidate;
                    break;
                }
            }
            if (overflow_backend < 0 && ggml_backend_supports_op(sched->backends[cpu_backend_id], node)) {
                overflow_backend = cpu_backend_id;
            }
            if (overflow_backend >= 0) {
                GGML_LOG_DEBUG("sequential MoE overflow: %s estimated active weights=%" PRIu64
                    " bytes exceed backend %s limit=%zu; routing to %s\n",
                    node->name, estimate, ggml_backend_name(sched->backends[*node_backend_id]), selected_limit,
                    ggml_backend_name(sched->backends[overflow_backend]));
                *node_backend_id = overflow_backend;
                SET_CAUSE(node, "3.moe-overflow");
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        size_t cur_split_weight_bytes = 0; // accumulated weight bytes for the current split
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        memset(split->inputs_allocated, 0, sizeof(split->inputs_allocated));
        memset(split->inputs_added, 0, sizeof(split->inputs_added));
        memset(split->input_prefetched, 0, sizeof(split->input_prefetched));
        memset(split->input_transient, 0, sizeof(split->input_transient));
        memset(split->input_full_moe_prefetch, 0, sizeof(split->input_full_moe_prefetch));
        memset(split->transient_buffers, 0, sizeof(split->transient_buffers));
        memset(split->transient_sizes, 0, sizeof(split->transient_sizes));
        memset(split->input_resident, 0, sizeof(split->input_resident));
        memset(split->input_resident_hit, 0, sizeof(split->input_resident_hit));
        memset(split->input_compact_moe, 0, sizeof(split->input_compact_moe));
        memset(split->compact_node, 0, sizeof(split->compact_node));
        memset(split->compact_original_ids, 0, sizeof(split->compact_original_ids));
        memset(split->compact_ids_copy, 0, sizeof(split->compact_ids_copy));
        memset(split->compact_ids_buffer, 0, sizeof(split->compact_ids_buffer));
        memset(split->compact_original_ne2, 0, sizeof(split->compact_original_ne2));
        memset(split->compact_experts, 0, sizeof(split->compact_experts));
        memset(split->compact_remapped_ids, 0, sizeof(split->compact_remapped_ids));
        memset(split->compact_slots, 0, sizeof(split->compact_slots));
        memset(split->compact_misses, 0, sizeof(split->compact_misses));
        split->has_prefetched_inputs = false;
        split->reason = GGML_BACKEND_SCHED_SPLIT_EXPLICIT_MANUAL;
        split->weight_bytes = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            enum ggml_backend_sched_split_reason split_reason = GGML_BACKEND_SCHED_SPLIT_EXPLICIT_MANUAL;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different backend:
                    // - incompatible backend -> force new split (existing behaviour)
                    // - VRAM-limited sequential mode -> break when accumulated weights exceed limit
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        const size_t id = hash_id(src);
                        const int src_backend_id = sched->hv_tensor_backend_ids[id];
                        const bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id || !supported) {
                            bool enrolled = false;
                            for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
                                enrolled = enrolled || split->inputs[input_id] == src;
                            }
                            const bool transient = ggml_backend_sched_input_is_transient(sched, src, cur_backend_id) ||
                                ggml_backend_sched_input_is_full_moe_prefetch(sched, node, src, cur_backend_id);
                            const bool requires_isolated_moe_upload =
                                node->op == GGML_OP_MUL_MAT_ID && node->src[0] == src;
                            // sequential/VRAM-constrained: cap weight accumulation per split
                            if (transient && !enrolled && sched->max_weight_bytes_per_split[cur_backend_id] > 0) {
                                const size_t weight_bytes = ggml_nbytes(src);
                                const size_t limit = sched->max_weight_bytes_per_split[cur_backend_id];
                                if (weight_bytes > limit || cur_split_weight_bytes > limit - weight_bytes) {
                                    need_new_split = true;
                                    split_reason = GGML_BACKEND_SCHED_SPLIT_SEQUENTIAL_BYTE_CAP;
                                    break; // don't count this weight; it starts the next split
                                }
                            }
                            // A sequential transient weight is deliberately host-backed. It is copied into a
                            // bounded device allocation before this split executes, so its source buffer being
                            // unsupported by the compute backend is not a reason to fragment the graph. Ordinary
                            // scheduler inputs retain the conservative incompatible-buffer split behavior.
                            if (!need_new_split && !supported && (!transient || requires_isolated_moe_upload)) {
                                need_new_split = true;
                                split_reason = GGML_BACKEND_SCHED_SPLIT_INCOMPATIBLE_BUFFER_OP;
                                break;
                            }
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        bool enrolled = false;
                        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
                            enrolled = enrolled || split->inputs[input_id] == src;
                        }
                        if (!supported && !enrolled) {
                            need_new_split = true;
                            split_reason = GGML_BACKEND_SCHED_SPLIT_INPUT_LIMIT;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                split->weight_bytes = cur_split_weight_bytes;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->reason = node_backend_id != cur_backend_id ?
                    GGML_BACKEND_SCHED_SPLIT_BACKEND_TRANSITION : split_reason;
                split->i_start = i;
                split->n_inputs = 0;
                cur_split_weight_bytes = 0; // reset for the new split
                memset(split->inputs_allocated, 0, sizeof(split->inputs_allocated));
                memset(split->inputs_added, 0, sizeof(split->inputs_added));
                memset(split->input_prefetched, 0, sizeof(split->input_prefetched));
                memset(split->input_transient, 0, sizeof(split->input_transient));
        memset(split->input_full_moe_prefetch, 0, sizeof(split->input_full_moe_prefetch));
                memset(split->transient_buffers, 0, sizeof(split->transient_buffers));
                memset(split->transient_sizes, 0, sizeof(split->transient_sizes));
                memset(split->input_resident, 0, sizeof(split->input_resident));
                memset(split->input_resident_hit, 0, sizeof(split->input_resident_hit));
                memset(split->input_compact_moe, 0, sizeof(split->input_compact_moe));
                memset(split->compact_node, 0, sizeof(split->compact_node));
                memset(split->compact_original_ids, 0, sizeof(split->compact_original_ids));
                memset(split->compact_ids_copy, 0, sizeof(split->compact_ids_copy));
                memset(split->compact_ids_buffer, 0, sizeof(split->compact_ids_buffer));
                memset(split->compact_original_ne2, 0, sizeof(split->compact_original_ne2));
                memset(split->compact_experts, 0, sizeof(split->compact_experts));
                memset(split->compact_remapped_ids, 0, sizeof(split->compact_remapped_ids));
                memset(split->compact_slots, 0, sizeof(split->compact_slots));
                memset(split->compact_misses, 0, sizeof(split->compact_misses));
                split->has_prefetched_inputs = false;
                split->weight_bytes = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (!ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    const bool transient = ggml_backend_sched_input_is_transient(sched, src, cur_backend_id) ||
                                ggml_backend_sched_input_is_full_moe_prefetch(sched, node, src, cur_backend_id);
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (transient) {
                                tensor_copy->flags = (enum ggml_tensor_flag) (tensor_copy->flags | GGML_TENSOR_FLAG_NO_ALLOC);
                            }
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                    }
                    bool enrolled = false;
                    for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
                        enrolled = enrolled || split->inputs[input_id] == src;
                    }
                    if (!enrolled) {
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                        split->inputs_allocated[n_inputs] = false;
                        split->inputs_added[n_inputs] = false;
                        split->input_prefetched[n_inputs] = false;
                        split->input_transient[n_inputs] = transient;
                        split->input_full_moe_prefetch[n_inputs] =
                            ggml_backend_sched_input_is_full_moe_prefetch(sched, node, src, cur_backend_id);
                        split->transient_buffers[n_inputs] = NULL;
                        split->transient_sizes[n_inputs] = 0;
                        if (transient) {
                            const size_t weight_bytes = ggml_nbytes(src);
                            GGML_ASSERT(SIZE_MAX - cur_split_weight_bytes >= weight_bytes);
                            cur_split_weight_bytes += weight_bytes;
                        }
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        split->weight_bytes = cur_split_weight_bytes;
        sched->n_splits = i_split + 1;
    }

    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        const auto & current = sched->splits[split_id];
        auto & row = sched->transient_metrics.backends[current.backend_id];
        ggml_backend_sched_counter_add(sched, &row.split_reason_count[current.reason], 1);
        const uint64_t nodes = current.i_end >= current.i_start ? (uint64_t) (current.i_end - current.i_start) : 0;
        ggml_backend_sched_counter_add(sched, &row.split_nodes_count, nodes);
        row.split_nodes_max = std::max(row.split_nodes_max, nodes);
        ggml_backend_sched_counter_add(sched,
            &row.split_nodes_histogram[ggml_backend_sched_histogram_bucket(nodes, 1)], 1);
        ggml_backend_sched_counter_add(sched, &row.split_weight_bytes_count, 1);
        ggml_backend_sched_counter_add(sched, &row.split_weight_bytes_total, current.weight_bytes);
        row.split_weight_bytes_max = std::max(row.split_weight_bytes_max, (uint64_t) current.weight_bytes);
        ggml_backend_sched_counter_add(sched,
            &row.split_weight_bytes_histogram[ggml_backend_sched_histogram_bucket(current.weight_bytes, 1 << 20)], 1);
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    const int graph_size_prefetch = sched->async_weight_prefetch ? graph_size + GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies : graph_size;

    if (sched->graph.size < graph_size_prefetch) {
        sched->graph.size = graph_size_prefetch;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size_prefetch * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size_prefetch * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);
    }

    struct ggml_cgraph * graph_copy = &sched->graph;

    auto add_split_input_copy = [&](struct ggml_backend_sched_split * split, int input_id, bool allocate) {
        if (split->inputs_added[input_id]) {
            return;
        }

        assert(graph_copy->size > (graph_copy->n_nodes + 1));

        struct ggml_tensor * input = split->inputs[input_id];
        const size_t input_id_hash = hash_id(input);
        struct ggml_tensor * input_cpy = tensor_id_copy(input_id_hash, split->backend_id, sched->cur_copy);

        // add a dependency to the input source so that it is not freed before the copy is done
        struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
        input_dep->src[0] = input;
        sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id_hash];
        graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

        // add a dependency to the input copy so that it is allocated before it is copied
        sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
        graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;

        split->inputs_allocated[input_id] = allocate;
        split->inputs_added[input_id] = true;
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        if (i + 1 < sched->n_splits) {
            struct ggml_backend_sched_split * next_split = &sched->splits[i + 1];

            ggml_backend_sched_init_prefetch_backend(sched, next_split->backend_id);

            for (int j = 0; j < next_split->n_inputs; j++) {
                if (ggml_backend_sched_split_input_can_prefetch(sched, next_split, j) && split->backend_id == next_split->backend_id &&
                        next_split->inputs[j]->view_src == NULL && tensor_backend_id(next_split->inputs[j]) == sched->n_backends - 1) {
                    add_split_input_copy(next_split, j, false);
                }
            }
        }

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            add_split_input_copy(split, j, true);
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    bool execution_instrumented = false;
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        for (int input_id = 0; input_id < splits[split_id].n_inputs; ++input_id) {
            execution_instrumented = execution_instrumented || splits[split_id].input_transient[input_id];
        }
    }

    ggml_backend_sched_prefetch_storage_inputs(sched, 0);
    if (execution_instrumented) {
        sched->transient_sources_seen->clear();
        ggml_backend_sched_counter_add(sched, &sched->transient_metrics.graph_compute_count, 1);
    }

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        size_t split_transient_bytes = 0;
        bool split_has_transients = false;
        auto & metrics = sched->transient_metrics.backends[split_backend_id];
        if (execution_instrumented) {
            ggml_backend_sched_counter_add(sched, &metrics.splits_seen_count, 1);
        }
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            if (!split->input_transient[input_id]) {
                continue;
            }
            struct ggml_tensor * input_cpy = tensor_copy(split->inputs[input_id], split_backend_id, 0);
            GGML_ASSERT(input_cpy != NULL && (input_cpy->flags & GGML_TENSOR_FLAG_NO_ALLOC));
            if (split->input_prefetched[input_id] && split->transient_buffers[input_id] != NULL) {
                GGML_ASSERT(input_cpy->buffer == split->transient_buffers[input_id] && input_cpy->data != NULL);
                GGML_ASSERT(SIZE_MAX - split_transient_bytes >= split->transient_sizes[input_id]);
                split_transient_bytes += split->transient_sizes[input_id];
                split_has_transients = true;
                continue;
            }
            const struct ggml_tensor * source = split->inputs[input_id];
            const bool cache_eligible = sched->residency_enabled[split_backend_id];

            // Compact MMID is deliberately limited to the sequential transient path and a
            // directly addressable expert-major layout. The original graph descriptors are
            // restored after execution; unsupported cases retain the full-shape path.
            if (split->graph.n_nodes > 0) {
                struct ggml_tensor * node = split->graph.nodes[0];
                if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] == input_cpy &&
                        ggml_backend_sched_compact_moe_layout_supported(source, node->src[2])) {
                    struct ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;
                    for (int i = 0; i < split->n_inputs; ++i) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, ids_tensor);
                            break;
                        }
                    }
                    const int64_t n_ids = ids_tensor->ne[0] * ids_tensor->ne[1];
                    auto * original_ids = new std::vector<int32_t>(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                    const int64_t stall_start = ggml_time_us();
                    if (ids_tensor->buffer == NULL && ids_tensor->data != NULL) {
                        memcpy(original_ids->data(), ids_tensor->data, ggml_nbytes(ids_tensor));
                    } else {
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, original_ids->data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);
                    }
                    ggml_backend_sched_counter_add(sched, &metrics.compact_remap_stall_count, 1);
                    ggml_backend_sched_counter_add(sched, &metrics.compact_remap_stall_us, ggml_backend_sched_elapsed_us(stall_start));

                    auto * experts = new std::vector<int32_t>();
                    experts->reserve((size_t) n_ids);
                    bool valid = n_ids > 0;
                    for (int64_t i1 = 0; valid && i1 < ids_tensor->ne[1]; ++i1) {
                        for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; ++i0) {
                            const int32_t id = (*original_ids)[i1 * ids_tensor->nb[1] / sizeof(int32_t) + i0];
                            if (id < 0 || id >= source->ne[2]) {
                                valid = false;
                                break;
                            }
                            experts->push_back(id);
                        }
                    }
                    std::sort(experts->begin(), experts->end());
                    experts->erase(std::unique(experts->begin(), experts->end()), experts->end());
                    valid = valid && !experts->empty() && experts->size() < (size_t) source->ne[2];
                    if (valid) {
                        const auto slab_key = ggml_backend_sched_expert_slab_key_make(source, split_backend_id);
                        auto slab_it = cache_eligible ? sched->residents->find(slab_key) : sched->residents->end();
                        // Grow compact capacity while preserving all currently resident expert
                        // payloads on-device. If temporary old+new residency is impossible, fall
                        // back to evict+reload rather than failing the graph.
                        if (slab_it != sched->residents->end() && experts->size() > slab_it->second.experts.size()) {
                            if (slab_it->second.executing) {
                                valid = false;
                            } else if (!ggml_backend_sched_grow_expert_slab(
                                    sched, split_backend_id, input_cpy, slab_it, experts->size(), (size_t) source->ne[2])) {
                                ggml_backend_sched_evict_resident(sched, slab_it);
                            }
                        }
                    }
                    if (valid) {
                        auto * remapped = new std::vector<int32_t>(original_ids->size());
                        auto * slots = new std::vector<int32_t>(experts->size());
                        auto * misses = new std::vector<uint8_t>(experts->size(), 1);
                        const auto slab_key = ggml_backend_sched_expert_slab_key_make(source, split_backend_id);
                        auto slab_it = cache_eligible ? sched->residents->find(slab_key) : sched->residents->end();
                        if (slab_it != sched->residents->end()) {
                            auto & slab = slab_it->second;
                            std::vector<bool> reserved(slab.experts.size(), false);
                            for (size_t i = 0; i < experts->size(); ++i) {
                                auto hit = std::find(slab.experts.begin(), slab.experts.end(), (*experts)[i]);
                                if (hit != slab.experts.end()) {
                                    const size_t slot = (size_t) (hit - slab.experts.begin());
                                    (*slots)[i] = (int32_t) slot;
                                    (*misses)[i] = 0;
                                    reserved[slot] = true;
                                    slab.expert_frequency[slot]++;
                            slab.expert_completed_use[slot] = ++sched->residency_use_clock;
                                }
                            }
                            for (size_t i = 0; i < experts->size(); ++i) {
                                if (!(*misses)[i]) {
                                    continue;
                                }
                                size_t victim = slab.experts.size();
                                for (size_t slot = 0; slot < slab.experts.size(); ++slot) {
                                    if (reserved[slot]) {
                                        continue;
                                    }
                                    if (slab.experts[slot] < 0) {
                                        victim = slot;
                                        break;
                                    }
                                    if (victim == slab.experts.size() ||
                                            slab.expert_frequency[slot] < slab.expert_frequency[victim] ||
                                            (slab.expert_frequency[slot] == slab.expert_frequency[victim] &&
                                             slab.expert_completed_use[slot] < slab.expert_completed_use[victim])) {
                                        victim = slot;
                                    }
                                }
                                if (victim == slab.experts.size()) {
                                    valid = false;
                                    break;
                                }
                                if (slab.experts[victim] >= 0) {
                                    ggml_backend_sched_counter_add(sched, &metrics.compact_expert_eviction_count, 1);
                                }
                                slab.experts[victim] = (*experts)[i];
                                slab.expert_frequency[victim] = 1;
                                slab.expert_completed_use[victim] = ++sched->residency_use_clock;
                                (*slots)[i] = (int32_t) victim;
                                reserved[victim] = true;
                            }
                        } else {
                            for (size_t i = 0; i < experts->size(); ++i) {
                                (*slots)[i] = (int32_t) i;
                            }
                        }
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; ++i1) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; ++i0) {
                                const size_t offset = i1 * ids_tensor->nb[1] / sizeof(int32_t) + i0;
                                const auto it = std::lower_bound(experts->begin(), experts->end(), (*original_ids)[offset]);
                                (*remapped)[offset] = (*slots)[(size_t) (it - experts->begin())];
                            }
                        }
                        struct ggml_tensor * ids_copy = ggml_dup_tensor_layout(sched->ctx, ids_tensor);
                        ids_copy->flags = (enum ggml_tensor_flag) (ids_copy->flags | GGML_TENSOR_FLAG_NO_ALLOC);
                        const size_t ids_alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[split_backend_id], ids_copy);
                        ggml_backend_buffer_t ids_buffer = ggml_backend_buft_alloc_buffer(sched->bufts[split_backend_id], ids_alloc_size);
                        if (ids_buffer != NULL && ggml_backend_tensor_alloc(ids_buffer, ids_copy,
                                ggml_backend_buffer_get_base(ids_buffer)) == GGML_STATUS_SUCCESS) {
                            split->input_compact_moe[input_id] = true;
                            split->compact_node[input_id] = node;
                            split->compact_original_ids[input_id] = node->src[2];
                            split->compact_ids_copy[input_id] = ids_copy;
                            split->compact_ids_buffer[input_id] = ids_buffer;
                            split->compact_original_ne2[input_id] = input_cpy->ne[2];
                            split->compact_experts[input_id] = experts;
                            split->compact_remapped_ids[input_id] = remapped;
                            split->compact_slots[input_id] = slots;
                            split->compact_misses[input_id] = misses;
                            input_cpy->ne[2] = slab_it != sched->residents->end() ?
                                (int64_t) slab_it->second.experts.size() : (int64_t) experts->size();
                            node->src[2] = ids_copy;
                            ggml_backend_tensor_set_async(split_backend, ids_copy, remapped->data(), 0, ggml_nbytes(ids_copy));
                        } else {
                            ggml_backend_buffer_free(ids_buffer);
                            delete experts;
                            delete remapped;
                            delete slots;
                            delete misses;
                            ggml_backend_sched_counter_add(sched, &metrics.compact_fallback_count, 1);
                        }
                    } else {
                        delete experts;
                        ggml_backend_sched_counter_add(sched, &metrics.compact_fallback_count, 1);
                    }
                    delete original_ids;
                } else if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] == input_cpy) {
                    ggml_backend_sched_counter_add(sched, &metrics.compact_fallback_count, 1);
                }
            }
            const std::vector<int32_t> empty_experts;
            const std::vector<int32_t> & resident_experts = split->input_compact_moe[input_id] ?
                *split->compact_experts[input_id] : empty_experts;
            const bool expert_tier = split->input_compact_moe[input_id];
            const auto resident_key = split->input_compact_moe[input_id] ?
                ggml_backend_sched_expert_slab_key_make(source, split_backend_id) :
                ggml_backend_sched_resident_key_make(source, split_backend_id, empty_experts);
            if (cache_eligible) {
                auto found = sched->residents->find(resident_key);
                if (found != sched->residents->end()) {
                    auto & resident = found->second;
                    if (resident.source_buffer == source->buffer && resident.source_data == source->data &&
                            resident.logical_size == ggml_nbytes(source) && resident.backend_id == split_backend_id &&
                            (resident.copy == NULL || resident.copy == input_cpy)) {
                        if (resident.copy == NULL) {
                            GGML_ASSERT(input_cpy->buffer == NULL && input_cpy->data == NULL);
                            resident.copy = input_cpy;
                        } else {
                            GGML_ASSERT(input_cpy->buffer == resident.buffer && input_cpy->data != NULL);
                        }
                        // Record the resident payload for every execution, including same-graph hits.
                        // Completion uses this identity to clear executing and graph rebuild/reset uses
                        // it to detach the descriptor without taking ownership of the resident buffer.
                        split->transient_buffers[input_id] = resident.buffer;
                        split->transient_sizes[input_id] = resident.allocation_size;
                        resident.executing = true;
                        resident.frequency++;
                        split->input_resident[input_id] = true;
                        const size_t miss_count = split->input_compact_moe[input_id] ?
                            std::count(split->compact_misses[input_id]->begin(), split->compact_misses[input_id]->end(), 1) : 0;
                        split->input_resident_hit[input_id] = miss_count == 0;
                        if (split->input_compact_moe[input_id]) {
                            ggml_backend_sched_counter_add(sched, &metrics.compact_expert_hit_count,
                                resident_experts.size() - miss_count);
                            ggml_backend_sched_counter_add(sched, &metrics.compact_expert_miss_count, miss_count);
                        }
                        if (miss_count == 0) {
                            ggml_backend_sched_counter_add(sched, &metrics.residency_hit_count, 1);
                            continue;
                        }
                        split_has_transients = true;
                        continue;
                    }
                    ggml_backend_sched_drain_residents(sched);
                }
                if (input_cpy->buffer != NULL || input_cpy->data != NULL) {
                    auto attached = sched->residents->end();
                    for (auto it = sched->residents->begin(); it != sched->residents->end(); ++it) {
                        if (it->second.copy == input_cpy && !it->second.executing) {
                            attached = it;
                            break;
                        }
                    }
                    if (attached != sched->residents->end()) {
                        ggml_backend_sched_evict_resident(sched, attached);
                    }
                }
            }
            GGML_ASSERT(input_cpy->buffer == NULL && input_cpy->data == NULL);
            const size_t alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[split_backend_id], input_cpy);
            ggml_backend_sched_counter_add(sched, &metrics.allocation_requested_bytes, alloc_size);
            const size_t split_limit = sched->max_weight_bytes_per_split[split_backend_id];
            const bool limit_rejected = split_limit > 0 &&
                alloc_size > split_limit - std::min(split_transient_bytes, split_limit);
            bool unknown_memory = false;
            bool live_guard_rejected = false;
            const bool resident_admitted = cache_eligible && alloc_size > 0 &&
                ggml_backend_sched_make_resident_space(sched, split_backend_id, alloc_size);
            const bool window_rejected = alloc_size > 0 && !resident_admitted &&
                !ggml_backend_sched_weight_window_admit(sched, split_backend_id, alloc_size, &unknown_memory, &live_guard_rejected);
            if (alloc_size == 0 || limit_rejected || window_rejected) {
                ggml_backend_sched_counter_add(sched, &metrics.allocation_rejected_bytes, alloc_size);
                if (alloc_size == 0) {
                    ggml_backend_sched_counter_add(sched, &metrics.allocation_failure_count, 1);
                }
                if (limit_rejected || window_rejected) {
                    ggml_backend_sched_counter_add(sched, &metrics.allocation_limit_rejection_count, 1);
                    if ((split_limit > 0 && alloc_size > split_limit) ||
                            (sched->weight_window_configured[split_backend_id] &&
                             alloc_size > sched->weight_window_limit[split_backend_id])) {
                        ggml_backend_sched_counter_add(sched, &metrics.oversized_tensor_rejection_count, 1);
                    }
                }
                if (live_guard_rejected) {
                    ggml_backend_sched_counter_add(sched, &metrics.allocation_live_guard_rejection_count, 1);
                }
                if (unknown_memory) {
                    ggml_backend_sched_counter_add(sched, &metrics.allocation_unknown_memory_rejection_count, 1);
                }
                ggml_backend_sched_release_transients(sched, split, false,
                    GGML_BACKEND_SCHED_TRANSIENT_DRAIN_ALLOCATION_FAILURE, false);
                ggml_backend_sched_drain_residents(sched);
                ggml_backend_sched_counter_add(sched, &sched->transient_metrics.graph_compute_failure_count, 1);
                return GGML_STATUS_ALLOC_FAILED;
            }
            split_transient_bytes += alloc_size;
            const int64_t allocation_start_us = ggml_time_us();
            ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(sched->bufts[split_backend_id], alloc_size);
            ggml_backend_sched_counter_add(sched, &metrics.allocation_time_us, ggml_backend_sched_elapsed_us(allocation_start_us));
            if (buffer == NULL) {
                ggml_backend_sched_counter_add(sched, &metrics.allocation_failure_count, 1);
                ggml_backend_sched_counter_add(sched, &metrics.allocation_rejected_bytes, alloc_size);
                ggml_backend_sched_release_transients(sched, split, false,
                    GGML_BACKEND_SCHED_TRANSIENT_DRAIN_ALLOCATION_FAILURE, false);
                ggml_backend_sched_drain_residents(sched);
                ggml_backend_sched_counter_add(sched, &sched->transient_metrics.graph_compute_failure_count, 1);
                return GGML_STATUS_ALLOC_FAILED;
            }
            split->transient_buffers[input_id] = buffer;
            split->transient_sizes[input_id] = alloc_size;
            if (resident_admitted) {
                ggml_backend_sched_resident resident{};
                resident.source = source;
                resident.source_buffer = source->buffer;
                resident.source_data = source->data;
                resident.logical_size = ggml_nbytes(source);
                resident.backend_id = split_backend_id;
                resident.copy = input_cpy;
                resident.buffer = buffer;
                resident.allocation_size = alloc_size;
                resident.executing = true;
                resident.frequency = 1;
                resident.expert_tier = expert_tier;
                resident.experts = resident_experts;
                resident.expert_frequency.assign(resident_experts.size(), 1);
                resident.expert_completed_use.assign(resident_experts.size(), ++sched->residency_use_clock);
                sched->residents->emplace(resident_key, resident);
                metrics.current_resident_bytes += alloc_size;
                metrics.current_resident_records++;
                split->input_resident[input_id] = true;
                ggml_backend_sched_counter_add(sched, &metrics.residency_miss_count, 1);
                ggml_backend_sched_counter_add(sched, &metrics.compact_expert_miss_count, resident.experts.size());
                ggml_backend_sched_resident_metrics_update(sched, split_backend_id);
            } else {
                ggml_backend_sched_ledger_enter(sched, split_backend_id, alloc_size);
                 if (std::any_of(sched->residency_enabled, sched->residency_enabled + sched->n_backends,
                         [](bool enabled) { return enabled; })) {
                    ggml_backend_sched_counter_add(sched, &metrics.residency_fallback_count, 1);
                }
            }
            ggml_backend_sched_counter_add(sched, &metrics.allocation_admitted_bytes, alloc_size);
            ggml_backend_sched_counter_add(sched, &metrics.allocation_count, 1);
            split_has_transients = true;
        }
        if (split_has_transients) {
            ggml_backend_sched_counter_add(sched, &metrics.transient_split_count, 1);
        }
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            ggml_backend_buffer_t buffer = split->transient_buffers[input_id];
            if (buffer == NULL) {
                continue;
            }
            struct ggml_tensor * input_cpy = tensor_copy(split->inputs[input_id], split_backend_id, 0);
            if (input_cpy->buffer == buffer) {
                GGML_ASSERT(input_cpy->data != NULL);
                continue;
            }
            const size_t required_size = ggml_backend_buft_get_alloc_size(sched->bufts[split_backend_id], input_cpy);
            if (required_size > ggml_backend_buffer_get_size(buffer)) {
                GGML_LOG_ERROR(
                    "%s: transient shape grew after allocation: tensor=%s backend=%s required=%zu buffer=%zu "
                    "ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] compact=%d resident=%d\n",
                    __func__, input_cpy->name, ggml_backend_name(split_backend), required_size,
                    ggml_backend_buffer_get_size(buffer), input_cpy->ne[0], input_cpy->ne[1], input_cpy->ne[2],
                    input_cpy->ne[3], split->input_compact_moe[input_id], split->input_resident[input_id]);
                ggml_backend_sched_release_transients(sched, split, false,
                    GGML_BACKEND_SCHED_TRANSIENT_DRAIN_ATTACHMENT_FAILURE, false);
                ggml_backend_sched_counter_add(sched, &sched->transient_metrics.graph_compute_failure_count, 1);
                ggml_backend_sched_drain_residents(sched);
                return GGML_STATUS_ALLOC_FAILED;
            }
            enum ggml_status ec = ggml_backend_tensor_alloc(buffer, input_cpy, ggml_backend_buffer_get_base(buffer));
            if (ec != GGML_STATUS_SUCCESS) {
                ggml_backend_sched_release_transients(sched, split, true,
                    GGML_BACKEND_SCHED_TRANSIENT_DRAIN_ATTACHMENT_FAILURE, false);
                ggml_backend_sched_counter_add(sched, &sched->transient_metrics.graph_compute_failure_count, 1);
                ggml_backend_sched_drain_residents(sched);
                return ec;
            }
        }

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (split->input_resident_hit[input_id]) {
                continue;
            }

            if (ggml_backend_sched_split_input_was_prefetched(sched, split, input_id)) {
                continue;
            }

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = 0; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        if (ids_tensor->buffer == NULL && ids_tensor->data != NULL) {
                            memcpy(ids.data(), ids_tensor->data, ggml_nbytes(ids_tensor));
                        } else {
                            ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                            ggml_backend_synchronize(ids_backend);
                        }

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    if (split->input_compact_moe[input_id]) {
                        const auto & compact_experts = *split->compact_experts[input_id];
                        const auto & compact_slots = *split->compact_slots[input_id];
                        const auto & compact_misses = *split->compact_misses[input_id];
                        for (size_t compact_id = 0; compact_id < compact_experts.size();) {
                            if (!compact_misses[compact_id]) {
                                ++compact_id;
                                continue;
                            }
                            size_t last = compact_id;
                            while (last + 1 < compact_experts.size() && compact_misses[last + 1] &&
                                    compact_experts[last + 1] == compact_experts[last] + 1 &&
                                    compact_slots[last + 1] == compact_slots[last] + 1) {
                                ++last;
                            }
                            const size_t src_offset = (size_t) compact_experts[compact_id] * expert_size;
                            const size_t dst_offset = (size_t) compact_slots[compact_id] * expert_size;
                            const size_t copy_size = (last - compact_id + 1) * expert_size;
                            ggml_backend_sched_weight_upload_chunked(sched, split_backend, split_backend_id,
                                input_cpy, (const uint8_t *) input->data + src_offset, dst_offset,
                                copy_size, split->input_transient[input_id]);
                            compact_id = last + 1;
                        }
                        const size_t full_alloc = ggml_backend_buft_get_alloc_size(sched->bufts[split_backend_id], input);
                        ggml_backend_sched_counter_add(sched, &metrics.compact_physical_bytes, split->transient_sizes[input_id]);
                        if (full_alloc > split->transient_sizes[input_id]) {
                            ggml_backend_sched_counter_add(sched, &metrics.compact_avoided_full_allocation_bytes,
                                full_alloc - split->transient_sizes[input_id]);
                        }
                    } else {
                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_sched_weight_upload_chunked(sched, split_backend, split_backend_id,
                            input_cpy,
                            (const uint8_t *) input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end, split->input_transient[input_id]);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                    }
                    if (split->input_transient[input_id]) {
                        ggml_backend_sched_counter_add(sched, &metrics.upload_count, 1);
                        ggml_backend_sched_counter_add(sched, &metrics.uploaded_backend_bytes, split->transient_sizes[input_id]);
                        if (!sched->transient_sources_seen->insert(input).second) {
                            ggml_backend_sched_counter_add(sched, &metrics.shared_reload_count, 1);
                        }
                        if (split->input_resident[input_id]) {
                            ggml_backend_sched_counter_add(sched, &metrics.residency_upload_count, 1);
                        }
                    }
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (input->data != NULL && ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                            ggml_backend_buffer_is_host(input->buffer) &&
                            !ggml_backend_buffer_is_host(input_cpy->buffer)) {
                        ggml_backend_sched_weight_upload_chunked(sched, split_backend, split_backend_id,
                            input_cpy, input->data, 0, ggml_nbytes(input_cpy), split->input_transient[input_id]);
                        if (split->input_transient[input_id]) {
                            ggml_backend_sched_counter_add(sched, &metrics.upload_count, 1);
                            ggml_backend_sched_counter_add(sched, &metrics.uploaded_backend_bytes, split->transient_sizes[input_id]);
                            if (!sched->transient_sources_seen->insert(input).second) {
                                ggml_backend_sched_counter_add(sched, &metrics.shared_reload_count, 1);
                            }
                            if (split->input_resident[input_id]) {
                                ggml_backend_sched_counter_add(sched, &metrics.residency_upload_count, 1);
                            }
                        }
                    } else if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        if (split->has_prefetched_inputs) {
            ggml_backend_event_wait(split_backend, sched->prefetch_events[split_backend_id][split_id & 1]);
        }

        // Uploads and graph compute are submitted to the same backend stream. Stream
        // ordering is the completion dependency, so a device-wide pre-compute wait is
        // unnecessary. Staging-slot events independently protect host-buffer reuse.

        // Start the next split's bounded storage read before launching this split's
        // compute so O_DIRECT latency can overlap GPU execution. Routed MoE inputs are
        // excluded because their expert slices are known only when that split begins.
        ggml_backend_sched_prefetch_storage_inputs(sched, split_id + 1);

        bool compute_submitted = false;
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            compute_submitted = true;
            if (ec != GGML_STATUS_SUCCESS) {
                ggml_backend_sched_release_transients(sched, split, true,
                    GGML_BACKEND_SCHED_TRANSIENT_DRAIN_COMPUTE_FAILURE, compute_submitted);
                ggml_backend_sched_counter_add(sched, &sched->transient_metrics.graph_compute_failure_count, 1);
                ggml_backend_sched_drain_residents(sched);
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                compute_submitted = true;
                if (ec != GGML_STATUS_SUCCESS) {
                    ggml_backend_sched_release_transients(sched, split, true,
                        GGML_BACKEND_SCHED_TRANSIENT_DRAIN_COMPUTE_FAILURE, compute_submitted);
                    ggml_backend_sched_counter_add(sched, &sched->transient_metrics.graph_compute_failure_count, 1);
                    ggml_backend_sched_drain_residents(sched);
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                const int64_t wait_start_us = split_has_transients ? ggml_time_us() : 0;
                ggml_backend_synchronize(split_backend);
                if (split_has_transients) {
                    ggml_backend_sched_counter_add(sched, &metrics.compute_completion_wait_count, 1);
                    ggml_backend_sched_counter_add(sched, &metrics.compute_completion_wait_us, ggml_backend_sched_elapsed_us(wait_start_us));
                    compute_submitted = false;
                }

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    if (execution_instrumented) {
                        ggml_backend_sched_counter_add(sched, &sched->transient_metrics.callback_early_stop_count, 1);
                    }
                    break;
                }

                j0 = j1;
            }
        }

        // With N executing asynchronously, allocate/admit N+1 only when spare
        // residency space is immediately available, then queue H2D on the
        // secondary backend stream. Different-GPU transitions are valid too.
        ggml_backend_sched_prefetch_split_inputs(sched, split_id + 1);

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }

        if (split_has_transients && sched->events[split_backend_id][sched->cur_copy] != NULL) {
            const int64_t wait_start_us = ggml_time_us();
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
            for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
                if (!sched->staging[split_backend_id].reserved[slot]) {
                    sched->staging[split_backend_id].pending[slot] = false;
                }
            }
            ggml_backend_sched_counter_add(sched, &metrics.compute_completion_wait_count, 1);
            ggml_backend_sched_counter_add(sched, &metrics.compute_completion_wait_us, ggml_backend_sched_elapsed_us(wait_start_us));
            compute_submitted = false;
        }

        ggml_backend_sched_release_transients(sched, split,
            split_has_transients && sched->events[split_backend_id][sched->cur_copy] == NULL,
            GGML_BACKEND_SCHED_TRANSIENT_DRAIN_NORMAL, compute_submitted);
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            if (!split->input_resident[input_id]) {
                continue;
            }
            for (auto & entry : *sched->residents) {
                if (entry.second.buffer == split->transient_buffers[input_id]) {
                    entry.second.executing = false;
                    entry.second.completed_use = ++sched->residency_use_clock;
                    break;
                }
            }
            split->transient_buffers[input_id] = NULL;
            split->transient_sizes[input_id] = 0;
            split->input_resident[input_id] = false;
            split->input_resident_hit[input_id] = false;
            ggml_backend_sched_compact_reset_input(split, input_id, true);
        }

    }

    if (sched->transient_count != 0 || sched->transient_bytes != 0) {
        ggml_backend_sched_counter_add(sched, &sched->transient_metrics.ledger_mismatch_count, 1);
        GGML_ASSERT(false && "scheduler transient ledger not empty after compute");
    }
    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = new ggml_backend_sched{};

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;
    sched->transient_sources_seen = new std::unordered_set<const struct ggml_tensor *>();
    sched->residents = new ggml_backend_sched_resident_map();
    sched->transient_metrics.n_backends = n_backends;
    for (int b = 0; b < n_backends; ++b) {
        sched->transient_metrics.backends[b].backend_index = b;
        sched->transient_metrics.backends[b].backend = backends[b];
    }

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
        if (ggml_backend_dev_type(backends[b]->device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
                sched->staging[b].events[slot] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    const char * prefetch_moe_env = getenv("GGML_SCHED_PREFETCH_EXPERTS");
    sched->prefetch_full_moe = op_offload && prefetch_moe_env != NULL && atoi(prefetch_moe_env) > 0;
    if (sched->prefetch_full_moe) {
        // Standard offloaded-MoE models do not otherwise enable the sequential
        // prefetch machinery. Reuse it rather than maintaining a second stream stack.
        sched->async_weight_prefetch = true;
    }

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    ggml_backend_sched_drain_transients(sched, GGML_BACKEND_SCHED_TRANSIENT_DRAIN_DESTRUCTION);
    ggml_backend_sched_drain_residents(sched);
    for (int b = 0; b < sched->n_backends; b++) {
        for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
            ggml_backend_sched_storage_prefetch_release(sched, b, slot);
        }
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
        if (sched->prefetch_backends[b] != NULL) {
            ggml_backend_synchronize(sched->prefetch_backends[b]);
        }
        for (int c = 0; c < 2; c++) {
            ggml_backend_event_free(sched->prefetch_events[b][c]);
        }
        if (sched->prefetch_backends[b] != NULL) {
            ggml_backend_free(sched->prefetch_backends[b]);
        }
        for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
            if (sched->staging[b].pending[slot]) {
                ggml_backend_event_synchronize(sched->staging[b].events[slot]);
            }
            ggml_backend_buffer_free(sched->staging[b].buffers[slot]);
            ggml_backend_event_free(sched->staging[b].events[slot]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    delete sched->transient_sources_seen;
    delete sched->residents;
    delete sched;
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int b = 0; b < sched->n_backends; ++b) {
        for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
            ggml_backend_sched_storage_prefetch_release(sched, b, slot);
        }
    }
    ggml_backend_sched_drain_transients(sched, GGML_BACKEND_SCHED_TRANSIENT_DRAIN_RESET);
    if (sched->persistent_weight_residency) {
        for (auto & entry : *sched->residents) {
            entry.second.copy = NULL;
            entry.second.executing = false;
        }
    } else {
        ggml_backend_sched_drain_residents(sched);
    }
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
        if (sched->prefetch_backends[i] != NULL) {
            ggml_backend_synchronize(sched->prefetch_backends[i]);
        }
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

bool ggml_backend_sched_get_transient_metrics(
        ggml_backend_sched_t sched, struct ggml_backend_sched_transient_metrics * out) {
    if (sched == NULL || out == NULL) {
        return false;
    }
    *out = sched->transient_metrics;
    return true;
}

static uint64_t ggml_backend_sched_counter_delta(uint64_t current, uint64_t baseline) {
    return current >= baseline ? current - baseline : current;
}

bool ggml_backend_sched_get_transient_metrics_delta(
        ggml_backend_sched_t sched,
        const struct ggml_backend_sched_transient_metrics * baseline,
        struct ggml_backend_sched_transient_metrics * out) {
    if (!ggml_backend_sched_get_transient_metrics(sched, out) || baseline == NULL) {
        return false;
    }
    auto subtract = [](uint64_t * current, const uint64_t * base, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            current[i] = ggml_backend_sched_counter_delta(current[i], base[i]);
        }
    };
    subtract(&out->graph_compute_count, &baseline->graph_compute_count, 5);
    for (int b = 0; b < out->n_backends; ++b) {
        auto & row = out->backends[b];
        const auto & base = baseline->backends[b];
        subtract(&row.allocation_requested_bytes, &base.allocation_requested_bytes,
            (&row.mmap_readahead_time_us - &row.allocation_requested_bytes) + 1);
        if (row.split_nodes_count == 0) {
            row.split_nodes_max = 0;
        }
        if (row.split_weight_bytes_count == 0) {
            row.split_weight_bytes_max = 0;
        }
    }
    return true;
}

void ggml_backend_sched_reset_transient_metrics(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    const auto old = sched->transient_metrics;
    memset(&sched->transient_metrics, 0, sizeof(sched->transient_metrics));
    sched->transient_metrics.n_backends = sched->n_backends;
    sched->transient_metrics.current_transient_bytes = old.current_transient_bytes;
    sched->transient_metrics.current_transient_records = old.current_transient_records;
    sched->transient_metrics.current_resident_bytes = old.current_resident_bytes;
    sched->transient_metrics.current_resident_records = old.current_resident_records;
    for (int b = 0; b < sched->n_backends; ++b) {
        auto & row = sched->transient_metrics.backends[b];
        const auto & prev = old.backends[b];
        row.backend_index = b;
        row.backend = sched->backends[b];
        row.current_transient_bytes = prev.current_transient_bytes;
        row.current_transient_records = prev.current_transient_records;
        row.current_resident_bytes = prev.current_resident_bytes;
        row.current_resident_records = prev.current_resident_records;
        row.weight_window_limit_bytes = prev.weight_window_limit_bytes;
        row.weight_window_safety_reserve_bytes = prev.weight_window_safety_reserve_bytes;
        row.weight_window_post_reservation_free_bytes = prev.weight_window_post_reservation_free_bytes;
        row.weight_window_total_bytes = prev.weight_window_total_bytes;
        row.weight_window_configured = prev.weight_window_configured;
        row.weight_window_memory_valid = prev.weight_window_memory_valid;
        row.staging_buffer_bytes = prev.staging_buffer_bytes;
    }
}

void ggml_backend_sched_test_counter_add(ggml_backend_sched_t sched, uint64_t * counter, uint64_t value) {
    ggml_backend_sched_counter_add(sched, counter, value);
}

void ggml_backend_sched_print_transient_metrics(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->transient_metrics.n_backends; ++i) {
        const auto & row = sched->transient_metrics.backends[i];
        if (row.split_weight_bytes_count == 0 && row.splits_seen_count == 0 && row.upload_count == 0 && row.residency_hit_count == 0) {
            continue;
        }
        GGML_LOG_INFO("sequential %-12s splits(total/transient)=%" PRIu64 "/%" PRIu64 " uploads=%" PRIu64
                      " logical/backend=%.2f/%.2f GiB resident(hit/miss/evict)=%" PRIu64 "/%" PRIu64 "/%" PRIu64
                      " compact=%.2f GiB avoided=%.2f GiB expert(hit/miss/evict)=%" PRIu64 "/%" PRIu64 "/%" PRIu64
                      " remap-stalls=%" PRIu64 "/%.2f s fallback=%" PRIu64
                      " split-reasons(b/i/n/c/m)=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
                      " nodes(mean/max)=%.1f/%" PRIu64 " weights(mean/max)=%.2f/%.2f MiB"
                      " faults(min/maj)=%" PRIu64 "/%" PRIu64 " readahead=%.2f GiB/%.3f s"
                      " staged=%.2f GiB/%zu MiB alloc=%.2f s upload-submit=%.2f s transfer-wait=%.2f s compute-wait=%.2f s drain=%.2f s\n",
            ggml_backend_name(row.backend), row.splits_seen_count, row.transient_split_count, row.upload_count,
            row.uploaded_logical_bytes / (1024.0 * 1024.0 * 1024.0),
            row.uploaded_backend_bytes / (1024.0 * 1024.0 * 1024.0),
            row.residency_hit_count, row.residency_miss_count, row.residency_eviction_count,
            row.compact_physical_bytes / (1024.0 * 1024.0 * 1024.0),
            row.compact_avoided_full_allocation_bytes / (1024.0 * 1024.0 * 1024.0),
            row.compact_expert_hit_count, row.compact_expert_miss_count, row.compact_expert_eviction_count,
            row.compact_remap_stall_count, row.compact_remap_stall_us / 1000000.0, row.compact_fallback_count,
            row.split_reason_count[GGML_BACKEND_SCHED_SPLIT_BACKEND_TRANSITION],
            row.split_reason_count[GGML_BACKEND_SCHED_SPLIT_INCOMPATIBLE_BUFFER_OP],
            row.split_reason_count[GGML_BACKEND_SCHED_SPLIT_INPUT_LIMIT],
            row.split_reason_count[GGML_BACKEND_SCHED_SPLIT_SEQUENTIAL_BYTE_CAP],
            row.split_reason_count[GGML_BACKEND_SCHED_SPLIT_EXPLICIT_MANUAL],
            row.split_weight_bytes_count ? (double) row.split_nodes_count / row.split_weight_bytes_count : 0.0,
            row.split_nodes_max,
            row.split_weight_bytes_count ? row.split_weight_bytes_total / (1024.0 * 1024.0 * row.split_weight_bytes_count) : 0.0,
            row.split_weight_bytes_max / (1024.0 * 1024.0), row.mmap_minor_faults, row.mmap_major_faults,
            row.mmap_readahead_bytes / (1024.0 * 1024.0 * 1024.0), row.mmap_readahead_time_us / 1e6,
            row.staged_upload_bytes / (1024.0 * 1024.0 * 1024.0), row.staging_buffer_bytes / (1024 * 1024),
            row.allocation_time_us / 1e6, row.upload_submission_time_us / 1e6,
            row.transfer_completion_wait_us / 1e6,
            row.compute_completion_wait_us / 1e6,
            row.drain_time_us[GGML_BACKEND_SCHED_TRANSIENT_DRAIN_NORMAL] / 1e6);
    }
}

void ggml_backend_sched_set_force_weight_offload(ggml_backend_sched_t sched, bool force) {
    GGML_ASSERT(sched);
    if (!force) {
        ggml_backend_sched_drain_residents(sched);
    }
    sched->force_weight_offload = force;
}

void ggml_backend_sched_set_force_weight_offload_split(
        ggml_backend_sched_t sched, const float * weights, int n_weights) {
    GGML_ASSERT(sched != NULL);
    const int n_devices = sched->n_backends - 1;
    sched->force_weight_offload_split_configured = false;
    memset(sched->force_weight_offload_split, 0, sizeof(sched->force_weight_offload_split));
    if (weights == NULL || n_weights <= 0 || n_devices <= 1) {
        return;
    }
    GGML_ASSERT(n_weights == n_devices);
    float total = 0.0f;
    for (int i = 0; i < n_weights; ++i) {
        total += std::max(weights[i], 0.0f);
    }
    if (total <= 0.0f) {
        return;
    }
    float cumulative = 0.0f;
    for (int i = 0; i < n_weights; ++i) {
        cumulative += std::max(weights[i], 0.0f) / total;
        sched->force_weight_offload_split[i] = cumulative;
    }
    sched->force_weight_offload_split[n_weights - 1] = 1.0f;
    sched->force_weight_offload_split_configured = true;
}

void ggml_backend_sched_set_weight_residency(
        ggml_backend_sched_t sched, ggml_backend_t backend, bool enabled) {
    GGML_ASSERT(sched);
    const int backend_id = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_id >= 0);
    if (!enabled) {
        for (auto it = sched->residents->begin(); it != sched->residents->end();) {
            if (it->second.backend_id != backend_id) {
                ++it;
                continue;
            }
            it->second.executing = false;
            auto victim = it++;
            ggml_backend_sched_evict_resident(sched, victim);
            ggml_backend_sched_counter_add(sched,
                &sched->transient_metrics.backends[backend_id].residency_drain_count, 1);
        }
    }
    sched->residency_enabled[backend_id] = enabled;
}

void ggml_backend_sched_set_persistent_weight_residency(ggml_backend_sched_t sched, bool persistent) {
    GGML_ASSERT(sched);
    if (!persistent) {
        ggml_backend_sched_drain_residents(sched);
    }
    sched->persistent_weight_residency = persistent;
}

void ggml_backend_sched_set_async_weight_prefetch(ggml_backend_sched_t sched, bool prefetch) {
    GGML_ASSERT(sched);
    sched->async_weight_prefetch = prefetch;
}

void ggml_backend_sched_set_weight_read_callback(
        ggml_backend_sched_t sched, ggml_backend_sched_weight_read_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->weight_read_callback = callback;
    sched->weight_read_callback_user_data = user_data;
}

void ggml_backend_sched_set_weight_read_padded_callback(
        ggml_backend_sched_t sched, ggml_backend_sched_weight_read_padded_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->weight_read_padded_callback = callback;
    sched->weight_read_callback_user_data = user_data;
}

void ggml_backend_sched_set_max_weight_bytes_per_split(
        ggml_backend_sched_t sched, ggml_backend_t backend, size_t max_bytes) {
    GGML_ASSERT(sched);
    const int backend_id = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_id >= 0);
    sched->max_weight_bytes_per_split[backend_id] = max_bytes;
}

bool ggml_backend_sched_set_weight_window(
        ggml_backend_sched_t sched, ggml_backend_t backend,
        size_t free_bytes, size_t total_bytes, size_t configured_cap,
        size_t * window_bytes, size_t * safety_reserve_bytes) {
    GGML_ASSERT(sched);
    const int backend_id = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_id >= 0);

    auto & row = sched->transient_metrics.backends[backend_id];
    const size_t reserve = ggml_backend_sched_weight_window_safety_reserve(total_bytes);
    const bool valid = free_bytes > 0 && total_bytes > 0 && free_bytes <= total_bytes && free_bytes > reserve;
    const size_t available = valid ? free_bytes - reserve : 0;
    const size_t limit = std::min(configured_cap, available);

    sched->weight_window_configured[backend_id] = true;
    sched->weight_window_memory_valid[backend_id] = valid;
    sched->weight_window_limit[backend_id] = limit;
    sched->weight_window_safety_reserve[backend_id] = reserve;
    row.weight_window_configured = true;
    row.weight_window_memory_valid = valid;
    row.weight_window_limit_bytes = limit;
    row.weight_window_safety_reserve_bytes = reserve;
    row.weight_window_post_reservation_free_bytes = free_bytes;
    row.weight_window_total_bytes = total_bytes;
    if (window_bytes != NULL) {
        *window_bytes = limit;
    }
    if (safety_reserve_bytes != NULL) {
        *safety_reserve_bytes = reserve;
    }
    return valid;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
