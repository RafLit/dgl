/**
 *  Copyright (c) 2016-2022 by Contributors
 * @file xpu_device_api.cc
 */
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/tensordispatch.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

#include <cstdlib>
#include <cstring>

#include "workspace_pool.h"

namespace dgl {
namespace runtime {
class XPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(DGLContext ctx) final {}
  void GetAttr(DGLContext ctx, DeviceAttrKind kind, DGLRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(
      DGLContext ctx, size_t nbytes, size_t alignment,
      DGLDataType type_hint) final {
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable())
      return tensor_dispatcher->CPUAllocWorkspace(nbytes);

    void* ptr;
#if _MSC_VER || defined(__MINGW32__)
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr) throw std::bad_alloc();
#elif defined(_LIBCPP_SGX_CONFIG)
    ptr = memalign(alignment, nbytes);
    if (ptr == nullptr) throw std::bad_alloc();
#else
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0) throw std::bad_alloc();
#endif
    return ptr;
  }

  void FreeDataSpace(DGLContext ctx, void* ptr) final {
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable())
      return tensor_dispatcher->CPUFreeWorkspace(ptr);

#if _MSC_VER || defined(__MINGW32__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void CopyDataFromTo(
      const void* from, size_t from_offset, void* to, size_t to_offset,
      size_t size, DGLContext ctx_from, DGLContext ctx_to,
      DGLDataType type_hint) final {
    memcpy(
        static_cast<char*>(to) + to_offset,
        static_cast<const char*>(from) + from_offset, size);
  }

  void RecordedCopyDataFromTo(
      void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
      DGLContext ctx_from, DGLContext ctx_to, DGLDataType type_hint,
      void* pytorch_ctx) final {
    BUG_IF_FAIL(false) << "This piece of code should not be reached.";
  }

  DGLStreamHandle CreateStream(DGLContext) final { return nullptr; }

  void StreamSync(DGLContext ctx, DGLStreamHandle stream) final {}

  void* AllocWorkspace(
      DGLContext ctx, size_t size, DGLDataType type_hint) final;
  void FreeWorkspace(DGLContext ctx, void* data) final;

  static const std::shared_ptr<XPUDeviceAPI>& Global() {
    static std::shared_ptr<XPUDeviceAPI> inst =
        std::make_shared<XPUDeviceAPI>();
    return inst;
  }
};

struct XPUWorkspacePool : public WorkspacePool {
  XPUWorkspacePool() : WorkspacePool(kDGLXPU, XPUDeviceAPI::Global()) {}
};

void* XPUDeviceAPI::AllocWorkspace(
    DGLContext ctx, size_t size, DGLDataType type_hint) {
  TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
  if (tensor_dispatcher->IsAvailable()) {
    return tensor_dispatcher->CPUAllocWorkspace(size);
  }

  return dmlc::ThreadLocalStore<XPUWorkspacePool>::Get()->AllocWorkspace(
      ctx, size);
}

void XPUDeviceAPI::FreeWorkspace(DGLContext ctx, void* data) {
  TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
  if (tensor_dispatcher->IsAvailable()) {
    return tensor_dispatcher->CPUFreeWorkspace(data);
  }

  dmlc::ThreadLocalStore<XPUWorkspacePool>::Get()->FreeWorkspace(ctx, data);
}

DGL_REGISTER_GLOBAL("device_api.xpu")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      DeviceAPI* ptr = XPUDeviceAPI::Global().get();
      *rv = static_cast<void*>(ptr);
    });
}  // namespace runtime
}  // namespace dgl
