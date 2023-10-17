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
#include <sycl/sycl.hpp>

#include "xpu_common.h"
//#include <xpu/Macros.h>
//#include <xpu/Stream.h>
#include <ipex.h>

#include "../workspace_pool.h"

constexpr size_t kDevAlignment = 512;
namespace dgl {
namespace runtime {
sycl::queue& getCurrentXPUStream() {
  static sycl::queue&  q = xpu::get_queue_from_stream(c10::impl::VirtualGuardImpl(c10::DeviceType::XPU).getStream(c10::Device(c10::DeviceType::XPU)));
  return q;
}
class XPUDeviceAPI final : public DeviceAPI {
 sycl::queue &q_;
 public:
  XPUDeviceAPI():q_(getCurrentXPUStream()) {
  }
  void SetDevice(DGLContext ctx) final {}
  void GetAttr(DGLContext ctx, DeviceAttrKind kind, DGLRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(
      DGLContext ctx, size_t nbytes, size_t alignment,
      DGLDataType type_hint) final {
    void *mem = nullptr; 
    mem = sycl::aligned_alloc_device(kDevAlignment, nbytes, q_);
    q_.memset(mem,0,nbytes);
    std::cout<<"allocated "<< nbytes<< " at address "<<mem<<std::endl;
    return mem;

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
    sycl::free(ptr, q_);

//#if _MSC_VER || defined(__MINGW32__)
//    _aligned_free(ptr);
//#else
//    free(ptr);
//#endif
  }

  void CopyDataFromTo(
      const void* from, size_t from_offset, void* to, size_t to_offset,
      size_t size, DGLContext ctx_from, DGLContext ctx_to,
      DGLDataType type_hint) final {
        q_.submit([&](sycl::handler &h) { h.memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from)+from_offset, size);});
        q_.wait();
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