#include <dgl/array.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/parallel_for.h>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/ranges>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/utility>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include <numeric>
#include "../../runtime/xpu/xpu_common.h"

using namespace oneapi::dpl::experimental::ranges;

namespace dgl {
using runtime::NDArray;
using runtime::parallel_for;
namespace aten {
namespace impl {


template <DGLDeviceType XPU, typename DType>
NDArray Full(DType val, int64_t length, DGLContext ctx) {
    sycl::queue &q = dgl::runtime::getCurrentXPUStream();
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    NDArray ret = NDArray::Empty({length}, DGLDataTypeTraits<DType>::dtype, ctx);
    DType* p = ret.Ptr<DType>();
    std::fill(policy, p, p + length, val);
    return ret;
}

template NDArray Full<kDGLXPU, int32_t>(
    int32_t val, int64_t length, DGLContext ctx);
template NDArray Full<kDGLXPU, int64_t>(
    int64_t val, int64_t length, DGLContext ctx);

template <DGLDeviceType XPU, typename IdType>
IdArray Range(IdType low, IdType high, DGLContext ctx) {
    sycl::queue &q = dgl::runtime::getCurrentXPUStream();
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    CHECK(high >= low) << "high must be bigger than low";
    IdArray ret = NewIdArray(high - low, ctx, sizeof(IdType) * 8);

    auto i = oneapi::dpl::counting_iterator(low);
    std::copy(policy, i, i+(high - low), ret.Ptr<IdType>());
    return ret;
}

template IdArray Range<kDGLXPU, int32_t>(int32_t, int32_t, DGLContext);
template IdArray Range<kDGLXPU, int64_t>(int64_t, int64_t, DGLContext);
}
}
}