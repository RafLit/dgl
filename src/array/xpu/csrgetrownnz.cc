#include <dgl/array.h>
#include <atomic>
#include <numeric>
#include <unordered_set>
#include <vector>
#include <sycl/sycl.hpp>

#include "../cpu/array_utils.h"
#include "../../runtime/xpu/xpu_common.h"
namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row) {
  std::cout<<"XPU CSRGetRowNNZ (int64_t)"<<std::endl;
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  return indptr_data[row + 1] - indptr_data[row];
}
template int64_t CSRGetRowNNZ<kDGLXPU, int32_t>(CSRMatrix, int64_t);
template int64_t CSRGetRowNNZ<kDGLXPU, int64_t>(CSRMatrix, int64_t);


template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowNNZ(CSRMatrix csr, NDArray rows) {
  std::cout<<"XPU CSRGetRowNNZ (NDarray)"<<std::endl;
  sycl::queue& q = dgl::runtime::getCurrentXPUStream();
  CHECK_SAME_DTYPE(csr.indices, rows);
  const auto len = rows->shape[0];
  const IdType* vid_data = static_cast<IdType*>(rows->data);
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  NDArray rst = NDArray::Empty({len}, rows->dtype, rows->ctx);
  IdType* rst_data = static_cast<IdType*>(rst->data);
  q.parallel_for(sycl::range<1>(len), [=](int64_t i){
    const auto vid = vid_data[i];
    rst_data[i] = indptr_data[vid + 1] - indptr_data[vid];
  }).wait();
  return rst;
}

template NDArray CSRGetRowNNZ<kDGLXPU, int32_t>(CSRMatrix, NDArray);
template NDArray CSRGetRowNNZ<kDGLXPU, int64_t>(CSRMatrix, NDArray);
}
}
}