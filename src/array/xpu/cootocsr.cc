#include <dgl/array.h>
#include "../cpu/array_utils.h"
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/ranges>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/utility>
#include "../../runtime/xpu/xpu_common.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {
template <class IdType>
CSRMatrix UnSortedDenseCOOToCSR(COOMatrix coo) {
  // Unsigned version of the original integer index data type.
  // It avoids overflow in (N + num_threads) and (n_start + n_chunk) below.
  typedef typename std::make_unsigned<IdType>::type UIdType;

  const auto& ctx = coo.row->ctx;
  const auto nbits = coo.row->dtype.bits;

  sycl::queue& q = dgl::runtime::getCurrentXPUStream();
  auto policy = oneapi::dpl::execution::make_device_policy(q);

  const UIdType N = coo.num_rows; const int64_t NNZ = coo.row->shape[0];
  const IdType *const row_data = static_cast<IdType *>(coo.row->data);
  const IdType *const col_data = static_cast<IdType *>(coo.col->data);
  const IdType *const data =
      COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;
  IdArray indptr = Full(0, coo.num_rows + 1, sizeof(IdType)*8, ctx);
  IdArray rowids = Range(0, coo.num_rows, sizeof(IdType)*8, ctx);
  bool row_sorted = coo.row_sorted;
  bool col_sorted = coo.col_sorted;
  if (!row_sorted) {
    coo = COOSort(coo, false);
    col_sorted = coo.col_sorted;
  }
  if (!COOHasData(coo))
    coo.data = aten::Range(0, NNZ, coo.row->dtype.bits, coo.row->ctx);
  auto rowbgn = coo.row.Ptr<IdType>();
  auto rowibgn = rowids.Ptr<IdType>();
  oneapi::dpl::upper_bound(policy, rowbgn, rowbgn + NNZ, rowibgn , rowibgn + coo.num_rows, indptr.Ptr<IdType>()+1);

  return CSRMatrix(
      coo.num_rows, coo.num_cols, indptr, coo.col, coo.data,
      coo.col_sorted);
}

template <DGLDeviceType XPU, typename IdType>
CSRMatrix COOToCSR(COOMatrix coo) {
    return UnSortedDenseCOOToCSR<IdType>(coo);
}
template CSRMatrix COOToCSR<kDGLXPU, int32_t>(COOMatrix coo);
template CSRMatrix COOToCSR<kDGLXPU, int64_t>(COOMatrix coo);
}
}
}