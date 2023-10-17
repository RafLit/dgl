#include <dgl/array.h>
#include <iostream>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <tuple>

#include "../../runtime/xpu/xpu_common.h"

//using namespace oneapi::dpl::execution;

namespace dgl {
namespace aten {
namespace impl {

///////////////////////////// COOSort_ /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
void COOSort_(COOMatrix* coo, bool sort_column) {
    sycl::queue& q = dgl::runtime::getCurrentXPUStream();
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    const int64_t nnz = coo->row->shape[0];
    if (!COOHasData(*coo))
        coo->data = aten::Range(0, nnz, coo->row->dtype.bits, coo->row->ctx);
    auto zipped_begin = oneapi::dpl::make_zip_iterator(coo->row.Ptr<IdType>(), coo->col.Ptr<IdType>(), coo->data.Ptr<IdType>());
    auto zipped_end = zipped_begin + nnz;
    typedef std::tuple<IdType, IdType, IdType> mytuple;
    if (sort_column) {
        std::sort(policy, zipped_begin, zipped_end, [](const mytuple& a, const mytuple& b) {
            return (std::get<0>(a) != std::get<0>(b))
                        ? (std::get<0>(a) < std::get<0>(b))
                        : (std::get<1>(a) < std::get<1>(b));
            });
    } else {
        std::sort(policy, zipped_begin, zipped_end, 
            [](const mytuple& a, const mytuple& b) {
            return std::get<0>(a) < std::get<0>(b);
            });
    }
    coo->row_sorted = true;
    coo->col_sorted = sort_column;
    q.wait();
}

template void COOSort_<kDGLXPU, int32_t>(COOMatrix*, bool);
template void COOSort_<kDGLXPU, int64_t>(COOMatrix*, bool);

///////////////////////////// COOIsSorted /////////////////////////////


}  // namespace impl
}  // namespace aten
}  // namespace dgl
