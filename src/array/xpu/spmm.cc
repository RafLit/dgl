
#include <dgl/array.h>

#include "../cpu/spmm_binary_ops.h"
#include <iostream>
#include "../../runtime/xpu/xpu_common.h"
#include <sycl/sycl.hpp>
namespace dgl {
namespace aten {

/** @brief Generalized SpMM on Csr format. */
template <int XPU, typename IdType, typename DType>
void SpMMCsr(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
        SWITCH_OP(op, Op, {
            sycl::queue& q = dgl::runtime::getCurrentXPUStream();
            std::cout<<"xpu spmm "<<std::endl;

            const DType* X = ufeat.Ptr<DType>();
            const DType* W = efeat.Ptr<DType>();
            DType* O = out.Ptr<DType>();
            int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len, rhs_dim = bcast.rhs_len;


            const bool has_idx = !IsNullArray(csr.data);
            const IdType* indptr = csr.indptr.Ptr<IdType>();
            const IdType* indices = csr.indices.Ptr<IdType>();
            const IdType* edges = csr.data.Ptr<IdType>();
            //NDArray lhs_offset = NDArray::Empty({bcast.lhs_len},DGLDataType{kDGLFloat, 64, 1}, csr.indptr->ctx);
            //NDArray rhs_offset = NDArray::Empty({bcast.rhs_len}, DGLDataType{kDGLFloat, 64, 1}, csr.indptr->ctx);
            auto lhs_offset = static_cast<int64_t*>(sycl::aligned_alloc_device(512, bcast.lhs_len*sizeof(int64_t), q));
            auto rhs_offset = static_cast<int64_t*>(sycl::aligned_alloc_device(512, bcast.rhs_len*sizeof(int64_t), q));
            q.wait();
            q.copy(static_cast<const int64_t*>(bcast.lhs_offset.data()), lhs_offset, bcast.lhs_offset.size());
            q.copy(static_cast<const int64_t*>(bcast.rhs_offset.data()), rhs_offset, bcast.rhs_offset.size());
            q.parallel_for(sycl::range<1>(csr.num_rows), [=,use_lhs=Op::use_lhs, use_rhs=Op::use_rhs, use_bcast=bcast.use_bcast](sycl::id<1> rid){
                const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
                DType* out_off = O + rid * dim;
                for (IdType j = row_start; j < row_end; ++j) {
                    const IdType cid = indices[j];
                    const IdType eid = has_idx ? edges[j] : j;
                    for (int64_t k = 0; k < dim; ++k) {
                    const int64_t lhs_add = use_bcast ? lhs_offset[k] : k;
                    const int64_t rhs_add = use_bcast ? rhs_offset[k] : k;
                
                
                    const DType* lhs_off =  use_lhs ? X + cid * lhs_dim + lhs_add : nullptr;
                    const DType* rhs_off =  use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
                
                    out_off[k] += Op::Call(lhs_off, rhs_off);
                    
                    }
                }
            }).wait();
        sycl::free(lhs_offset,q);
        sycl::free(rhs_offset,q);
        });
    }


template void SpMMCsr<kDGLXPU, int32_t, BFloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLXPU, int64_t, BFloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLXPU, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLXPU, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLXPU, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLXPU, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
}
}