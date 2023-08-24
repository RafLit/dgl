/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/sddmm.h
 * @brief SDDMM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SDDMM_H_
#define DGL_ARRAY_CPU_SDDMM_H_

#include <CL/sycl.hpp>
#include <dgl/array.h>
#include <dgl/bcast.h>
#include <dgl/runtime/parallel_for.h>

#include "../selector.h"

#include <iostream>
#define show(x) std::cout << "[GPU]: " <<  x << std::endl;
#define show_error(x) std::cout << "[GPU-ERROR]: " <<  x << std::endl;
// #define show_debug(x) std::cout << "[GPU-DEBUG]: " <<  x << std::endl;

#ifndef show_debug
#define show_debug(x)
#endif


struct gpu_handler {
      std::atomic<uint64_t> cp_time;
      std::unique_ptr<sycl::device> dev;
      std::unique_ptr<sycl::queue> qptr;
      gpu_handler() { 

             if(std::getenv("SELECTOR_CPU"))
             {
                dev = std::make_unique<sycl::device>( sycl::cpu_selector{} );                     
             } else {
                dev = std::make_unique<sycl::device>( sycl::gpu_selector{} ); 
             }              
             
             qptr =  std::make_unique<sycl::queue>(*dev);
             if(!qptr)
             {
                show_error("Can't create stream ");
             }
             show_debug("stream created :)");

      }

      ~gpu_handler() {
                info();
      }

      void info() {

        if(dev)
        {
          show("Name =" << dev->get_info<sycl::info::device::name>() << " mem size = " << dev->get_info<sycl::info::device::global_mem_size>());
        }
      }

       void copy(sycl::queue &q, void *dst, const void *src, size_t size) {
       q.submit([&](sycl::handler &h) { h.memcpy(dst, src, size); });
       q.wait();
       }

      void copy(void *dst, const void *src, size_t size) {
          copy(*qptr, dst, src, size);
      }

      void* alloc_bytes(size_t size) {

            void *mem = nullptr; 
            mem = sycl::aligned_alloc_device(64, size, *qptr);
            if(!mem)
            {
              show_error("Can't allocate memory!! " << size);
            }
            //qptr->memset(mem,0,size);
            show_debug("alloc "<< size);
            return mem;
      }

      template<class T>
      T* alloc(size_t size) {
            return reinterpret_cast<T*>(alloc_bytes(size*sizeof(T)));
      }
     


      void dealloc(void *mem) {
          sycl::free(mem,*qptr);
          show_debug("dealloc mem");
      }


     template<class F>
     void submit_for(size_t N, F f) {
             show_debug("submit work " << N); 
             qptr->submit([&](sycl::handler &h){ h.parallel_for(N,f); });
             qptr->wait();
     } 

};




namespace dgl {
namespace aten {
namespace cpu {
static gpu_handler gpu = gpu_handler();

template<class T>
struct SyclFree {
   void operator()(T* ptr) const {  if(ptr) gpu.dealloc(ptr); }
};

template<class T>
using sycl_ptr = std::unique_ptr<T,SyclFree<T>>;
// using sycl_ptr = std::unique_ptr<T>;


template<class T> 
sycl_ptr<T> make_sycl_ptr_from_nd(const dgl::IdArray& tab)
{
    sycl_ptr<T> ptr(gpu.alloc<T>(tab.NumElements()));   
    gpu.copy(ptr.get(),tab.Ptr<T>(),tab.NumElements()*sizeof(T));
    return ptr;
}


template <typename DType>
using AccType = typename std::conditional<
    std::is_same<DType, BFloat16>::value, float, DType>::type;

/**
 * @brief CPU kernel of g-SDDMM on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param lhs The left hand side operand feature.
 * @param rhs The right hand size operand feature.
 * @param out The result feature on edges.
 * @note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 */
template <
    typename IdType, typename DType, typename Op, int LhsTarget = 0,
    int RhsTarget = 2>
void SDDMMCsr(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray lhs, NDArray rhs,
    NDArray out) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  const IdType* edges = csr.data.Ptr<IdType>();
  const DType* X = lhs.Ptr<DType>();
  const DType* Y = rhs.Ptr<DType>();
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len, reduce_size = bcast.reduce_size;
  std::cout<<"SDDMMCSR"<<std::endl;
  DType* O = out.Ptr<DType>();
  runtime::parallel_for(0, csr.num_rows, [=](IdType b, IdType e) {
    for (auto rid = b; rid < e; ++rid) {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const IdType eid = has_idx ? edges[j] : j;
        DType* out_off = O + eid * dim;
        for (int64_t k = 0; k < dim; ++k) {
          const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
          const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
          const DType* lhs_off =
              Op::use_lhs
                  ? X + Selector<LhsTarget>::Call(rid, eid, cid) * lhs_dim +
                        lhs_add * reduce_size
                  : nullptr;
          const DType* rhs_off =
              Op::use_rhs
                  ? Y + Selector<RhsTarget>::Call(rid, eid, cid) * rhs_dim +
                        rhs_add * reduce_size
                  : nullptr;
          out_off[k] = Op::Call(lhs_off, rhs_off, reduce_size);
        }
      }
    }
  });
}

/**
 * @brief CPU kernel of g-SDDMM on Coo format.
 * @param bcast Broadcast information.
 * @param coo The COO matrix.
 * @param lhs The left hand side operand feature.
 * @param rhs The right hand size operand feature.
 * @param out The result feature on edges.
 * @note it uses edge parallel strategy, different threads are responsible
 *       for the computation of different edges.
 */
template <
    typename IdType, typename DType, typename Op, int LhsTarget = 0,
    int RhsTarget = 2>
void SDDMMCoo(
    const BcastOff& bcast, const COOMatrix& coo, NDArray lhs, NDArray rhs,
    NDArray out) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdType* edges = coo.data.Ptr<IdType>();
  int rows = coo.row->shape[0];

  auto gpu_row = make_sycl_ptr_from_nd<IdType>(coo.row);
  auto gpu_col = make_sycl_ptr_from_nd<IdType>(coo.col);
  auto gpu_edges = has_idx?make_sycl_ptr_from_nd<IdType>(coo.data):nullptr;

  const DType* X = lhs.Ptr<DType>();
  const DType* Y = rhs.Ptr<DType>();
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len, reduce_size = bcast.reduce_size;
  std::cout<<"SDDMMCoo"<<std::endl;
  sycl_ptr<DType> gpu_O(gpu.alloc<DType>(rows*dim));

  row = gpu_row.get();
  col = gpu_col.get();
  edges = gpu_edges.get();

  edges = gpu_edges.get();
  DType* O = out.Ptr<DType>();
  DType* tmp_output = O;

  if(gpu_O)
  {
     tmp_output = gpu_O.get();
  }

  const int64_t* tmp_rhs_offset = bcast.rhs_offset.data();
  const int64_t* tmp_lhs_offset = bcast.lhs_offset.data();
  sycl_ptr<int64_t> lhs_offset((bcast.use_bcast) ? gpu.alloc<int64_t>(dim) : nullptr);
  sycl_ptr<int64_t> rhs_offset((bcast.use_bcast) ? gpu.alloc<int64_t>(dim) : nullptr);
  if(lhs_offset) {
       gpu.copy(lhs_offset.get(),tmp_lhs_offset,sizeof(int64_t)*dim);      
       tmp_lhs_offset = lhs_offset.get();
  }
  if(rhs_offset) {
       gpu.copy(rhs_offset.get(),tmp_rhs_offset,sizeof(int64_t)*dim);      
       tmp_rhs_offset = rhs_offset.get();
  }
//#pragma omp parallel for
  gpu.submit_for(rows,[=,OutPut=tmp_output,use_lhs=Op::use_lhs, use_rhs=Op::use_rhs, use_bcast=bcast.use_bcast](sycl::id<1> i){
    IdType j = i;
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx ? edges[j] : j;
    DType* out_off = OutPut + eid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = use_bcast ? tmp_lhs_offset[k] : k;
      const int64_t rhs_add = use_bcast ? tmp_rhs_offset[k] : k;
      const DType* lhs_off =
          use_lhs ? X + Selector<LhsTarget>::Call(rid, eid, cid) * lhs_dim +
                            lhs_add * reduce_size
                      : nullptr;
      const DType* rhs_off =
          use_rhs ? Y + Selector<RhsTarget>::Call(rid, eid, cid) * rhs_dim +
                            rhs_add * reduce_size
                      : nullptr;
      out_off[k] = Op::Call(lhs_off, rhs_off, reduce_size);
    }
  });
  if(gpu_O)
  {
     gpu.copy(O,gpu_O.get(),sizeof(DType)*rows*dim);
  }

}

namespace op {

////////////////////////// binary operators on CPU /////////////////////////////
template <typename DType>
struct Add {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    return *lhs_off + *rhs_off;
  }
};

template <typename DType>
struct Sub {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    return *lhs_off - *rhs_off;
  }
};

template <typename DType>
struct Mul {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    return *lhs_off * *rhs_off;
  }
};

template <typename DType>
struct Div {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    return *lhs_off / *rhs_off;
  }
};

template <typename DType>
struct CopyLhs {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  inline static DType Call(
      const DType* lhs_off, const DType*, int64_t len = 1) {
    return *lhs_off;
  }
};

template <typename DType>
struct CopyRhs {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType*, const DType* rhs_off, int64_t len = 1) {
    return *rhs_off;
  }
};

template <typename DType>
struct Dot {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    DType rst = 0;
    for (int64_t l = 0; l < len; ++l) {
      rst += lhs_off[l] * rhs_off[l];
    }
    return rst;
  }
};

#define SWITCH_OP(op, Op, ...)                                   \
  do {                                                           \
    if ((op) == "add") {                                         \
      typedef dgl::aten::cpu::op::Add<DType> Op;                 \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "sub") {                                  \
      typedef dgl::aten::cpu::op::Sub<DType> Op;                 \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "mul") {                                  \
      typedef dgl::aten::cpu::op::Mul<DType> Op;                 \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "div") {                                  \
      typedef dgl::aten::cpu::op::Div<DType> Op;                 \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "copy_lhs") {                             \
      typedef dgl::aten::cpu::op::CopyLhs<DType> Op;             \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "copy_rhs") {                             \
      typedef dgl::aten::cpu::op::CopyRhs<DType> Op;             \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "dot") {                                  \
      typedef dgl::aten::cpu::op::Dot<DType> Op;                 \
      { __VA_ARGS__ }                                            \
    } else {                                                     \
      LOG(FATAL) << "Unsupported SDDMM binary operator: " << op; \
    }                                                            \
  } while (0)

}  // namespace op

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SDDMM_H_
