#ifndef GPU_HANDLER_H
#define GPU_HANDLER_H 
#include <CL/sycl.hpp>
#include <dgl/array.h>
#include <dgl/bcast.h>
#include <dgl/runtime/config.h>
#include <dgl/runtime/parallel_for.h>
#include <math.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#define show(x) std::cout << "[GPU]: " <<  x << std::endl;
#define show_error(x) std::cout << "[GPU-ERROR]: " <<  x << std::endl;
// #define show_debug(x) std::cout << "[GPU-DEBUG]: " <<  x << std::endl;

#ifndef show_debug
#define show_debug(x)
#endif
struct gpu_handler {
      std::string algo_;
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
#endif
