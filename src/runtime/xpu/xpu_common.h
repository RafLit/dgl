#include <sycl/sycl.hpp>
namespace dgl {
namespace runtime {
sycl::queue& getCurrentXPUStream();
}
}