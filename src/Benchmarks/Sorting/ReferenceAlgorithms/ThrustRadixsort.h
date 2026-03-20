// FIXME: thrust/sort.h does not work on CUDA 13.2
//#include <thrust/sort.h>
#include <TNL/Containers/Array.h>

namespace TNL {

template< typename ValueType >
struct ThrustRadixsort
{
   static void
   sort( Containers::ArrayView< ValueType, Devices::Cuda >& view )
   {
      //thrust::sort(thrust::device, view.getData(), view.getData() + view.getSize());
      //cudaDeviceSynchronize();
   }
};

} // namespace TNL
