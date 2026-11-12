#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/SegmentedScan.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
void
segmentedScan( Array< double, Device >& v, Array< bool, Device >& flags )
{
   /***
    * Reduction is sum of two numbers.
    */
   auto reduce = [] __cuda_callable__( const double& a, const double& b )
   {
      return a + b;
   };

   /***
    * As parameters, we pass array on which the scan is to be performed, interval
    * where the scan is performed, lambda function which is used by the scan and
    * zero as the identity element of the 'sum' operation.
    */
   SegmentedScan< Device >::perform( v, flags, 0, v.getSize(), reduce, 0.0 );
}

int
main( int argc, char* argv[] )
{
   /***
    * Firstly, test the segmented prefix sum with arrays allocated on CPU.
    */
   Array< bool, Devices::Host > host_flags{ 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0 };
   Array< double, Devices::Host > host_v{ 1, 3, 5, 2, 4, 6, 9, 3, 5, 3, 6, 9, 12, 15 };
   std::cout << "host_flags = " << host_flags << '\n';
   std::cout << "host_v     = " << host_v << '\n';
   segmentedScan( host_v, host_flags );
   std::cout << "The segmented prefix sum of the host array is " << host_v << ".\n";

   /***
    * And then also on GPU.
    */
#if defined( __CUDACC__ ) || defined( __HIP__ )
   //Array< bool, Devices::GPU > gpu_flags{ 1,0,0,1,0,0,0,1,0,1,0,0, 0, 0 };
   //Array< double, Devices::GPU > gpu_v { 1,3,5,2,4,6,9,3,5,3,6,9,12,15 };
   //std::cout << "gpu_flags = " << gpu_flags << '\n';
   //std::cout << "gpu_v     = " << gpu_v << '\n';
   //segmentedScan( gpu_v, gpu_flags );
   //std::cout << "The segmnted prefix sum of the GPU array is " << gpu_v << ".\n";
#endif
   return EXIT_SUCCESS;
}
