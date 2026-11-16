#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Timer.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
double
mapReduce( Vector< double, Device >& u )
{
   auto u_view = u.getView();
   auto fetch = [ = ] __cuda_callable__( int i ) -> double
   {
      if( i % 2 == 0 )
         return u_view[ i ];
      return 0.0;
   };
   auto reduction = [] __cuda_callable__( const double& a, const double& b )
   {
      return a + b;
   };
   return reduce< Device >( 0, u_view.getSize(), fetch, reduction, 0.0 );
}

int
main( int argc, char* argv[] )
{
   Timer timer;
   Vector< double, Devices::Host > host_u( 100000 );
   host_u = 1.0;
   timer.start();
   double result = mapReduce( host_u );
   timer.stop();
   std::cout << "Host result is:" << result << ". It took " << timer.getRealTime() << " seconds.\n";

#if defined( __CUDACC__ ) || defined( __HIP__ )
   Vector< double, Devices::GPU > gpu_u( 100000 );
   gpu_u = 1.0;
   timer.reset();
   timer.start();
   result = mapReduce( gpu_u );
   timer.stop();
   std::cout << "GPU result is:" << result << ". It took " << timer.getRealTime() << " seconds.\n";
#endif
   return EXIT_SUCCESS;
}
