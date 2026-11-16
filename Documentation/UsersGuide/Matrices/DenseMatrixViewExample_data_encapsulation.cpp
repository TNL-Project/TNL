#include <iostream>
#if defined( __CUDACC__ )
   #include <cuda_runtime.h>
#elif defined( __HIP__ )
   #include <hip/hip_runtime.h>
#endif
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
encapsulation()
{
   const int size = 5;

   /***
    * Allocate the dense matrix with no use of TNL
    */
   double* host_data = new double[ size * size ];
   for( int row = 0; row < size; row++ )
      for( int column = 0; column < size; column++ )
         host_data[ row * size + column ] = row * size + column + 1;
   double* data = nullptr;
   if( std::is_same_v< Device, TNL::Devices::Host > ) {
      data = new double[ size * size ];
      memcpy( data, host_data, sizeof( double ) * size * size );
   }
#if defined( __CUDACC__ ) || defined( __HIP__ )
   else if( std::is_same_v< Device, TNL::Devices::GPU > ) {
   #if defined( __CUDACC__ )
      TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &data, sizeof( double ) * size * size ) );
      TNL_BACKEND_SAFE_CALL( cudaMemcpy( data, host_data, sizeof( double ) * size * size, cudaMemcpyHostToDevice ) );
   #else
      TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &data, sizeof( double ) * size * size ) );
      TNL_BACKEND_SAFE_CALL( hipMemcpy( data, host_data, sizeof( double ) * size * size, hipMemcpyHostToDevice ) );
   #endif
   }
#endif

   /***
    * Encapsulate the matrix into DenseMatrixView.
    */
   TNL::Containers::VectorView< double, Device > dataView( data, size * size );
   TNL::Matrices::DenseMatrixView< double, Device, int, TNL::Algorithms::Segments::RowMajorOrder > matrix( 5, 5, dataView );

   std::cout << "Dense matrix view reads as:\n";
   std::cout << matrix << '\n';

   auto f = [ = ] __cuda_callable__( int i ) mutable
   {
      matrix.setElement( i, i, -i );
   };
   TNL::Algorithms::parallelFor< Device >( 0, 5, f );

   std::cout << "Dense matrix view after elements manipulation:\n";
   std::cout << matrix << '\n';

   /***
    * Do not forget to free allocated memory :)
    */
   delete[] host_data;
   if( std::is_same_v< Device, TNL::Devices::Host > )
      delete[] data;
#if defined( __CUDACC__ ) || defined( __HIP__ )
   else if( std::is_same_v< Device, TNL::Devices::GPU > ) {
   #if defined( __CUDACC__ )
      TNL_BACKEND_SAFE_CALL( cudaFree( data ) );
   #else
      TNL_BACKEND_SAFE_CALL( hipFree( data ) );
   #endif
   }
#endif
}

int
main( int argc, char* argv[] )
{
   std::cout << "Dense matrix encapsulation on host:\n";
   encapsulation< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   std::cout << "Dense matrix encapsulation on GPU (" << TNL::Backend::getBackendName() << "):\n";
   encapsulation< TNL::Devices::GPU >();
#endif
}
