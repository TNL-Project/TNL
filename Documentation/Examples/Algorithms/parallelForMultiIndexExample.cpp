#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/parallelFor.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
void initMeshFunction( const int xSize,
                       const int ySize,
                       const int zSize,
                       Vector< double, Device >& v,
                       const double& c )
{
   auto view = v.getView();
   auto init = [=] __cuda_callable__ ( const StaticArray< 3, int >& i ) mutable
   {
      view[ ( i.z() * ySize + i.y() ) * xSize + i.x() ] = c;
   };
   StaticArray< 3, int > begin{ 0, 0, 0 };
   StaticArray< 3, int > end{ xSize, ySize, zSize };
   parallelFor< Device >( begin, end, init );
}

int main( int argc, char* argv[] )
{
   /***
    * Define dimensions of a 3D mesh function.
    */
   const int xSize( 10 ), ySize( 10 ), zSize( 10 );
   const int size = xSize * ySize * zSize;

   /***
    * Firstly, test the mesh function initiation on CPU.
    */
   Vector< double, Devices::Host > host_v( size );
   initMeshFunction( xSize, ySize, zSize, host_v, 1.0 );

   /***
    * And then also on GPU.
    */
#ifdef __CUDACC__
   Vector< double, Devices::Cuda > cuda_v( size );
   initMeshFunction( xSize, ySize, zSize, cuda_v, 1.0 );
#endif
   return EXIT_SUCCESS;
}
