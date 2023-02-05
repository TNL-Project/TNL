#include <iostream>
#include <TNL/Containers/Partitioner.h>
#include <TNL/Containers/DistributedArray.h>
#include <TNL/MPI/ScopedInitializer.h>

/***
 * The following works for any device (CPU, GPU ...).
 */
template< typename Device >
void distributedArrayExample()
{
   using ArrayType = TNL::Containers::DistributedArray< int, Device >;
   using IndexType = typename ArrayType::IndexType;
   using LocalRangeType = typename ArrayType::LocalRangeType;

   const TNL::MPI::Comm communicator = MPI_COMM_WORLD;

   // We set the global array size to a prime number to force non-uniform distribution.
   const int size = 97;
   const int ghosts = (communicator.size() > 1) ? 4 : 0;

   const LocalRangeType localRange = TNL::Containers::Partitioner< IndexType >::splitRange( size, communicator );
   ArrayType a( localRange, ghosts, size, communicator );
   a.forElements( 0, size, [] __cuda_callable__ ( int idx, int& value ) { value = idx; } );
   std::cout << "Rank " << communicator.rank() << ": " << a.getLocalView() << std::endl;
}

int main( int argc, char* argv[] )
{
   TNL::MPI::ScopedInitializer mpi(argc, argv);

   if( TNL::MPI::GetRank() == 0 )
      std::cout << "The first test runs on CPU ..." << std::endl;
   distributedArrayExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   TNL::MPI::Barrier();

   if( TNL::MPI::GetRank() == 0 )
      std::cout << "The second test runs on GPU ..." << std::endl;
   distributedArrayExample< TNL::Devices::Cuda >();
#endif
}
