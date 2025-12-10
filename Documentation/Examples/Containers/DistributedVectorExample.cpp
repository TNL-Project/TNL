#include <iostream>
#include <TNL/Containers/BlockPartitioning.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/MPI/ScopedInitializer.h>

/***
 * The following works for any device (CPU, GPU ...).
 */
template< typename Device >
void
distributedVectorExample()
{
   using VectorType = TNL::Containers::DistributedVector< int, Device >;
   using IndexType = typename VectorType::IndexType;
   using LocalRangeType = typename VectorType::LocalRangeType;

   const TNL::MPI::Comm communicator = MPI_COMM_WORLD;

   // We set the global vector size to a prime number to force non-uniform distribution.
   const int size = 97;
   const int ghosts = ( communicator.size() > 1 ) ? 4 : 0;

   const LocalRangeType localRange = TNL::Containers::splitRange< IndexType >( size, communicator );
   VectorType v( localRange, ghosts, size, communicator );
   v.forElements( 0,
                  size,
                  [] __cuda_callable__( int idx, int& value )
                  {
                     value = idx;
                  } );
   std::cout << "Rank " << communicator.rank() << " has subrange " << localRange << '\n';
   const int sum = TNL::sum( v );

   if( communicator.rank() == 0 )
      std::cout << "Global sum is " << sum << '\n';
}

int
main( int argc, char* argv[] )
{
   TNL::MPI::ScopedInitializer mpi( argc, argv );

   if( TNL::MPI::GetRank() == 0 )
      std::cout << "The first test runs on CPU ...\n";
   distributedVectorExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   TNL::MPI::Barrier();

   if( TNL::MPI::GetRank() == 0 )
      std::cout << "The second test runs on GPU ...\n";
   distributedVectorExample< TNL::Devices::Cuda >();
#endif
}
