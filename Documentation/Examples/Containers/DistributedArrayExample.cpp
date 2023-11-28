#include <iostream>
#include <TNL/Containers/BlockPartitioning.h>
#include <TNL/Containers/DistributedArray.h>
#include <TNL/MPI/ScopedInitializer.h>

/***
 * The following works for any device (CPU, GPU ...).
 */
template< typename Device >
void
distributedArrayExample()
{
   using ArrayType = TNL::Containers::DistributedArray< int, Device >;
   using IndexType = typename ArrayType::IndexType;
   using LocalRangeType = typename ArrayType::LocalRangeType;

   const TNL::MPI::Comm communicator = MPI_COMM_WORLD;

   // We set the global array size to a prime number to force non-uniform distribution.
   const int size = 97;
   const int ghosts = ( communicator.size() > 1 ) ? 4 : 0;

   const LocalRangeType localRange = TNL::Containers::splitRange< IndexType >( size, communicator );
   ArrayType a( localRange, ghosts, size, communicator );
   a.forElements( 0,
                  size,
                  [] __cuda_callable__( int idx, int& value )
                  {
                     value = idx;
                  } );
   ArrayType b( localRange, ghosts, size, communicator );
   b.forElements( 0,
                  size,
                  [] __cuda_callable__( int idx, int& value )
                  {
                     value = idx - ( idx == 90 );
                  } );
   for( int i = 0; i < communicator.size(); i++ ) {
      if( communicator.rank() == i )
         std::cout << "MPI rank = " << communicator.rank() << std::endl
                   << " size = " << a.getSize() << std::endl
                   << " local range = " << a.getLocalRange().getBegin() << " - " << a.getLocalRange().getEnd() << std::endl
                   << " ghosts = " << a.getGhosts() << std::endl
                   << " local data = " << a.getLocalView() << std::endl
                   << " local data with ghosts = " << a.getLocalViewWithGhosts() << std::endl;
      TNL::MPI::Barrier();
   }
}

int
main( int argc, char* argv[] )
{
   TNL::MPI::ScopedInitializer mpi( argc, argv );

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
