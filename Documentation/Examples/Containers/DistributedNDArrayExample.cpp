#include <iostream>
#include <TNL/Containers/Partitioner.h>
#include <TNL/Containers/DistributedNDArray.h>
#include <TNL/Containers/DistributedNDArraySynchronizer.h>
#include <TNL/MPI/ScopedInitializer.h>

// The following works for any device (Host, Cuda, etc.)
template< typename Device >
void
distributedNDArrayExample()
{
   using namespace TNL::Containers;
   using LocalArrayType = NDArray< int,                          // Value
                                   SizesHolder< int, 0, 0 >,     // Sizes
                                   std::index_sequence< 1, 0 >,  // Permutation
                                   Device,                       // Device
                                   int,                          // Index
                                   std::index_sequence< 1, 0 >   // Overlaps
                                   >;
   using ArrayType = DistributedNDArray< LocalArrayType >;
   using LocalRangeType = typename ArrayType::LocalRangeType;

   // set input parameters
   const TNL::MPI::Comm communicator = MPI_COMM_WORLD;
   const int num_rows = 10;            // number of rows
   const int num_cols = 4;             // number of columns
   constexpr int distributedAxis = 0;  // 0: num_rows gets distributed, 1: num_cols gets distributed

   // decompose the range of rows
   const LocalRangeType localRange =
      TNL::Containers::Partitioner< typename LocalArrayType::IndexType >::splitRange( num_rows, communicator );

   // create the distributed array
   ArrayType a;
   a.setSizes( num_rows, num_cols );
   a.template setDistribution< distributedAxis >( localRange.getBegin(), localRange.getEnd(), communicator );
   a.allocate();

   // do some work with the array
   auto a_view = a.getLocalView();
   a.forAll(
      [ = ] __cuda_callable__( int gi, int j ) mutable
      {
         // convert global row index to local
         const int i = gi - localRange.getBegin();
         // write the global row index to the array using local coordinates
         a_view( i, j ) = gi;
      } );
   a.forGhosts(
      [ = ] __cuda_callable__( int gi, int j ) mutable
      {
         // convert global row index to local
         const int i = gi - localRange.getBegin();
         // use the local indices to set ghost elements to -1
         a_view( i, j ) = -1;
      } );

   // output the local elements as a flat array
   using ArrayView = TNL::Containers::ArrayView< int, Device, int >;
   ArrayView flat_view( a_view.getData(), a_view.getStorageSize() );
   std::cout << "Rank " << communicator.rank() << " before synchronization: " << flat_view << std::endl;

   // synchronize the ghost regions and output again
   DistributedNDArraySynchronizer< ArrayType > synchronizer;
   synchronizer.synchronize( a );
   std::cout << "Rank " << communicator.rank() << " after synchronization: " << flat_view << std::endl;
}

int
main( int argc, char* argv[] )
{
   TNL::MPI::ScopedInitializer mpi( argc, argv );

   if( TNL::MPI::GetRank() == 0 )
      std::cout << "The first test runs on CPU ..." << std::endl;
   distributedNDArrayExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   TNL::MPI::Barrier();

   if( TNL::MPI::GetRank() == 0 )
      std::cout << "The second test runs on GPU ..." << std::endl;
   distributedNDArrayExample< TNL::Devices::Cuda >();
#endif
}
