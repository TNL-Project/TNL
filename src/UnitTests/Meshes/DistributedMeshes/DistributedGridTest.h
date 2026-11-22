#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Containers/Array.h>
#include <TNL/MPI/Wrappers.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGrid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGridSynchronizer.h>
#include <TNL/Meshes/DistributedMeshes/reduce.h>

namespace DistributedGridTest {

using namespace TNL;
using namespace TNL::Meshes;
using namespace TNL::Meshes::DistributedMeshes;

// Value of an entity given by its global coordinates and orientation. The values are integers, so that they are
// represented exactly. It is a class instead of a lambda function, because nvcc does not allow local types in
// the template arguments of the functions with extended lambda functions.
template< typename Grid >
struct EntityValue
{
   using CoordinatesType = typename Grid::CoordinatesType;

   // shift from the local to the global coordinates
   CoordinatesType shift;

   template< typename Entity >
   __cuda_callable__
   double
   operator()( const Entity& entity ) const
   {
      const CoordinatesType coordinates = entity.getCoordinates() + shift;
      double value = entity.getOrientationIndex() + 1;
      for( int i = 0; i < Grid::getMeshDimension(); i++ )
         value = value * 1000 + coordinates[ i ];
      return value;
   }
};

// Sets the values of the entities to value( entity ), only the entities with the orientation index `orientation`
// are set if it is non-negative.
template< typename Grid, typename View >
struct SetValues
{
   EntityValue< Grid > value;
   mutable View view;
   int orientation;

   template< typename Entity >
   __cuda_callable__
   void
   operator()( const Entity& entity ) const
   {
      if( orientation < 0 || entity.getOrientationIndex() == orientation )
         view[ entity.getIndex() ] = value( entity );
   }
};

template< typename View >
struct ReadValue
{
   View view;

   template< typename Entity >
   __cuda_callable__
   double
   operator()( const Entity& entity ) const
   {
      return view[ entity.getIndex() ];
   }
};

template< typename View >
struct WriteValue
{
   mutable View view;

   template< typename Entity >
   __cuda_callable__
   void
   operator()( const Entity& entity, double value ) const
   {
      view[ entity.getIndex() ] = value;
   }
};

// Counts the entities in the traversal.
struct Counter
{
   int* count;

   template< typename Entity >
   __cuda_callable__
   void
   operator()( const Entity& entity ) const
   {
      ( *count )++;
   }
};

// Counts the entities for which isBoundary() differs from the expected value.
struct BoundaryChecker
{
   int* errors;
   bool expected;

   template< typename Entity >
   __cuda_callable__
   void
   operator()( const Entity& entity ) const
   {
      if( entity.isBoundary() != expected )
         ( *errors )++;
   }
};

template< typename Grid >
typename Grid::CoordinatesType
getGlobalSizes()
{
   if constexpr( Grid::getMeshDimension() == 1 )
      return { 23 };
   else if constexpr( Grid::getMeshDimension() == 2 )
      return { 17, 13 };
   else
      return { 11, 9, 10 };
}

template< typename Grid >
class DistributedGridTest : public ::testing::Test
{
protected:
   using GridType = Grid;
   using DistributedGridType = DistributedMesh< Grid >;
};

template< int Dimension, typename Device >
using TestGrid = Grid< Dimension, double, Device, int >;

using GridTypes = ::testing::Types<
#if defined( __CUDACC__ )
   TestGrid< 1, Devices::Cuda >,
   TestGrid< 2, Devices::Cuda >,
   TestGrid< 3, Devices::Cuda >,
   Grid< 3, float, Devices::Cuda, long >
#elif defined( __HIP__ )
   TestGrid< 1, Devices::Hip >,
   TestGrid< 2, Devices::Hip >,
   TestGrid< 3, Devices::Hip >,
   Grid< 3, float, Devices::Hip, long >
#else
   TestGrid< 1, Devices::Host >,
   TestGrid< 2, Devices::Host >,
   TestGrid< 3, Devices::Host >,
   Grid< 3, float, Devices::Host, long >
#endif
   >;

TYPED_TEST_SUITE( DistributedGridTest, GridTypes );

// Checks the decomposition of the global grid and the ownership of the entities. The traversals of the local
// grid run on the host, so the test uses the grid with the same parameters on the host.
TYPED_TEST( DistributedGridTest, decomposition )
{
   using Grid = typename TestFixture::GridType;
   using HostGrid =
      Meshes::Grid< Grid::getMeshDimension(), typename Grid::RealType, Devices::Sequential, typename Grid::IndexType >;
   using DistributedGridType = DistributedMesh< HostGrid >;
   using CoordinatesType = typename HostGrid::CoordinatesType;
   constexpr int Dimension = HostGrid::getMeshDimension();

   HostGrid globalGrid;
   globalGrid.setSizes( getGlobalSizes< HostGrid >() );
   globalGrid.setDomain( typename HostGrid::PointType( 0.0 ), typename HostGrid::PointType( 1.0 ) );
   DistributedGridType distributedGrid;
   distributedGrid.setGlobalGrid( globalGrid );
   distributedGrid.setGhostLevels( 1 );
   const HostGrid& localGrid = distributedGrid.getLocalMesh();
   const MPI::Comm& communicator = distributedGrid.getCommunicator();

   // the subdomains cover the global grid without overlaps
   int localCells = 1;
   for( int i = 0; i < Dimension; i++ )
      localCells *= distributedGrid.getLocalSize()[ i ];
   int cells = 0;
   MPI::Allreduce( &localCells, &cells, 1, MPI_SUM, communicator );
   EXPECT_EQ( cells, globalGrid.template getEntitiesCount< Dimension >() );
   EXPECT_EQ( distributedGrid.isDistributed(), communicator.size() > 1 );

   // the local grid consists of the subdomain and the overlaps, which are only between the subdomains
   EXPECT_EQ(
      localGrid.getSizes(),
      distributedGrid.getLocalSize() + distributedGrid.getLowerOverlap() + distributedGrid.getUpperOverlap() );
   EXPECT_EQ( localGrid.getLocalBegin(), distributedGrid.getLowerOverlap() );
   EXPECT_EQ( localGrid.getLocalEnd(), localGrid.getSizes() - distributedGrid.getUpperOverlap() );
   EXPECT_EQ( localGrid.getSpaceSteps(), globalGrid.getSpaceSteps() );
   const auto& subdomain = distributedGrid.getSubdomainCoordinates();
   const auto& decomposition = distributedGrid.getDomainDecomposition();
   for( int i = 0; i < Dimension; i++ ) {
      const bool lower = subdomain[ i ] == 0;
      const bool upper = subdomain[ i ] == decomposition[ i ] - 1;
      EXPECT_EQ( localGrid.getLowerGlobalBoundarySides()[ i ], lower );
      EXPECT_EQ( localGrid.getUpperGlobalBoundarySides()[ i ], upper );
      EXPECT_EQ( distributedGrid.getLowerOverlap()[ i ], lower ? 0 : 1 );
      EXPECT_EQ( distributedGrid.getUpperOverlap()[ i ], upper ? 0 : 1 );
      EXPECT_NEAR(
         localGrid.getOrigin()[ i ],
         globalGrid.getOrigin()[ i ]
            + globalGrid.getSpaceSteps()[ i ]
                 * ( distributedGrid.getGlobalBegin()[ i ] - distributedGrid.getLowerOverlap()[ i ] ),
         1e-6 );
   }

   // the neighbors
   for( int i = 0; i < distributedGrid.getNeighborsCount(); i++ ) {
      const CoordinatesType direction( Directions::template getXYZ< Dimension >( i ) );
      const CoordinatesType neighbor = subdomain + direction;
      bool exists = true;
      for( int j = 0; j < Dimension; j++ )
         if( neighbor[ j ] < 0 || neighbor[ j ] >= decomposition[ j ] )
            exists = false;
      EXPECT_EQ( distributedGrid.getNeighbors()[ i ], exists ? distributedGrid.getRankOfProcCoord( neighbor ) : -1 );
   }

   // each entity is owned by exactly one subdomain, the local entities are either interior or boundary
   Algorithms::staticFor< int, 0, Dimension + 1 >(
      [ & ]( auto entityDimension )
      {
         constexpr int EntityDimension = decltype( entityDimension )::value;
         SCOPED_TRACE( "Entity dimension: " + std::to_string( EntityDimension ) );
         int local = 0;
         int interior = 0;
         int boundary = 0;
         int globalBoundary = 0;
         int errors = 0;
         forLocalEntities< EntityDimension >( localGrid, Counter{ &local } );
         forInteriorEntities< EntityDimension >( localGrid, Counter{ &interior } );
         forBoundaryEntities< EntityDimension >( localGrid, Counter{ &boundary } );
         forBoundaryEntities< EntityDimension >( globalGrid, Counter{ &globalBoundary } );
         forInteriorEntities< EntityDimension >( localGrid, BoundaryChecker{ &errors, false } );
         forBoundaryEntities< EntityDimension >( localGrid, BoundaryChecker{ &errors, true } );
         EXPECT_EQ( interior + boundary, local );
         EXPECT_EQ( errors, 0 );
         int sum = 0;
         MPI::Allreduce( &local, &sum, 1, MPI_SUM, communicator );
         EXPECT_EQ( sum, globalGrid.template getEntitiesCount< EntityDimension >() );
         sum = 0;
         MPI::Allreduce( &boundary, &sum, 1, MPI_SUM, communicator );
         EXPECT_EQ( sum, globalBoundary );
      } );
}

TYPED_TEST( DistributedGridTest, overlapsOnGlobalBoundary )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;
   Grid globalGrid;
   globalGrid.setSizes( getGlobalSizes< Grid >() );
   DistributedGridType distributedGrid;
   distributedGrid.setGlobalGrid( globalGrid );
   if( distributedGrid.isBoundarySubdomain() )
      EXPECT_THROW( distributedGrid.setOverlaps( 1, 1 ), std::invalid_argument );
   else
      EXPECT_NO_THROW( distributedGrid.setOverlaps( 1, 1 ) );
}

// A global grid with less cells than subdomains along some axis is rejected and the distributed grid is not changed.
TYPED_TEST( DistributedGridTest, tooSmallGlobalGrid )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;
   constexpr int Dimension = Grid::getMeshDimension();
   const int nproc = MPI::GetSize( MPI_COMM_WORLD );

   Grid globalGrid;
   globalGrid.setSizes( getGlobalSizes< Grid >() );
   DistributedGridType distributedGrid;
   distributedGrid.setGlobalGrid( globalGrid );
   const auto decomposition = distributedGrid.getDomainDecomposition();

   // the grid has one cell less than subdomains along the axes with more than one subdomain
   typename Grid::CoordinatesType smallSizes = decomposition - 1;
   for( int i = 0; i < Dimension; i++ )
      if( decomposition[ i ] == 1 )
         smallSizes[ i ] = 1;
   Grid smallGrid;
   smallGrid.setSizes( smallSizes );
   if( nproc > 1 ) {
      DistributedGridType small;
      small.setDomainDecomposition( decomposition );
      EXPECT_THROW( small.setGlobalGrid( smallGrid ), std::invalid_argument );
      // the original distributed grid stays valid after a failed call
      EXPECT_THROW( distributedGrid.setGlobalGrid( smallGrid ), std::invalid_argument );
      EXPECT_EQ( distributedGrid.getGlobalGrid().getSizes(), globalGrid.getSizes() );
      EXPECT_EQ( distributedGrid.getDomainDecomposition(), decomposition );
   }
   else {
      DistributedGridType small;
      EXPECT_NO_THROW( small.setGlobalGrid( smallGrid ) );
   }

   // the grid with exactly one cell per subdomain is accepted
   Grid minimalGrid;
   minimalGrid.setSizes( decomposition );
   DistributedGridType minimal;
   minimal.setDomainDecomposition( decomposition );
   EXPECT_NO_THROW( minimal.setGlobalGrid( minimalGrid ) );
   EXPECT_EQ( minimal.getLocalMesh().getSizes(), typename Grid::CoordinatesType( 1 ) );
}

// Owned entities get their values, the ghost entities get -1. After the synchronization, all entities of the local
// grid must have their values. With one orientation, the ghost entities of the other orientations stay -1.
// All orientations are synchronized if Orientation is negative.
template< int EntityDimension, int Orientation, typename DistributedGridType >
void
testSynchronizer( const DistributedGridType& distributedGrid )
{
   constexpr int orientation = Orientation;
   using Grid = typename DistributedGridType::GridType;
   using Device = typename Grid::DeviceType;
   using Index = typename Grid::IndexType;
   using ArrayType = Containers::Array< double, Device, Index >;
   using ViewType = typename ArrayType::ViewType;
   SCOPED_TRACE( "Entity dimension: " + std::to_string( EntityDimension ) + ", orientation: " + std::to_string( orientation ) );

   const Grid& localGrid = distributedGrid.getLocalMesh();
   const EntityValue< Grid > value{ distributedGrid.getGlobalBegin() - distributedGrid.getLowerOverlap() };
   const Index entitiesCount = localGrid.template getEntitiesCount< EntityDimension >();

   ArrayType values( entitiesCount, -1 );
   forLocalEntities< EntityDimension >( localGrid, SetValues< Grid, ViewType >{ value, values.getView(), -1 } );

   ArrayType expected( entitiesCount, -1 );
   forLocalEntities< EntityDimension >( localGrid, SetValues< Grid, ViewType >{ value, expected.getView(), -1 } );
   forAllEntities< EntityDimension >( localGrid, SetValues< Grid, ViewType >{ value, expected.getView(), orientation } );

   DistributedGridSynchronizer< DistributedGridType > synchronizer( distributedGrid );
   const ReadValue< ViewType > read{ values.getView() };
   const WriteValue< ViewType > write{ values.getView() };
   if constexpr( Orientation < 0 )
      synchronizer.template synchronize< EntityDimension >( read, write );
   else
      synchronizer.template synchronize< EntityDimension, Orientation >( read, write );

   EXPECT_EQ( values, expected );
}

// Tests the synchronization of the entities of all dimensions, all orientations at once and each orientation alone.
template< typename DistributedGridType >
void
testSynchronizerForAllEntities( const DistributedGridType& distributedGrid )
{
   using Grid = typename DistributedGridType::GridType;
   constexpr int Dimension = Grid::getMeshDimension();
   Algorithms::staticFor< int, 0, Dimension + 1 >(
      [ & ]( auto entityDimension )
      {
         constexpr int EntityDimension = decltype( entityDimension )::value;
         constexpr int orientationsCount = Grid::EntitiesOrientations::template getOrientationsCount< EntityDimension >();
         testSynchronizer< EntityDimension, -1 >( distributedGrid );
         Algorithms::staticFor< int, 0, orientationsCount >(
            [ & ]( auto orientation )
            {
               testSynchronizer< EntityDimension, decltype( orientation )::value >( distributedGrid );
            } );
      } );
}

// Compares the sum over the distributed grid with the sum over the global grid for all entity dimensions.
template< typename DistributedGridType >
void
testSum( const DistributedGridType& distributedGrid )
{
   using Grid = typename DistributedGridType::GridType;
   constexpr int Dimension = Grid::getMeshDimension();
   const EntityValue< Grid > localValue{ distributedGrid.getGlobalBegin() - distributedGrid.getLowerOverlap() };
   const EntityValue< Grid > globalValue{ typename Grid::CoordinatesType( 0 ) };
   Algorithms::staticFor< int, 0, Dimension + 1 >(
      [ & ]( auto entityDimension )
      {
         constexpr int EntityDimension = decltype( entityDimension )::value;
         SCOPED_TRACE( "Entity dimension: " + std::to_string( EntityDimension ) );
         EXPECT_EQ(
            Meshes::reduce< EntityDimension >( distributedGrid, localValue ),
            Meshes::reduce< EntityDimension >( distributedGrid.getGlobalGrid(), globalValue ) );
      } );
}

TYPED_TEST( DistributedGridTest, synchronizer )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;

   for( int levels = 0; levels <= 2; levels++ ) {
      SCOPED_TRACE( "Ghost levels: " + std::to_string( levels ) );
      Grid globalGrid;
      globalGrid.setSizes( getGlobalSizes< Grid >() );
      DistributedGridType distributedGrid;
      distributedGrid.setGlobalGrid( globalGrid );
      try {
         distributedGrid.setGhostLevels( levels );
      }
      catch( const std::invalid_argument& ) {
         // the overlaps are not smaller than the subdomains, which depends only on the number of ranks,
         // so all ranks skip the test
         continue;
      }
      testSynchronizerForAllEntities( distributedGrid );
   }
}

// Returns all decompositions of the given number of ranks into Dimension factors.
template< int Dimension >
std::vector< Containers::StaticVector< Dimension, int > >
getDecompositions( int ranks )
{
   std::vector< Containers::StaticVector< Dimension, int > > result;
   if constexpr( Dimension == 1 )
      result.push_back( Containers::StaticVector< 1, int >( ranks ) );
   else
      for( int first = 1; first <= ranks; first++ )
         if( ranks % first == 0 )
            for( const auto& rest : getDecompositions< Dimension - 1 >( ranks / first ) ) {
               Containers::StaticVector< Dimension, int > decomposition;
               decomposition[ 0 ] = first;
               for( int i = 1; i < Dimension; i++ )
                  decomposition[ i ] = rest[ i - 1 ];
               result.push_back( decomposition );
            }
   return result;
}

// Tests all decompositions of the ranks given by setDomainDecomposition, including the decompositions along the
// z-axis, which MPI_Dims_create does not produce for small numbers of ranks. Checks that the subdomains are mapped
// to distinct ranks, the synchronization and the reduction.
TYPED_TEST( DistributedGridTest, explicitDecompositions )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;
   using CoordinatesType = typename Grid::CoordinatesType;
   constexpr int Dimension = Grid::getMeshDimension();
   const int nproc = MPI::GetSize( MPI_COMM_WORLD );
   const int rank = MPI::GetRank( MPI_COMM_WORLD );

   Grid globalGrid;
   globalGrid.setSizes( getGlobalSizes< Grid >() );
   for( const auto& decomposition : getDecompositions< Dimension >( nproc ) ) {
      SCOPED_TRACE( "Decomposition: " + std::string( convertToString( decomposition ).getString() ) );
      DistributedGridType distributedGrid;
      distributedGrid.setDomainDecomposition( CoordinatesType( decomposition ) );
      distributedGrid.setGlobalGrid( globalGrid );
      EXPECT_EQ( distributedGrid.getDomainDecomposition(), CoordinatesType( decomposition ) );

      // each rank owns a different subdomain and getRankOfProcCoord is the inverse mapping
      std::vector< int > coordinates( Dimension * nproc );
      std::vector< int > localCoordinates( Dimension );
      for( int i = 0; i < Dimension; i++ )
         localCoordinates[ i ] = distributedGrid.getSubdomainCoordinates()[ i ];
      MPI::Allgather( localCoordinates.data(), Dimension, coordinates.data(), Dimension, MPI_COMM_WORLD );
      for( int r = 0; r < nproc; r++ ) {
         CoordinatesType subdomain;
         for( int i = 0; i < Dimension; i++ )
            subdomain[ i ] = coordinates[ r * Dimension + i ];
         EXPECT_EQ( distributedGrid.getRankOfProcCoord( subdomain ), r ) << "Subdomain: " << subdomain;
      }

      // the ghost levels are rejected on all ranks if some subdomain is too small
      try {
         distributedGrid.setGhostLevels( 1 );
      }
      catch( const std::invalid_argument& ) {
         if( rank == 0 )
            std::cout << "Skipping the decomposition " << decomposition << " with too small subdomains.\n";
         continue;
      }
      EXPECT_EQ( distributedGrid.getGhostLevels(), nproc > 1 ? 1 : 0 );
      testSynchronizerForAllEntities( distributedGrid );
      testSum( distributedGrid );
   }
}

// The overlaps of the interfaces between the subdomains differ, the interface between the subdomains k and k + 1
// along the axis i has the width 1 + ( k + i ) % 2 on both sides.
TYPED_TEST( DistributedGridTest, differentOverlaps )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;
   using OverlapsType = typename DistributedGridType::SubdomainOverlapsType;
   constexpr int Dimension = Grid::getMeshDimension();

   Grid globalGrid;
   globalGrid.setSizes( getGlobalSizes< Grid >() );
   DistributedGridType distributedGrid;
   distributedGrid.setGlobalGrid( globalGrid );
   const auto& decomposition = distributedGrid.getDomainDecomposition();
   const auto& subdomain = distributedGrid.getSubdomainCoordinates();
   OverlapsType lower = 0;
   OverlapsType upper = 0;
   for( int i = 0; i < Dimension; i++ ) {
      // the decision depends only on the global data, so all ranks skip the test together
      if( decomposition[ i ] > 1 && globalGrid.getSizes()[ i ] / decomposition[ i ] <= 2 )
         GTEST_SKIP() << "The subdomains are too small for the overlaps of width 2.";
      if( subdomain[ i ] > 0 )
         lower[ i ] = 1 + ( subdomain[ i ] - 1 + i ) % 2;
      if( subdomain[ i ] < decomposition[ i ] - 1 )
         upper[ i ] = 1 + ( subdomain[ i ] + i ) % 2;
   }
   distributedGrid.setOverlaps( lower, upper );
   EXPECT_EQ( distributedGrid.getLowerOverlap(), lower );
   EXPECT_EQ( distributedGrid.getUpperOverlap(), upper );
   testSynchronizerForAllEntities( distributedGrid );
   testSum( distributedGrid );
}

// Invalid overlaps are rejected and the distributed grid is not changed.
TYPED_TEST( DistributedGridTest, overlapsValidation )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;
   using OverlapsType = typename DistributedGridType::SubdomainOverlapsType;
   using CoordinatesType = typename Grid::CoordinatesType;
   constexpr int Dimension = Grid::getMeshDimension();

   DistributedGridType distributedGrid;
   EXPECT_THROW( distributedGrid.setOverlaps( 0, 0 ), std::logic_error );
   EXPECT_THROW( distributedGrid.setGhostLevels( 1 ), std::logic_error );

   Grid globalGrid;
   globalGrid.setSizes( getGlobalSizes< Grid >() );
   distributedGrid.setGlobalGrid( globalGrid );
   EXPECT_THROW( distributedGrid.setOverlaps( -1, -1 ), std::invalid_argument );

   // overlaps equal to the minimal size of the subdomains on the sides with neighbors
   OverlapsType lower = 0;
   OverlapsType upper = 0;
   bool hasNeighbor = false;
   for( int i = 0; i < Dimension; i++ ) {
      const int minimalSize = globalGrid.getSizes()[ i ] / distributedGrid.getDomainDecomposition()[ i ];
      CoordinatesType direction = 0;
      direction[ i ] = -1;
      if( distributedGrid.isThereNeighbor( direction ) ) {
         lower[ i ] = minimalSize;
         hasNeighbor = true;
      }
      direction[ i ] = 1;
      if( distributedGrid.isThereNeighbor( direction ) ) {
         upper[ i ] = minimalSize;
         hasNeighbor = true;
      }
   }
   if( hasNeighbor )
      EXPECT_THROW( distributedGrid.setOverlaps( lower, upper ), std::invalid_argument );

   // the failed calls did not change the local grid
   EXPECT_EQ( distributedGrid.getLowerOverlap(), OverlapsType( 0 ) );
   EXPECT_EQ( distributedGrid.getUpperOverlap(), OverlapsType( 0 ) );
   EXPECT_EQ( distributedGrid.getLocalMesh().getSizes(), distributedGrid.getLocalSize() );
}

// The distributed grid works on a communicator which is a part of MPI_COMM_WORLD.
TYPED_TEST( DistributedGridTest, subcommunicator )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;
   const int nproc = MPI::GetSize( MPI_COMM_WORLD );
   const int rank = MPI::GetRank( MPI_COMM_WORLD );

   // two groups of ranks, the larger one has the upper half of the ranks
   const int color = rank < nproc / 2 ? 0 : 1;
   const MPI::Comm communicator = MPI::Comm::split( MPI_COMM_WORLD, color, rank );
   Grid globalGrid;
   globalGrid.setSizes( getGlobalSizes< Grid >() );
   DistributedGridType distributedGrid;
   distributedGrid.setCommunicator( communicator );
   distributedGrid.setGlobalGrid( globalGrid );
   EXPECT_EQ( distributedGrid.isDistributed(), communicator.size() > 1 );
   distributedGrid.setGhostLevels( 1 );
   testSynchronizerForAllEntities( distributedGrid );
   testSum( distributedGrid );
}

// The results of the ranks are combined in the order of the ranks: the reduction returning its first valid
// argument, which is associative but not commutative, gives the value of the first rank.
struct RankValue
{
   int rank;

   template< typename Entity >
   __cuda_callable__
   int
   operator()( const Entity& entity ) const
   {
      return rank;
   }
};

struct FirstValid
{
   __cuda_callable__
   int
   operator()( int a, int b ) const
   {
      return a >= 0 ? a : b;
   }
};

struct LastValid
{
   __cuda_callable__
   int
   operator()( int a, int b ) const
   {
      return b >= 0 ? b : a;
   }
};

TYPED_TEST( DistributedGridTest, nonCommutativeReduce )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;
   constexpr int Dimension = Grid::getMeshDimension();
   const int nproc = MPI::GetSize( MPI_COMM_WORLD );
   const int rank = MPI::GetRank( MPI_COMM_WORLD );

   Grid globalGrid;
   globalGrid.setSizes( getGlobalSizes< Grid >() );
   DistributedGridType distributedGrid;
   distributedGrid.setGlobalGrid( globalGrid );
   EXPECT_EQ( Meshes::reduce< Dimension >( distributedGrid, RankValue{ rank }, FirstValid{}, -1 ), 0 );
   EXPECT_EQ( Meshes::reduce< Dimension >( distributedGrid, RankValue{ rank }, LastValid{}, -1 ), nproc - 1 );
}

TYPED_TEST( DistributedGridTest, directionIndex )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;
   using CoordinatesType = typename Grid::CoordinatesType;
   constexpr int Dimension = Grid::getMeshDimension();
   for( int i = 0; i < DistributedGridType::getNeighborsCount(); i++ ) {
      const CoordinatesType direction( Directions::template getXYZ< Dimension >( i ) );
      EXPECT_EQ( DistributedGridSynchronizer< DistributedGridType >::getDirectionIndex( direction ), i )
         << "Direction: " << direction;
   }
}

// The synchronization of several values at once.
template< typename Grid, typename View >
struct SetVectorValues
{
   EntityValue< Grid > value;
   mutable View view;

   template< typename Entity >
   __cuda_callable__
   void
   operator()( const Entity& entity ) const
   {
      view[ entity.getIndex() ] = { value( entity ), -value( entity ) };
   }
};

template< typename View >
struct ReadVector
{
   View view;

   template< typename Entity >
   __cuda_callable__
   auto
   operator()( const Entity& entity ) const
   {
      return view[ entity.getIndex() ];
   }
};

template< typename View, typename Value >
struct WriteVector
{
   mutable View view;

   template< typename Entity >
   __cuda_callable__
   void
   operator()( const Entity& entity, const Value& value ) const
   {
      view[ entity.getIndex() ] = value;
   }
};

TYPED_TEST( DistributedGridTest, synchronizerWithStaticVector )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;
   using Device = typename Grid::DeviceType;
   using Index = typename Grid::IndexType;
   using Value = Containers::StaticVector< 2, double >;
   using ArrayType = Containers::Array< Value, Device, Index >;
   using ViewType = typename ArrayType::ViewType;
   constexpr int Dimension = Grid::getMeshDimension();

   Grid globalGrid;
   globalGrid.setSizes( getGlobalSizes< Grid >() );
   DistributedGridType distributedGrid;
   distributedGrid.setGlobalGrid( globalGrid );
   distributedGrid.setGhostLevels( 1 );
   const Grid& localGrid = distributedGrid.getLocalMesh();
   const EntityValue< Grid > value{ distributedGrid.getGlobalBegin() - distributedGrid.getLowerOverlap() };

   ArrayType values( localGrid.template getEntitiesCount< Dimension - 1 >(), Value( -1 ) );
   forLocalEntities< Dimension - 1 >( localGrid, SetVectorValues< Grid, ViewType >{ value, values.getView() } );
   ArrayType expected( values.getSize() );
   forAllEntities< Dimension - 1 >( localGrid, SetVectorValues< Grid, ViewType >{ value, expected.getView() } );

   DistributedGridSynchronizer< DistributedGridType > synchronizer( distributedGrid );
   synchronizer.template synchronize< Dimension - 1 >(
      ReadVector< ViewType >{ values.getView() }, WriteVector< ViewType, Value >{ values.getView() } );
   EXPECT_EQ( values, expected );
}

// The reduction over the distributed grid must be equal to the reduction over the global grid. The values are
// given by the global coordinates of the entities, they are integers, so the sums are exact in any order.
TYPED_TEST( DistributedGridTest, reduce )
{
   using Grid = typename TestFixture::GridType;
   using DistributedGridType = typename TestFixture::DistributedGridType;
   using CoordinatesType = typename Grid::CoordinatesType;
   constexpr int Dimension = Grid::getMeshDimension();

   Grid globalGrid;
   globalGrid.setSizes( getGlobalSizes< Grid >() );
   DistributedGridType distributedGrid;
   distributedGrid.setGlobalGrid( globalGrid );
   distributedGrid.setGhostLevels( 1 );
   const EntityValue< Grid > localValue{ distributedGrid.getGlobalBegin() - distributedGrid.getLowerOverlap() };
   const EntityValue< Grid > globalValue{ CoordinatesType( 0 ) };

   Algorithms::staticFor< int, 0, Dimension + 1 >(
      [ & ]( auto entityDimension )
      {
         constexpr int EntityDimension = decltype( entityDimension )::value;
         SCOPED_TRACE( "Entity dimension: " + std::to_string( EntityDimension ) );
         EXPECT_EQ(
            Meshes::reduce< EntityDimension >( distributedGrid, localValue ),
            Meshes::reduce< EntityDimension >( globalGrid, globalValue ) );
         EXPECT_EQ(
            Meshes::reduce< EntityDimension >( distributedGrid, localValue, TNL::Max{} ),
            Meshes::reduce< EntityDimension >( globalGrid, globalValue, TNL::Max{} ) );
         EXPECT_EQ(
            Meshes::reduce< EntityDimension >( distributedGrid, localValue, TNL::Min{}, 1e100 ),
            Meshes::reduce< EntityDimension >( globalGrid, globalValue, TNL::Min{}, 1e100 ) );
      } );
}

}  // namespace DistributedGridTest
