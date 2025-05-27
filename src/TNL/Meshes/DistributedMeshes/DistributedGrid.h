// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Logger.h>
#include <TNL/Meshes/DistributedMeshes/Directions.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL::Meshes::DistributedMeshes {

template< int Dimension, typename Real, typename Device, typename Index >
class DistributedMesh< Grid< Dimension, Real, Device, Index > >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using GlobalIndexType = Index;
   using GridType = Grid< Dimension, Real, Device, IndexType >;
   using PointType = typename GridType::PointType;
   using CoordinatesType = Containers::StaticVector< Dimension, IndexType >;
   using SubdomainOverlapsType = Containers::StaticVector< Dimension, IndexType >;

   [[nodiscard]] static constexpr int
   getMeshDimension()
   {
      return Dimension;
   }

   [[nodiscard]] static constexpr int
   getNeighborsCount()
   {
      return Directions::i3pow( Dimension ) - 1;
   }

   DistributedMesh() = default;

   ~DistributedMesh() = default;

   void
   setDomainDecomposition( const CoordinatesType& domainDecomposition );

   [[nodiscard]] const CoordinatesType&
   getDomainDecomposition() const;

   void
   setGlobalGrid( const GridType& globalGrid );

   [[nodiscard]] const GridType&
   getGlobalGrid() const;

   void
   setOverlaps( const SubdomainOverlapsType& lower, const SubdomainOverlapsType& upper );

   // for compatibility with DistributedMesh
   void
   setGhostLevels( int levels );
   [[nodiscard]] int
   getGhostLevels() const;

   [[nodiscard]] bool
   isDistributed() const;

   [[nodiscard]] bool
   isBoundarySubdomain() const;

   // currently used overlaps at this subdomain
   [[nodiscard]] const SubdomainOverlapsType&
   getLowerOverlap() const;

   [[nodiscard]] const SubdomainOverlapsType&
   getUpperOverlap() const;

   // returns the local grid WITH overlap
   [[nodiscard]] const GridType&
   getLocalMesh() const;

   // number of elements of local sub domain WITHOUT overlap
   //  TODO: getSubdomainDimensions
   [[nodiscard]] const CoordinatesType&
   getLocalSize() const;

   // TODO: delete
   // Dimensions of global grid
   [[nodiscard]] const CoordinatesType&
   getGlobalSize() const;

   // coordinates of begin of local subdomain without overlaps in global grid
   [[nodiscard]] const CoordinatesType&
   getGlobalBegin() const;

   [[nodiscard]] const CoordinatesType&
   getSubdomainCoordinates() const;

   void
   setCommunicator( const MPI::Comm& communicator );

   [[nodiscard]] const MPI::Comm&
   getCommunicator() const;

   template< int EntityDimension >
   [[nodiscard]] IndexType
   getEntitiesCount() const;

   template< typename Entity >
   [[nodiscard]] IndexType
   getEntitiesCount() const;

   [[nodiscard]] const int*
   getNeighbors() const;

   [[nodiscard]] const int*
   getPeriodicNeighbors() const;

   template< typename DistributedGridType >
   [[nodiscard]] bool
   SetupByCut( DistributedGridType& inputDistributedGrid,
               Containers::StaticVector< Dimension, int > savedDimensions,
               Containers::StaticVector< DistributedGridType::getMeshDimension() - Dimension, int > reducedDimensions,
               Containers::StaticVector< DistributedGridType::getMeshDimension() - Dimension, IndexType > fixedIndexes );

   [[nodiscard]] int
   getRankOfProcCoord( const CoordinatesType& nodeCoordinates ) const;

   [[nodiscard]] String
   printProcessCoords() const;

   [[nodiscard]] String
   printProcessDistr() const;

   void
   writeProlog( Logger& logger );

   [[nodiscard]] bool
   operator==( const DistributedMesh& other ) const;

   [[nodiscard]] bool
   operator!=( const DistributedMesh& other ) const;

   [[nodiscard]] bool
   isThereNeighbor( const CoordinatesType& direction ) const;

   void
   setupNeighbors();

   GridType globalGrid, localGrid;
   CoordinatesType localSize = 0;
   CoordinatesType globalBegin = 0;

   SubdomainOverlapsType lowerOverlap = 0;
   SubdomainOverlapsType upperOverlap = 0;

   CoordinatesType domainDecomposition = 0;
   CoordinatesType subdomainCoordinates = 0;

   // TODO: static arrays
   int neighbors[ getNeighborsCount() ];
   int periodicNeighbors[ getNeighborsCount() ];

   bool distributed = false;

   bool isSet = false;

   MPI::Comm communicator = MPI_COMM_WORLD;
};

template< int Dimension, typename Real, typename Device, typename Index >
std::ostream&
operator<<( std::ostream& str, const DistributedMesh< Grid< Dimension, Real, Device, Index > >& grid );

}  // namespace TNL::Meshes::DistributedMeshes

#include <TNL/Meshes/DistributedMeshes/DistributedGrid.hpp>
