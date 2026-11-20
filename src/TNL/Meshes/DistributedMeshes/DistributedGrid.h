// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/StaticArray.h>
#include <TNL/Logger.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedMeshes/Directions.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL::Meshes::DistributedMeshes {

/**
 * \brief Grid decomposed into rectangular subdomains, one per MPI rank.
 *
 * The global grid is decomposed into `getDomainDecomposition()` subdomains
 * along each axis.
 * Each rank stores its subdomain as a local grid (see \ref getLocalMesh),
 * which consists of the cells owned by the rank
 * and the layers of ghost cells (overlaps) along the sides
 * adjoining the neighboring subdomains.
 * There are no ghost cells beyond the global boundary.
 *
 * In the local grid,
 * the owned cells form the region `[ getLocalBegin(), getLocalEnd() )`
 * (see \ref Grid::getLocalBegin and \ref Grid::getLocalEnd)
 * and the sides on the global boundary are marked by \ref Grid::setGlobalBoundarySides.
 * Hence, \ref TNL::Meshes::forLocalEntities traverses the entities owned by the rank,
 * and \ref TNL::Meshes::forInteriorEntities and \ref TNL::Meshes::forBoundaryEntities
 * traverse the owned entities in the interior and on the boundary of the global grid.
 *
 * \tparam Dimension is the dimension of the grid.
 * \tparam Real is the type of the floating point numbers of the grid.
 * \tparam Device is the device where the grid is allocated.
 * \tparam Index is the type of the indices of the grid entities.
 */
template< int Dimension, typename Real, typename Device, typename Index >
class DistributedMesh< Grid< Dimension, Real, Device, Index > >
{
public:
   //! \brief Type of the floating point numbers of the grid.
   using RealType = Real;

   //! \brief Device where the grid is allocated.
   using DeviceType = Device;

   //! \brief Type of the indices of the grid entities.
   using IndexType = Index;

   //! \brief Type of the global indices of the grid entities.
   using GlobalIndexType = Index;

   //! \brief Type of the global and the local grid.
   using GridType = Grid< Dimension, Real, Device, IndexType >;

   //! \brief Type of the points in the space.
   using PointType = typename GridType::PointType;

   //! \brief Type of the coordinates of the grid entities and the subdomains.
   using CoordinatesType = Containers::StaticVector< Dimension, IndexType >;

   //! \brief Type of the widths of the overlaps along each axis.
   using SubdomainOverlapsType = Containers::StaticVector< Dimension, IndexType >;

   //! \brief Type of the array with the ranks of the neighbors, see \ref getNeighbors.
   using NeighborsType = Containers::StaticArray< Directions::i3pow( Dimension ) - 1, int >;

   //! \brief Returns the dimension of the grid.
   [[nodiscard]] static constexpr int
   getMeshDimension()
   {
      return Dimension;
   }

   //! \brief Returns the number of all possible neighbors of a subdomain, including the diagonal ones.
   [[nodiscard]] static constexpr int
   getNeighborsCount()
   {
      return Directions::i3pow( Dimension ) - 1;
   }

   DistributedMesh() = default;

   ~DistributedMesh() = default;

   /**
    * \brief Sets the number of subdomains along each axis.
    *
    * The zero values are computed by `MPI_Dims_create` in \ref setGlobalGrid.
    * It must be called before \ref setGlobalGrid.
    *
    * \param domainDecomposition is the number of subdomains along each axis.
    */
   void
   setDomainDecomposition( const CoordinatesType& domainDecomposition );

   //! \brief Returns the number of subdomains along each axis.
   [[nodiscard]] const CoordinatesType&
   getDomainDecomposition() const;

   /**
    * \brief Sets the global grid and decomposes it into the subdomains.
    *
    * The local grid is set up without overlaps,
    * call \ref setGhostLevels or \ref setOverlaps to add the ghost cells.
    *
    * \param globalGrid is the global grid.
    */
   void
   setGlobalGrid( const GridType& globalGrid );

   //! \brief Returns the global grid.
   [[nodiscard]] const GridType&
   getGlobalGrid() const;

   /**
    * \brief Sets the widths of the overlaps of the local subdomain.
    *
    * The overlaps must be zero on the sides on the global boundary
    * and smaller than the number of cells of all subdomains along the axis,
    * so that all ghost entities are owned by the neighboring subdomains.
    * Otherwise, `std::invalid_argument` is thrown.
    * The overlaps of the neighboring subdomains must agree,
    * i.e. the upper overlap of a subdomain must be equal to the lower overlap of its upper neighbor,
    * as with \ref setGhostLevels.
    *
    * \param lower are the widths of the overlaps at the lower sides along each axis.
    * \param upper are the widths of the overlaps at the upper sides along each axis.
    */
   void
   setOverlaps( const SubdomainOverlapsType& lower, const SubdomainOverlapsType& upper );

   /**
    * \brief Sets the overlaps with the given width on all sides adjoining a neighboring subdomain.
    *
    * \param levels is the number of the layers of the ghost cells.
    */
   void
   setGhostLevels( int levels );

   //! \brief Returns the maximal width of the overlaps.
   [[nodiscard]] int
   getGhostLevels() const;

   //! \brief Returns `true` if the grid is decomposed into more than one subdomain.
   [[nodiscard]] bool
   isDistributed() const;

   //! \brief Returns `true` if the local subdomain has some side on the global boundary.
   [[nodiscard]] bool
   isBoundarySubdomain() const;

   //! \brief Returns the widths of the overlaps at the lower sides of the local subdomain.
   [[nodiscard]] const SubdomainOverlapsType&
   getLowerOverlap() const;

   //! \brief Returns the widths of the overlaps at the upper sides of the local subdomain.
   [[nodiscard]] const SubdomainOverlapsType&
   getUpperOverlap() const;

   //! \brief Returns the local grid including the ghost cells.
   [[nodiscard]] const GridType&
   getLocalMesh() const;

   //! \brief Returns the number of cells of the local subdomain without the ghost cells.
   [[nodiscard]] const CoordinatesType&
   getLocalSize() const;

   //! \brief Returns the number of cells of the global grid along each axis.
   [[nodiscard]] const CoordinatesType&
   getGlobalSize() const;

   //! \brief Returns the coordinates of the first cell of the local subdomain (without ghost cells) in the global grid.
   [[nodiscard]] const CoordinatesType&
   getGlobalBegin() const;

   //! \brief Returns the coordinates of the local subdomain in the decomposition.
   [[nodiscard]] const CoordinatesType&
   getSubdomainCoordinates() const;

   //! \brief Sets the MPI communicator. It must be called before \ref setGlobalGrid.
   void
   setCommunicator( const MPI::Comm& communicator );

   //! \brief Returns the MPI communicator.
   [[nodiscard]] const MPI::Comm&
   getCommunicator() const;

   //! \brief Returns the number of entities with the given dimension in the global grid.
   template< int EntityDimension >
   [[nodiscard]] IndexType
   getEntitiesCount() const;

   //! \brief Returns the number of entities of the given type in the global grid.
   template< typename Entity >
   [[nodiscard]] IndexType
   getEntitiesCount() const;

   /**
    * \brief Returns the ranks of the neighboring subdomains.
    *
    * The neighbor in the direction `direction`, where `direction[ i ]` is -1, 0 or 1,
    * has the index `Directions::getDirection( direction )`
    * (see \ref Directions::getXYZ for the inverse mapping).
    * The value is -1 if there is no neighbor in the direction.
    */
   [[nodiscard]] const NeighborsType&
   getNeighbors() const;

   //! \brief Returns the rank of the subdomain with the given coordinates in the decomposition.
   [[nodiscard]] int
   getRankOfProcCoord( const CoordinatesType& nodeCoordinates ) const;

   //! \brief Returns the coordinates of the local subdomain as a string, e.g. `1-0-2`.
   [[nodiscard]] String
   printProcessCoords() const;

   //! \brief Returns the decomposition as a string, e.g. `2-2-1`.
   [[nodiscard]] String
   printProcessDistr() const;

   //! \brief Writes the decomposition to the logger.
   void
   writeProlog( Logger& logger );

   //! \brief Comparison operator.
   [[nodiscard]] bool
   operator==( const DistributedMesh& other ) const;

   //! \brief Comparison operator.
   [[nodiscard]] bool
   operator!=( const DistributedMesh& other ) const;

   //! \brief Returns `true` if there is a neighboring subdomain in the given direction.
   [[nodiscard]] bool
   isThereNeighbor( const CoordinatesType& direction ) const;

protected:
   void
   setupNeighbors();

   GridType globalGrid, localGrid;
   CoordinatesType localSize = 0;
   CoordinatesType globalBegin = 0;

   SubdomainOverlapsType lowerOverlap = 0;
   SubdomainOverlapsType upperOverlap = 0;

   CoordinatesType domainDecomposition = 0;
   CoordinatesType subdomainCoordinates = 0;

   NeighborsType neighbors = -1;

   bool distributed = false;

   bool isSet = false;

   MPI::Comm communicator = MPI_COMM_WORLD;
};

/**
 * \brief Writes the distributed grid to the output stream.
 *
 * The ranks of the communicator write in turns, so the function must be called by all ranks.
 */
template< int Dimension, typename Real, typename Device, typename Index >
std::ostream&
operator<<( std::ostream& str, const DistributedMesh< Grid< Dimension, Real, Device, Index > >& grid );

}  // namespace TNL::Meshes::DistributedMeshes

#include <TNL/Meshes/DistributedMeshes/DistributedGrid.hpp>
