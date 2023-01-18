// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Logger.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/GridEntitiesOrientations.h>
#include <TNL/Meshes/GridDetails/Templates/BooleanOperations.h>
#include <TNL/Meshes/GridDetails/Templates/Functions.h>
#include <TNL/Meshes/GridDetails/GridTraits.h>
#include <TNL/Meshes/GridDetails/GridEntityBase.h>

namespace TNL {
namespace Meshes {

template< class, int >
class GridEntity;

/**
 * \brief Orthogonal n-dimensional grid.
 *
 * This data structure represents regular orthogonal numerical mesh. It provides indexing of mesh
 * entities like vertexes, edges, faces or cells together with parallel traversing of all, interior
 * or boundary mesh entities.
 *
 * \tparam Dimension is grid dimension.
 * \tparam Real is type of the floating point numbers.
 * \tparam Device is the device to be used for the execution of grid operations.
 * \tparam Index is type for indexing of the mesh entities of the grid.
 */
template< int Dimension, typename Real = double, typename Device = Devices::Host, typename Index = int >
class Grid
{
public:
   /**
    * \brief Type of the floating point numbers.
    */
   using RealType = Real;

   /**
    * \brief Device to be used for the execution of grid operations.
    */
   using DeviceType = Device;

   /**
    * \brief Type for indexing of the mesh entities of the grid.
    */
   using IndexType = Index;

   /**
    * \brief Type for indexing of the mesh entities of the grid.
    *
    * This is for compatiblity with unstructured meshes.
    */
   using GlobalIndexType = Index;

   using GridTraitsType = GridTraits< Dimension, Real, Index >;

   /**
    * \brief Type for grid entities coordinates.
    */
   using CoordinatesType = typename GridTraitsType::CoordinatesType;

   /**
    * \brief Type for world coordinates.
    */
   using PointType = typename GridTraitsType::PointType;

   using NormalsType = typename GridTraitsType::NormalsType;

   using EntitiesCounts = Containers::StaticVector< Dimension + 1, Index >;

   using EntitiesOrientations = GridEntitiesOrientations< Dimension >;

   /**
    * \brief Alias for grid entities with given dimension.
    *
    * \tparam EntityDimension is dimensions of the grid entity.
    */
   template< int EntityDimension >
   using EntityType = GridEntity< Grid, EntityDimension >;

   static constexpr IndexType getTotalOrientationsCount() { return EntitiesOrientations::getTotalOrientationsCount(); }

   using OrientationNormalsContainer = Containers::StaticVector< getTotalOrientationsCount(), CoordinatesType >;

   using CoordinatesMultiplicatorsContainer = Containers::StaticVector< getTotalOrientationsCount(),  CoordinatesType >;

   template< int EntityDimension, int SuperentityDimension  >
   using SuperentitiesContainer = Containers::StaticVector< 1 << ( SuperentityDimension - Dimension - 1), IndexType >;

   /**
    * \brief Type of grid entity expressing vertexes, i.e. grid entity with dimension equal to zero.
    */
   using Vertex = EntityType< 0 >;

   /**
    * \brief Type of grid entity expressing edges, i.e. grid entity with dimension equal to one.
    *
    */
   using Edge = EntityType< 1 >;

   /**
    * \brief Type of grid entity expressing faces, i.e. grid entity with dimension equal to
    *        the grid dimension minus one.
    */
   using Face = EntityType< Dimension - 1 >;

   /**
    * \brief Type of grid entity expressing cells, i.e. grid entity with dimension equal to the grid dimension.
    */
   using Cell = EntityType< Dimension >;

   /**
    * \brief Returns the dimension of grid
    */
   static constexpr int
   getMeshDimension();

   /**
    * \brief Returns the coefficient powers size.
    */
   // TODO: Move this to FDM = Finite Difference Method implementation
   static constexpr int spaceStepsPowersSize = 5;

   using SpaceProductsContainer =
      Containers::StaticVector< std::integral_constant< Index, discretePow( spaceStepsPowersSize, Dimension ) >::value,
                                Real >;

   /**
    * \brief Grid constructor with no parameters.
    */
   Grid() = default;

   /**
    * \brief Grid constructor with grid dimensions parameters.
    *
    * \tparam Dimensions is variadic template pack.
    * \param dimensions are dimensions along particular axes of the grid. The number of parameters must
    *    be equal to the size od the grid.
    */
   template< typename... Dimensions,
             std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Dimensions >... >, bool > = true,
             std::enable_if_t< sizeof...( Dimensions ) == Dimension, bool > = true >
   Grid( Dimensions... dimensions );

   /**
    * \brief Grid constructor with grid dimensions given as \ref TNL::Meshes::Grid::CoordinatesType.
    *
    * \param dimensions are dimensions along particular axes of the grid.
    */
   Grid( const CoordinatesType& dimensions );

   __cuda_callable__
   const EntitiesOrientations& getEntitiesOrientations() const;

   /**
    * \brief Returns the number of orientations for entity dimension.
    *        For example in 2-D Grid the edge can be vertical or horizontal.
    *
    * \param[in] entityDimension is dimension of grid entities to be counted.
    */
   static constexpr Index
   getEntityOrientationsCount( IndexType entityDimension );

   /**
    * \brief Set the dimensions (or resolution) of the grid.
    *    The resolution must be given in terms on grid cells not grid vertices. The
    *    mthod accepts as many indexes for the dimensions as the dimension of the grid.
    *
    * \tparam Dimensions variadic template accepting a serie of indexes.
    * \param[in] dimensions serie of indexes defining resolution of the grid.
    */
   template< typename... Dimensions,
             std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Dimensions >... >, bool > = true,
             std::enable_if_t< sizeof...( Dimensions ) == Dimension, bool > = true >
   void
   setDimensions( Dimensions... dimensions );

   /**
    * \brief Set the dimensions (or resolution) of the grid.
    *    This method accepts particular dimensions packed in a static vector.
    *
    * \param dimensions grid dimensions given in a form of coordinate vector.
    */
   void
   setDimensions( const CoordinatesType& dimensions );

   /**
    * \brief Returns dimensions as a number of edges along each axis in a form of coordinate vector.
    *
    *\return Coordinate vector with number of edges along each axis.
    */
   __cuda_callable__
   const CoordinatesType&
   getDimensions() const noexcept;

   /**
    * \brief Returns number of entities of specific dimension.
    *
    * \param[in] dimension is a dimension of grid entities to be counted.
    * \return number of entities of specific dimension.
    */
   __cuda_callable__
   Index
   getEntitiesCount( IndexType dimension ) const;

    /**
    * \brief Returns number of entities of specific dimension given as a template parameter.
    *
    * \tparam EntityDimension is dimension of grid entities to be counted.
    *
    * \return Number of grid entities with given dimension.
    */
   template< int EntityDimension >
   __cuda_callable__
   Index
   getEntitiesCount() const noexcept; // TODO: remove this if it is not necessary for compatibility with Mesh

   /**
    * \brief Returns number of entities of specific entity type as a template parameter.
    *
    * \tparam Entity is type of grid entities to be counted.
    *
    * \return Number of grid entities with given dimension.
    */
   template< typename Entity >
   __cuda_callable__
   Index
   getEntitiesCount() const; // TODO: remove this if it is not necessary for compatibility with Mesh

   /**
    * \brief Returns count of entities for all dimensions.
    *
    * \return vector of count of entities for all dimensions.
    */
   __cuda_callable__
   const EntitiesCounts&
   getEntitiesCounts() const noexcept;

   /**
    * \brief Returns number of entities of specific dimension and orientation.
    *
    * \param[in] dimension is dimension of grid entities.
    * \param[in] orientation is orientation of the entities.
    * \return number of entities of specific dimension and orientation.
    */
   __cuda_callable__
   Index
   getOrientedEntitiesCount( IndexType dimension, IndexType orientation ) const;

   /**
    * \brief Returns number of entities of specific dimension and orientation given as template parameters.
    *
    * \tparam EntityDimension is dimension of the grid entities.
    * \tparam EntityOrientation is orientation of the grid entitie.
    * \return number of entities of specific dimension and orientation.
    */
   template< int EntityDimension,
             int EntityOrientation,
             std::enable_if_t< Templates::isInClosedInterval( 0, EntityDimension, Dimension ), bool > = true,
             std::enable_if_t< Templates::isInClosedInterval( 0, EntityOrientation, Dimension ), bool > = true >
   __cuda_callable__
   Index
   getOrientedEntitiesCount() const noexcept;

   /**
    * \brief Returns normals of the entity with the specific orientation.
    *
    * Normals is integer vector having ones for axis along which the entity has zero length.
    * For example in 3D grid we have the following possibilities:
    *
    * | Entity                     | Normals      |
    * |---------------------------:|-------------:|
    * | Vertexes                   | ( 1, 1, 1 )  |
    * | Edges along z-axis         | ( 1, 1, 0 )  |
    * | Edges along y-axis         | ( 1, 0, 1 )  |
    * | Edges along x-axis         | ( 0, 1, 1 )  |
    * | Faces along y- and z- axes | ( 1, 0, 0 )  |
    * | Faces along x- and z- axes | ( 0, 1, 0 )  |
    * | Faces along x- and y- axes | ( 0, 0, 1 )  |
    * | Cells                      | ( 0, 0, 0 )  |
    *
    * \tparam EntityDimension is dimensions of grid entity.
    * \param[in] orientation is orientation of the entity
    * \return normals of the grid entity.
    */
   template< int EntityDimension >
   __cuda_callable__
   NormalsType
   getNormals( Index orientation ) const noexcept;

   __cuda_callable__
   NormalsType
   getNormals( Index totalOrientation ) const noexcept;

   /**
    * \brief Returns basis of the entity with the specific orientation.
    *
    * Basis is integer vector having ones for axis along which the entity has non-zero lengths.
    * For example in 3D grid we have the following possibilities:
    *
    * | Entity                     | Basis        |
    * |---------------------------:|-------------:|
    * | Vertexes                   | ( 0, 0, 0 )  |
    * | Edges along z-axis         | ( 0, 0, 1 )  |
    * | Edges along y-axis         | ( 0, 1, 0 )  |
    * | Edges along x-axis         | ( 1, 0, 0 )  |
    * | Faces along y- and z- axes | ( 0, 1, 1 )  |
    * | Faces along x- and z- axes | ( 1, 0, 1 )  |
    * | Faces along x- and y- axes | ( 1, 1, 0 )  |
    * | Cells                      | ( 1, 1, 1 )  |
    *
    * \tparam EntityDimension is dimensions of grid entity.
    * \param[in] orientation is orientation of the entity
    * \return normals of the grid entity.
    */
   template< int EntityDimension >
   __cuda_callable__
   CoordinatesType
   getBasis( Index orientation ) const noexcept;

   /**
    * \brief Computes orientation index of a grid entity based on normals.
    *
    * \tparam EntityDimension is dimension of the grid entity.
    * \param normals defines the orientation of an entity.
    * \return index of grid entity orientation.
    */
   __cuda_callable__
   IndexType
   getOrientationIndex( const NormalsType& normals ) const noexcept;

   /**
    * \brief Computes coordinates of a grid entity based on an index of the entity.
    *
    * Remark: Computation of grid coordinates and its orientation based on the grid entity index
    *  is **highly inefficient** and it should not be used at critical parts of algorithms.
    *
    * \tparam EntityDimension is dimension of an entity.
    * \param entityIdx is an index of the entity.
    * \param orientation is an index of the grid entity orientation.
    * \return coordinates of the grid entity.
    */
   template< int EntityDimension >
   __cuda_callable__
   CoordinatesType
   getEntityCoordinates( IndexType entityIdx, IndexType& totalOrientationIndex ) const noexcept;

   /**
    * \brief Sets the origin and proportions of this grid.
    *
    * \param origin is the origin of the grid.
    * \param proportions is total length of this grid along particular axis.
    */
   void
   setDomain( const PointType& origin, const PointType& proportions );

   /**
    * \brief Set the origin of the grid in a form of a point.
    *
    * \param[in] origin of the grid.
    */
   void
   setOrigin( const PointType& origin ) noexcept;

   /**
    * \brief Set the origin of the grid in a form of a pack of real numbers.
    *
    * \tparam Coordinates is a pack of templates types.
    * \param[in] coordinates is a pack of real numbers defining the origin.
    */
   template< typename... Coordinates,
             std::enable_if_t< Templates::conjunction_v< std::is_convertible< Real, Coordinates >... >, bool > = true,
             std::enable_if_t< sizeof...( Coordinates ) == Dimension, bool > = true >
   void
   setOrigin( Coordinates... coordinates ) noexcept;

   /**
    * \brief Returns the origin of the grid.
    *
    * \return the origin of the grid.
    */
   __cuda_callable__
   const PointType&
   getOrigin() const noexcept;

   /**
    * \brief Set the space steps along each dimension of the grid.
    *
    * Calling of this method may change the grid proportions.
    *
    * \param[in] spaceSteps are the space steps along each dimension of the grid.
    */
   void
   setSpaceSteps( const PointType& spaceSteps ) noexcept;

   /**
    * \brief Set the space steps along each dimension of the grid in a form of a pack of real numbers.
    *
    * \tparam Steps is a pack of template types.
    * \param[in] spaceSteps is a pack of real numbers defining the space steps of the grid.
    */
   template< typename... Steps,
             std::enable_if_t< Templates::conjunction_v< std::is_convertible< Real, Steps >... >, bool > = true,
             std::enable_if_t< sizeof...( Steps ) == Dimension, bool > = true >
   void
   setSpaceSteps( Steps... spaceSteps ) noexcept;

   /**
    * \brief Returns the space steps of the grid.
    *
    * \return the space steps of the grid.
    */
   __cuda_callable__
   const PointType&
   getSpaceSteps() const noexcept;

   /**
    * \brief Returns product of given space steps powers.
    *
    * For example in 3D grid if powers are \f$ 1, 2, 3 \f$ the methods returns \f$ h_x^1 \cdot h_y^2 \cdot h_z^3\f$.
    *
    * \tparam Powers is a pack of template types.
    * \param[in] powers is a pack of numbers telling power of particular space steps.
    * \return product of given space steps powers.
    */
   template< typename... Powers,
             std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Powers >... >, bool > = true,
             std::enable_if_t< sizeof...( Powers ) == Dimension, bool > = true >
   __cuda_callable__
   Real
   getSpaceStepsProducts( Powers... powers ) const;

   /**
    * \brief Returns product of space steps powers.
    *
    * For example in 3D grid if powers are \f$ 1, 2, 3 \f$ the methods returns \f$ h_x^1 \cdot h_y^2 \cdot h_z^3\f$.
    *
    * \param[in] powers is vector of numbers telling power of particular space steps.
    * \return product of given space steps powers.
    */
   __cuda_callable__
   Real
   getSpaceStepsProducts( const CoordinatesType& powers ) const;

   __cuda_callable__
   Real
   getCellMeasure() const
   {
      return this->getSpaceStepsProducts( CoordinatesType( 1 ) );
   }

   /**
    * \brief Returns product of space step powers given as template parameters.
    *
    * The powers can be only integers.
    *
    * For example in 3D grid if powers are \f$ 1, 2, 3 \f$ the methods returns \f$ h_x^1 \cdot h_y^2 \cdot h_z^3\f$.
    *
    * \tparam Powers is a pack of indexes.
    * \return product of given space steps powers.
    */
   template< Index... Powers, std::enable_if_t< sizeof...( Powers ) == Dimension, bool > = true >
   __cuda_callable__
   Real
   getSpaceStepsProducts() const noexcept;

   /**
    * \brief Get the smallest space step.
    *
    * \return the smallest space step.
    */
   __cuda_callable__
   Real
   getSmallestSpaceStep() const noexcept;

   /**
    * \brief Get the proportions of the grid.
    *
    * \return the proportions of the grid.
    */
   __cuda_callable__
   const PointType&
   getProportions() const noexcept;

   /**
    * \brief Grid entity getter based on entity type and entity index.
    *
    * Remark: Computation of grid coordinates and its orientation based on the grid entity index
    *  is **highly inefficient** and it should not be used at critical parts of algorithms.
    *
    * \tparam EntityType is type of the grid entity.
    * \param entityIdx is index of the grid entity.
    * \return grid entity of given type.
    */
   template< typename EntityType >
   __cuda_callable__
   EntityType
   getEntity( IndexType entityIdx ) const;

   /**
    * \brief Grid entity getter based on entity type and entity coordinates.
    *
    * Grid entity orientation is set to the default value. This is especially no problem in
    * case of cells and vertexes.
    *
    * \tparam EntityType is type of the grid entity.
    * \param coordinates are coordinates of the grid entity.
    * \return grid entity of given type.
    */
   template< typename EntityType >
   __cuda_callable__
   EntityType
   getEntity( const CoordinatesType& coordinates ) const;

   /**
    * \brief Grid entity getter based on entity dimension and entity index.
    *
    * Remark: Computation of grid coordinates and its orientation based on the grid entity index
    *  is **highly inefficient** and it should not be used at critical parts of algorithms.
    *
    * \tparam EntityDimension is dimension of the grid entity.
    * \param entityIdx is index of the grid entity.
    * \return grid entity of given type.
    */
   template< int EntityDimension >
   __cuda_callable__
   EntityType< EntityDimension >
   getEntity( IndexType entityIdx ) const;

   /**
    * \brief Grid entity getter based on entity dimension and entity coordinates.
    *
    * Grid entity orientation is set to the default value. This is especially no problem in
    * case of cells and vertexes.
    *
    * \tparam EntityDimension is dimension of the grid entity.
    * \param coordinates are coordinates of the grid entity.
    * \return grid entity of given type.
    */
   template< int EntityDimension >
   __cuda_callable__
   EntityType< EntityDimension >
   getEntity( const CoordinatesType& coordinates ) const;

   /**
    * \brief Gets entity index using entity type.
    *
    * \tparam Entity is a type of the entity.
    * \param entity is instance of the entity.
    * \return index of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   Index
   getEntityIndex( const Entity& entity ) const;

   template< typename Entity >
   __cuda_callable__
   Index
   getNeighbourEntityIndex( const Entity& entity, const CoordinatesType& offset ) const;

   template< int NeighbourEntityDimension, typename Entity >
   __cuda_callable__
   Index
   getNeighbourEntityIndex( const Entity& entity, const CoordinatesType& offset,
                            Index neighbourEntityOrientation ) const;

   template< typename Entity >
   __cuda_callable__
   Entity
   getNeighbourEntity( const Entity& entity, const CoordinatesType& offset ) const;

   template< int NeighbourEntityDimension, typename Entity >
   __cuda_callable__
   EntityType< NeighbourEntityDimension >
   getNeighbourEntity( const Entity& entity, const CoordinatesType& offset,
                       const NormalsType& neighbourEntityOrientation ) const;

   template< int Direction, int Step, typename Entity >
   __cuda_callable__
   IndexType
   getAdjacentEntityIndex( const Entity& entity ) const;

   template< typename Entity >
   __cuda_callable__
   void
   getAdjacentCells( const Entity& entity, IndexType& closer, IndexType& remoter ) const;

   /**
    * \brief Gives indexes of all superentities of given entity.
    *
    * Super entity is entity with higher dimension containing the original entity. In terms
    * of basis vectors the superentity must have ones everywhere where the entity has ones.
    * In addition, the superentity must have more ones. For example
    *
    * entity with basis (0,1) is vertical face in 2D grid
    * superentity with basis (1,1) is cell in 2D grid
    *
    * \tparam Entity
    * \param entity
    * \param closer
    * \param remoter
    */
   template< int SuperentityDimension, typename Entity >
   __cuda_callable__
   void getSuperentitiesIndexes( const Entity& entity,
      SuperentitiesContainer< SuperentityDimension, Entity::getDimension() >& closer,
      SuperentitiesContainer< SuperentityDimension, Entity::getDimension() >& remoter ) const;

   template< typename Entity >
   __cuda_callable__
   void
   getAdjacentFacesIndexes( const Entity& entity, CoordinatesType& closer, CoordinatesType& remoter ) const;


   //__cuda_callable__
   //void getFacesIndexes( const Entity& entity, ??? ) const;

   template< typename Entity >
   __cuda_callable__
   PointType getEntityOrigin( const Entity& entity ) const;

   template< typename Entity >
   __cuda_callable__
   PointType getEntityCenter( const Entity& entity ) const;

   template< typename Entity >
   __cuda_callable__
   RealType getEntityMeasure( const Entity& entity ) const;

   template< typename Entity >
   __cuda_callable__
   bool isBoundaryEntity( const Entity& entity ) const;

   /**
    * \brief Sets the subdomain of distributed grid.
    *
    * \param begin is "lower left" corner of the subdomain.
    * \param end is "upper right" corner of the subdomain.
    */
   void
   setLocalSubdomain( const CoordinatesType& begin, const CoordinatesType& end );

   /**
    * \brief Sets the "lower left" corfner of subdomain of distributed grid.
    *
    * \param begin is "lower left" corner of the subdomain.
    */
   void
   setLocalBegin( const CoordinatesType& begin );

   /**
    * \brief Sets the "upper right" corfner of subdomain of distributed grid.
    *
    * \param end is "upper right" corner of the subdomain.
    */
   void
   setLocalEnd( const CoordinatesType& end );

   /**
    * \brief Gets the "lower left" corner of subdomain for distributed grid.
    *
    * \return const CoordinatesType& is "lower left" corner of subdomain for distributed grid.
    */
   const CoordinatesType&
   getLocalBegin() const;

   /**
    * \brief Gets the "upper right" corner of subdomain for distributed grid.
    *
    * \return const CoordinatesType& is "upper right" corner of subdomain for distributed grid.
    */
   const CoordinatesType&
   getLocalEnd() const;

   /**
    * \brief Sets begin of the region of interior cells, i.e. all cells without the boundary cells.
    *
    * \param begin is begin of the region of interior cells.
    */
   void
   setInteriorBegin( const CoordinatesType& begin );

   /**
    * \brief Sets end of the region of interior cells, i.e. all cells without the boundary cells.
    *
    * \param end is end of the region of interior cells.
    */
   void
   setInteriorEnd( const CoordinatesType& end );

   /**
    * \brief Gets begin of the region of interior cells, i.e. all cells without the boundary cells.
    *
    * \return begin of the region of interior cells.
    */
   const CoordinatesType&
   getInteriorBegin() const;

   /**
    * \brief Gets end of the region of interior cells, i.e. all cells without the boundary cells.
    *
    * \return end of the region of interior cells.
    */
   const CoordinatesType&
   getInteriorEnd() const;

   /**
    * \brief Writes info about the grid.
    *
    * \param[in] logger is a logger used to write the grid.
    */
   void
   writeProlog( TNL::Logger& logger ) const noexcept;

   /**
    * \brief Iterate over all mesh entities with given dimension and perform given lambda function
    *    on each of them.
    *
    * Entities processed by this method are such that their coordinates \f$c\f$ fullfil \f$ origin \leq c < origin +
    * proportions\f$.
    *
    * \tparam EntityDimension is dimension of the grid entities.
    * \param func is an instance of the lambda function to be performed on each grid entity.
    *     It is supposed to have the following form:
    *
    * ```
    * auto func = [=] __cuda_callable__( const typename Grid< Dimension, Real, Device, Index >::template EntityType<
    * EntityDimension >&entity ) mutable {};
    * ```
    * where \e entity represents given grid entity. See \ref TNL::Meshes::GridEntity.
    *
    * \param args are packed arguments that are going to be passed to the lambda function.
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   forAllEntities( Func func, FuncArgs... args ) const;

   /**
    * \brief Iterate over all mesh entities within given region with given dimension and perform given lambda function
    *    on each of them.
    *
    * Entities processed by this method are such that their coordinates \f$c\f$ fullfil \f$ begin \leq c < end\f$.
    *
    * \tparam EntityDimension is dimension of the grid entities.
    * \param begin is the 'lower left' corner of the region.
    * \param end is the 'upper right' corner of the region.
    * \param func is an instance of the lambda function to be performed on each grid entity.
    *     It is supposed to have the following form:
    *
    * ```
    * auto func = [=] __cuda_callable__( const typename Grid< Dimension, Real, Device, Index >::template EntityType<
    * EntityDimension >&entity ) mutable {};
    * ```
    * where \e entity represents given grid entity. See \ref TNL::Meshes::GridEntity.
    * \param args are packed arguments that are going to be passed to the lambda function.
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   forEntities( const CoordinatesType& begin, const CoordinatesType& end, Func func, FuncArgs... args ) const;

   /**
    * \brief Iterate over all interior mesh entities with given dimension and perform given lambda function
    *    on each of them.
    *
    * Entities processed by this method are such that their coordinates \f$c\f$ fullfil \f$ origin < c < origin + proportions -
    * 1\f$.
    *
    * \tparam EntityDimension is dimension of the grid entities.
    * \param func is an instance of the lambda function to be performed on each grid entity.
    *     It is supposed to have the following form:
    *
    * ```
    * auto func = [=] __cuda_callable__( const typename Grid< Dimension, Real, Device, Index >::template EntityType<
    * EntityDimension >&entity ) mutable {};
    * ```
    * where \e entity represents given grid entity. See \ref TNL::Meshes::GridEntity.
    * \param args are packed arguments that are going to be passed to the lambda function.
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   forInteriorEntities( Func func, FuncArgs... args ) const;

   /**
    * \brief Iterate over all boundary mesh entities with given dimension and perform given lambda function
    *    on each of them.
    *
    * \tparam EntityDimension is dimension of the grid entities.
    * \param func is an instance of the lambda function to be performed on each grid entity.
    *     It is supposed to have the following form:
    *
    * ```
    * auto func = [=] __cuda_callable__( const typename Grid< Dimension, Real, Device, Index >::template EntityType<
    * EntityDimension >&entity ) mutable {};
    * ```
    * where \e entity represents given grid entity. See \ref TNL::Meshes::GridEntity.
    * \param args are packed arguments that are going to be passed to the lambda function.
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   forBoundaryEntities( Func func, FuncArgs... args ) const;

   /**
    * \brief Iterate over all boundary mesh entities of given region with given dimension and perform given lambda function
    *    on each of them.
    *
    * \tparam EntityDimension is dimension of the grid entities.
    * \param begin is the 'lower left' corner of the region.
    * \param end is the 'upper right' corner of the region.
    * \param func is an instance of the lambda function to be performed on each grid entity.
    *     It is supposed to have the following form:
    *
    * ```
    * auto func = [=] __cuda_callable__( const typename Grid< Dimension, Real, Device, Index >::template EntityType<
    * EntityDimension >&entity ) mutable {};
    * ```
    * where \e entity represents given grid entity. See \ref TNL::Meshes::GridEntity.
    * \param args are packed arguments that are going to be passed to the lambda function.
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   forBoundaryEntities( const CoordinatesType& begin, const CoordinatesType& end, Func func, FuncArgs... args ) const;

   /**
    * \brief Iterate over all mesh entities within the local subdomain with given dimension and perform given lambda function
    *    on each of them.
    *
    * Entities processed by this method are such that their coordinates \f$c\f$ fullfil \f$ localBegin \leq c < localEnd\f$.
    *
    * \tparam EntityDimension is dimension of the grid entities.
    *     It is supposed to have the following form:
    *
    * ```
    * auto func = [=] __cuda_callable__( const typename Grid< Dimension, Real, Device, Index >::template EntityType<
    * EntityDimension >&entity ) mutable {};
    * ```
    * where \e entity represents given grid entity. See \ref TNL::Meshes::GridEntity.
    * \param func is an instance of the lambda function to be performed on each grid entity.
    * \param args are packed arguments that are going to be passed to the lambda function.
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   forLocalEntities( Func func, FuncArgs... args ) const;

   template< typename Vector >
   typename Vector::ConstViewType
   partitionEntities( const Vector& allEntities, int entitiesDimension, int entitiesOrientation ) const;

   template< typename Vector >
   typename Vector::ViewType
   partitionEntities( Vector& allEntities, int entitiesDimension, int entitiesOrientation ) const;


protected:

   void
   setEntitiesIndexesOffsets();

   void
   setCoordinatesMultiplicators();

   //void
   //fillEntitiesCount();

   void
   fillSpaceSteps();

   void
   fillSpaceStepsPowers();

   void
   fillProportions();

   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   traverseAll( Func func, FuncArgs... args ) const;

   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   traverseAll( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const;

   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   traverseInterior( Func func, FuncArgs... args ) const;

   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   traverseInterior( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const;

   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   traverseBoundary( Func func, FuncArgs... args ) const;

   template< int EntityDimension, typename Func, typename... FuncArgs >
   void
   traverseBoundary( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const;

   /**
    * \brief Grid dimensions.
    */
   CoordinatesType dimensions = 0;

   PointType origin = 0, proportions = 0, spaceSteps = 0;

   /**
    * \brief Region of subdomain if this grid represents one sudbdomain of distributed grid.
    */
   CoordinatesType localBegin = 0, localEnd = 0;

   CoordinatesType interiorBegin = 0 , interiorEnd = 0;  // TODO: Why we need it?

   EntitiesCounts entitiesCounts = 0;

   // TODO: Explain meaning of this container
   /**
    * \brief Container holding offsets of entities with various orientations.
    *
    * The grid is mapping all entities of the same dimension into one linear container. For
    * entities other then cells and vertexes this includes entities with various orientations.
    * The following figure shows the mapping on an example of faces in 2D grid having sizes 4x4:
    *
    *
    * ```
    *   +-( 36)-+-( 37)-+-( 38)-+-( 39)-+
    *   |       |       |       |       |
    * ( 15)   ( 16)   ( 17)   ( 18)   ( 19)
    *   |       |       |       |       |
    *   +-( 32)-+-( 33)-+-( 34)-+-( 35)-+
    *   |       |       |       |       |
    * ( 10)   ( 11)   ( 12)   ( 13)   ( 14)
    *   |       |       |       |        |
    *   +-( 28)-+-( 29)-+-( 30)-+-( 31)-+
    *   |       |       |       |       |
    * ( 5 )   ( 6 )   ( 7 )   ( 8 )   ( 9 )
    *   |       |       |       |       |
    *   +-( 24)-+-( 25)-+-( 26)-+-( 27)-+
    *   |       |       |       |       |
    * ( 0 )   ( 1 )   ( 2 )   ( 3 )   ( 4 )
    *   |       |       |       |       |
    *   +-( 20)-+-( 21)-+-( 22)-+-( 23)-+
    *```
    *
    * We can see that there are 4*5=20 horizontal faces and 5*4=20 vertical faces.
    * The horizontal faces are mapped to indexes 0..19 and the vertical to 20..39. When it comes
    * to other entities of the 2D grid with sizes 4x4, we have:
    *
    * 1. 5*5 vertexes indexed as 0..24
    * 2. 5*4 vertical faces going along y-axis indexed as 0..19
    * 3. 4*5 horizontal faces going along x-axis indexed as 20..39
    * 4. 4*4 cells indexed as 0..15
    *
    * We encode the indexing offsets of the entities depending on their dimension an orientation as follows:
    *
    * [0,25],       [0,20,40],  [0,16].
    * ^-> vertexes  ^-> faces   ^-> cells
    *
    * Here each bracket is related to entities with the same dimension and within the bracket we have
    * indexing offsets of entities with given orientation. Number of all different entities orientations
    * is \e 2^grid-dimension and for each entity dimension there is one more index at the end of the
    * bracket telling the total number of all entities with given dimension. Therefore the size of
    * the container is \e 2^grid-dimension+(grid-dimension+1) where \e grid-dimension+1 is number of
    * different entities dimensions. The container is therefore organized as follows:
    *
    * [0,25,0,20,40,0,16]
    *
    * The indexing offset of entity with dimension \e EntityDimensions and orientation given by
    * \e totalOrientationIndex can be obtained as \e totalOrientationIndex+EntityDimension.
    *
    * We summarize with one more example with 3D grid having sizes 4x4x4. In this case we have
    *
    * 1. 5*5*5 vertexes indexed as 0..124
    * 2. 5*4*4 faces along y and z axes indexed as 0..79
    * 3. 4*5*4 faces along x and z axes indexed as 80..159
    * 4. 4*4*5 faces along x and y axes indexed as 160..239
    * 5. 5*5*4 edges along z axis indexed as 0..99
    * 6. 5*4*5 edges along y axis indexed as 100..199
    * 7. 4*5*5 edges along x axis indexed as 200..299
    * 8. 4*4*4 cells indexed as 0..63
    *
    * The indexing offsets can be therefore encoded as:
    *
    * [0,125],[0,80,160,240],[0,100,200,300],[0,64].
    */
   Containers::StaticVector< ( 1 << Dimension ) + Dimension + 1, Index > entitiesIndexesOffsets = 0;

   /**
    * \brief This container helps with computation of entities indexes.
    *
    * We explain meaning of this container on an example. Consider again a 3D grid with sizes
    * \e xSize , \e ySize and \e zSize. Consider cell with coordinates
    *
    * ```
    * c = ( i, j, k )
    * ```
    *
    * its index can be computed as
    *
    * ```
    * cell_idx = k * ySize * xSize + j * xSize + i
    * ```
    *
    * If we define a vector \e m as
    *
    * ```
    * m = ( 1, xSize, xSize * ySize )
    * ```
    *
    * we may write
    *
    * ```
    * cell_idx = ( m, c)
    * ```
    *
    * i.e. the index can be computed just as scalar product of the vector of cell coordinates and
    * vector \e m which we refer as coordinates multiplicator.
    *
    * One more example. Consider a face along axes \e y and \e z. There are \e xSize+1 of such faces along
    * x axis, \e ySize along y axis and \e zSize along z axis. The index of such a faces with coordinates
    *
    * ```
    * c = ( i, j, k)
    * ```
    *
    * can be computed as
    *
    * ```
    * faces_idx = k * ( xSize + 1 ) * ySize + j * ( xSize + 1 ) + i
    * ```
    *
    * or by defining the coordinates multiplicator vector \e m as
    *
    * ```
    * m = ( 1, xSize+1,  (xSize+1) * ySize )
    * ```
    *
    * we may write
    *
    * ```
    * face_idx = ( c, m )
    * ```
    *
    * The coordinates multiplicator vector therefore depends on the entity dimension and its orientation,
    * or simply on the total orientation index. The following container holds appropriate coordinates
    * multiplicator vector for each index of total orientation. The container is precomputed every time
    * the when sizes of the grid change in method \ref TNL::Meshes::Grid::setCoordinatesMultiplicators.
    * After that, it makes the evaluation if the grid entity index much simpler and more efficient.
    */
   CoordinatesMultiplicatorsContainer coordinatesMultiplicators;

   // TODO: remove this container
   SpaceProductsContainer spaceStepsProducts = 0; // TODO: remove

   //__cuda_callable__ inline static
   EntitiesOrientations entitiesOrientations; // TODO: make this static - I do not know any good solution working with CUDA

};

/**
 * \brief Serialization of the grid structure.
 *
 * \tparam Dimension is grid dimension.
 * \tparam Real is type of the floating point numbers of grid.
 * \tparam Device is the device to be used for the execution of grid operations.
 * \tparam Index is type for indexing of the mesh entities of grid.
 * \param str is output stream.
 * \param grid is an instance of grid.
 * \return std::ostream& is reference on the input stream.
 */
template< int Dimension, typename Real, typename Device, typename Index >
std::ostream&
operator<<( std::ostream& str, const Grid< Dimension, Real, Device, Index >& grid )
{
   TNL::Logger logger( 100, str );

   grid.writeProlog( logger );

   return str;
}

/**
 * \brief Comparison operator for grid.
 *
 * \tparam Dimension is grid dimension.
 * \tparam Real is type of the floating point numbers of grid.
 * \tparam Device is the device to be used for the execution of grid operations.
 * \tparam Index is type for indexing of the mesh entities of grid.
 * \param lhs is an instance of one grid.
 * \param rhs is an instance of another grid.
 * \return true if both grids are equal.
 * \return false if the grids are different.
 */
template< int Dimension, typename Real, typename Device, typename Index >
bool
operator==( const Grid< Dimension, Real, Device, Index >& lhs, const Grid< Dimension, Real, Device, Index >& rhs )
{
   return lhs.getDimensions() == rhs.getDimensions() && lhs.getOrigin() == rhs.getOrigin()
       && lhs.getProportions() == rhs.getProportions();
}

/**
 * \brief Comparison operator for grid.
 *
 * \tparam Dimension is grid dimension.
 * \tparam Real is type of the floating point numbers of grid.
 * \tparam Device is the device to be used for the execution of grid operations.
 * \tparam Index is type for indexing of the mesh entities of grid.
 * \param lhs is an instance of one grid.
 * \param rhs is an instance of another grid.
 * \return true if the grids are different.
 * \return false if both grids are equal.
 */
template< int Dimension, typename Real, typename Device, typename Index >
bool
operator!=( const Grid< Dimension, Real, Device, Index >& lhs, const Grid< Dimension, Real, Device, Index >& rhs )
{
   return ! ( lhs == rhs );
}

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/Grid.hpp>
#include <TNL/Meshes/GridEntity.h>
