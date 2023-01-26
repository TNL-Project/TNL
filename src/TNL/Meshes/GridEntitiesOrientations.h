// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/GridDetails/NormalsGetter.h>

namespace TNL {
namespace Meshes {

/**
 * \brief Structure holding all grid entities orientations.
 *
 * Grid is a orthogonal regular numerical mesh consisting of various mesh entities having various orientations. For example:
 *
 * 1. 1D-grid consists of:
 *    1. 0D vertexes,
 *    2. 1D cells.
 * 2. 2D-grid consists of:
 *    1. 0D vertexes,
 *    2. 1D faces going along x axis,
 *    3. 1D faces going along y axis,
 *    4. 2D cells.
 * 3. 3D-grid consists of:
 *    1. 0D vertexes,
 *    2. 1D edges going along x axis,
 *    3. 1D edges going along y axis,
 *    4. 1D edges going along z axis,
 *    5. 2D faces spanning along x and y axes,
 *    6. 2D faces spanning along x and z axes,
 *    7. 2D faces spanning along y and z axes,
 *    8. 3D cells.
 *
 * To be able to specify what kind of grid entities we aim to work with we have to encode somehow type and orientation of the entity.
 * Each entity can be given by a subset of vector of [standard basis](https://en.wikipedia.org/wiki/Standard_basis) of \f$R^n\f$ which the entity spans.
 * All the basis vectors generating the grid entity can be packed into one vector having ones at positions where the vectors of the standard basis
 * generating the grid entity have number one. See the following table for examples:
 *
 * | Grid dimension | Entity type              | Entity dimension  | Vectors of standard basis             | Packed vectors of standard basis |
 * |---------------:|-------------------------:|------------------:|--------------------------------------:|---------------------------------:|
 * | 1              | Vertex                   | 0                 | none or ( 0 )                         | ( 0 )                            |
 * | 1              | Cell                     | 1                 | ( 1 )                                 | ( 1 )                            |
 * | 2              | Vertex                   | 0                 | none or ( 0, 0 )                      | ( 0, 0 )                         |
 * | 2              | Face along y axis        | 1                 | ( 0, 1 )                              | ( 0, 1 )                         |
 * | 2              | Face along x axis        | 1                 | ( 1, 0 )                              | ( 1, 0 )                         |
 * | 2              | Cell                     | 2                 | ( 1, 0 ), ( 0, 1 )                    | ( 1, 1 )                         |
 * | 3              | Vertexes                 | 0                 | none or ( 0, 0, 0 )                   | ( 0, 0, 0 )                      |
 * | 3              | Edges along z axis       | 1                 | ( 0, 0, 1 )                           | ( 0, 0, 1 )                      |
 * | 3              | Edges along y axis       | 1                 | ( 0, 1, 0 )                           | ( 0, 1, 0 )                      |
 * | 3              | Edges along x axis       | 1                 | ( 1, 0, 0 )                           | ( 1, 0, 0 )                      |
 * | 3              | Faces along y and z axes | 2                 | ( 0, 1, 0 ), ( 0, 0, 1 )              | ( 0, 1, 1 )                      |
 * | 3              | Faces along x and z axes | 2                 | ( 1, 0, 0 ), ( 0, 0, 1 )              | ( 1, 0, 1 )                      |
 * | 3              | Faces along x and y axes | 2                 | ( 1, 0, 0 ), ( 0, 1, 0 )              | ( 1, 1, 0 )                      |
 * | 3              | Cells                    | 3                 | ( 1, 0, 0 ), ( 0, 1, 0 ), ( 0, 0, 1 ) | ( 1, 1, 1 )                      |
 *
 * Another way to encode the orientation is by vectors of the standard basis which are normal (or orthogonal) to standard basis vectors which the
 * grid entity spans. Clearly the normal vectors are complement to the basis vectors which the entity spans. Even the normal vectors can be packed
 * into one vector. See the following table for examples:
 *
 * | Grid dimension | Entity type              | Entity dimension  | Normal vectors of standard basis      | Packed  normal vectors  |
 * |---------------:|-------------------------:|------------------:|--------------------------------------:|------------------------:|
 * | 1              | Vertex                   | 0                 | ( 1 )                                 | ( 1 )                   |
 * | 1              | Cell                     | 1                 | none or ( 0 )                         | ( 0 )                   |
 * | 2              | Vertex                   | 0                 | ( 1, 0 ), ( 0, 1 )                    | ( 1, 1 )                |
 * | 2              | Face along y axis        | 1                 | ( 1, 0 )                              | ( 1, 0 )                |
 * | 2              | Face along x axis        | 1                 | ( 0, 1 )                              | ( 0, 1 )                |
 * | 2              | Cell                     | 2                 | none or ( 0, 0 )                      | ( 0, 0 )                |
 * | 3              | Vertexes                 | 0                 | ( 1, 0, 0 ), ( 0, 1, 0 ), ( 0, 0, 1 ) | ( 1, 1, 1 )             |
 * | 3              | Edges along z axis       | 1                 | ( 1, 0, 0 ), ( 0, 1, 0 )              | ( 1, 1, 0 )             |
 * | 3              | Edges along y axis       | 1                 | ( 1, 0, 0 ), ( 0, 0, 1 )              | ( 1, 0, 1 )             |
 * | 3              | Edges along x axis       | 1                 | ( 0, 1, 0 ), ( 0, 0, 1 )              | ( 0, 1, 1 )             |
 * | 3              | Faces along y and z axes | 2                 | ( 1, 0, 0 )                           | ( 1, 0, 0 )             |
 * | 3              | Faces along x and z axes | 2                 | ( 0, 1, 0 )                           | ( 0, 1, 0 )             |
 * | 3              | Faces along x and y axes | 2                 | ( 0, 0, 1 )                           | ( 0, 0, 1 )             |
 * | 3              | Cells                    | 3                 | none or ( 0, 0, 0 )                   | ( 0, 0, 0 )             |
 *
 * While the basis vectors generating the grid entity are useful for computing the center of the grid entity, for example, the normal
 * vectors are more suitable for computations of grid entities indexes - see \ref TNL::Meshes::Grid for more details.
 *
 * For these reasons, the packed normal vectors are preferred and the basis vectors generating the grid entity are easily deduced.
 *
 * To make the encoding more efficient we assign orientation indexes to each grid entity orientation. There are two of them:
 *
 *  1. Orientation index - is the index of orientation within all orientations of entities with given dimension.
 *  2. Total orientation index - is the index of orientation within all orientations of all grid entities.
 *
 * See the following table for examples:
 *
 * | Grid dimension | Entity type              | Entity dimension  | Packed  normal vectors  | Orientation idx. | Total orientation idx. |
 * |---------------:|-------------------------:|------------------:|------------------------:|-----------------:|-----------------------:|
 * | 1              | Vertex                   | 0                 | ( 1 )                   | 0                | 0                      |
 * | 1              | Cell                     | 1                 | ( 0 )                   | 1                | 1                      |
 * | 2              | Vertex                   | 0                 | ( 1, 1 )                | 0                | 0                      |
 * | 2              | Face along y axis        | 1                 | ( 1, 0 )                | 0                | 1                      |
 * | 2              | Face along x axis        | 1                 | ( 0, 1 )                | 1                | 2                      |
 * | 2              | Cell                     | 2                 | ( 0, 0 )                | 0                | 3                      |
 * | 3              | Vertexes                 | 0                 | ( 1, 1, 1 )             | 0                | 0                      |
 * | 3              | Edges along z axis       | 1                 | ( 1, 1, 0 )             | 0                | 1                      |
 * | 3              | Edges along y axis       | 1                 | ( 1, 0, 1 )             | 1                | 2                      |
 * | 3              | Edges along x axis       | 1                 | ( 0, 1, 1 )             | 2                | 3                      |
 * | 3              | Faces along y and z axes | 2                 | ( 1, 0, 0 )             | 0                | 4                      |
 * | 3              | Faces along x and z axes | 2                 | ( 0, 1, 0 )             | 1                | 5                      |
 * | 3              | Faces along x and y axes | 2                 | ( 0, 0, 1 )             | 2                | 6                      |
 * | 3              | Cells                    | 3                 | ( 0, 0, 0 )             | 0                | 7                      |
 *
 * The following example demonstrates the use of grid entities orientations in real code:
 *
 * \includelineno Meshes/Grid/GridEntitiesOrientationsExample.cpp
 *
 * The result looks as follows:
 *
 * \include GridEntitiesOrientationsExample.out
 */
template< int GridDimension >
struct GridEntitiesOrientations
{
   /**
    * \brief Type for storing of packed normals defining the entity orientation.
    */
   using NormalsType = Containers::StaticVector< GridDimension, short int >;

   /**
    * \brief Gives number of orientations of all grid entities.
    */
   constexpr static int getTotalOrientationsCount();

   /**
    * \brief Gives dimension of entity based on the total orientation index.
    *
    * \tparam TotalOrientationIndex is total orientation index.
    */
   template< int TotalOrientationIndex >
   constexpr static int getEntityDimension();

   /**
    * \brief Gives dimension of entity based on the total orientation index.
    *
    * \param totalOrientationIndex is total orientation index.
    */
   constexpr static int getEntityDimension( int totalOrientationIndex );

   /**
    * \brief Gives number of orientations of grid entities with given dimension.
    *
    * \tparam EntityDimension is the grid entity dimension.
    */
   template< int EntityDimension >
   constexpr static int getOrientationsCount();

   /**
    * \brief Gives number of orientations of grid entities with given dimension.
    *
    * \param entityDimension is the grid entity dimension.
    */
   constexpr static int getOrientationsCount( int entityDimension );

   /**
    * \brief Gives dimension specific orientation index based on grid entity dimension and packed normal vectors.
    *
    * The index is evaluated at compile time.
    *
    * \tparam EntityDimension is the entity dimension.
    * \tparam Normals is a vector with packed normals given as a template parameter pack.
    * \return constexpr int is the dimension specific orientation index.
    */
   template< int EntityDimension, int... Normals >
   constexpr static int getOrientationIndex();

   template< int EntityDimension >
   constexpr static int getOrientationIndex( int totalOrientationIndex );

   /**
    * \brief Gives total orientation index based on packed normal vectors.
    *
    * The index is evaluated at compile time.
    *
    * \tparam Normals is a vector with packed normals given as a template parameter pack.
    * \return constexpr int is the dimension specific orientation index.
    */
   template< int... Normals >
   constexpr static int getTotalOrientationIndex();

   /**
    * \brief Gives total orientation index based on entity dimension and dimension specific orientation index.
    *
    * \tparam EntityDimension is the entity dimension.
    * \param orientation is the dimension specific index of entity orientation.
    * \return constexpr int is the total orientation index.
    */
   template< int EntityDimension >
   constexpr static int getTotalOrientationIndex( int orientation );

   /**
    * \brief Gives total orientation index based on entity dimension and dimension specific orientation index.
    *
    * \param entityDimension is the entity dimension.
    * \param orientation is the dimension specific index of entity orientation.
    * \return constexpr int is the total orientation index.
    */
   constexpr static int getTotalOrientationIndex( int entityDimension, int orientation );

   /**
    * \brief Gives packed normal vectors based on entity dimension and dimension specific orientation index.
    *
    * The packed normal vectors are evaluated at the compile time.
    *
    * \tparam EntityDimension is the dimension of the entity.
    * \tparam Orientation is the dimension specific orientation index.
    * \return NormalsType are packed normal vectors.
    */
   template< int EntityDimension, int Orientation >
   __cuda_callable__
   static NormalsType getNormals();

   /**
    * \brief Gives packed normal vectors based on total orientation index.
    *
    * The packed normal vectors are evaluated at the compile time.
    *
    * \tparam TotalOrientation is the dimension specific orientation index.
    * \return NormalsType are packed normal vectors.
    */
   template< int TotalOrientation >
   __cuda_callable__
   static NormalsType getNormals();

   /**
    * \brief Constructor with no parameters.
    */
   __cuda_callable__
   GridEntitiesOrientations();

   /**
    * \brief Gives entity dimension based on a packed normal vectors.
    *
    * \param normals represents packed normal vectors.
    * \return entity dimension.
    */
   __cuda_callable__
   static int getEntityDimension( const NormalsType& normals );

   /**
    * \brief Gives entity orientation index based on the packed normal vectors.
    *
    * \param normals represents packed normal vectors.
    * \return entity orientation index.
    */
   __cuda_callable__
   int getOrientationIndex( const NormalsType& normals ) const;

   /**
    * \brief Gives entity total orientation index based on the packed normal vectors.
    *
    * \param normals represents packed normal vectors.
    * \return entity total orientation index.
    */
   __cuda_callable__
   int getTotalOrientationIndex( const NormalsType& normals ) const;

   /**
    * \brief Gives packed normal vectors based on entity dimension and dimension specific orientation index.
    *
    * The packed normal vectors are obtained at the run-time from precomputed table.
    *
    * \tparam EntityDimension is the grid entity dimension.
    * \param orientation is the dimension specific orientation index.
    * \return NormalsType are packed normal vectors.
    */
   template< int EntityDimension >
   __cuda_callable__
   const NormalsType& getNormals( int orientation ) const;

   /**
    * \brief Gives packed normal vectors based on entity dimension and orientation index.
    *
    * The packed normal vectors are obtained at the run-time from precomputed table.
    *
    * \param[in] entityDimension is dimension of grid entity.
    * \param[in] orientation is the orientation index of grid entity.
    * \return NormalsType are packed normal vectors.
    */
   __cuda_callable__
   const NormalsType& getNormals( int entityDimension, int orientation ) const;

   /**
    * \brief Gives packed normal vectors based on total orientation index.
    *
    * The packed normal vectors are obtained at the run-time from precomputed table.
    *
    * \param totalOrientation is the total orientation index.
    * \return NormalsType are packed normal vectors.
    */
   __cuda_callable__
   const NormalsType& getNormals( int totalOrientation ) const;

protected:

   using OrientationNormalsContainer = Containers::StaticVector< getTotalOrientationsCount(), NormalsType >;

   template< int EntityDimension, int Orientation >
   __cuda_callable__
   void addNormalsToTable( int offset );

   OrientationNormalsContainer normalsTable;
};

} //namespace Meshes
} //namespace TNL

#include <TNL/Meshes/GridEntitiesOrientations.hpp>
