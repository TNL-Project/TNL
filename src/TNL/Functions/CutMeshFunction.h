// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/StaticVector.h>

namespace TNL::Functions {

template< typename MeshFunctionType,
          typename OutMesh,
          typename OutDof,
          int outDimension = OutMesh::getMeshDimension(),
          int codimension = MeshFunctionType::getMeshDimension() - OutMesh::getMeshDimension() >
class CutMeshFunction
{
   template< typename Index, typename Function, typename... FunctionArgs, int dim >
   static void
   staticVectorFor( const Containers::StaticVector< dim, Index >& begin,
                    const Containers::StaticVector< dim, Index >& end,
                    Function f,
                    FunctionArgs... args )
   {
      static_assert( 1 <= dim && dim <= 3, "unsupported dimension" );
      Containers::StaticVector< dim, Index > index;

      if( dim == 1 ) {
         for( index[ 0 ] = begin[ 0 ]; index[ 0 ] < end[ 0 ]; index[ 0 ]++ )
            f( index, args... );
      }

      if( dim == 2 ) {
         for( index[ 1 ] = begin[ 1 ]; index[ 1 ] < end[ 1 ]; index[ 1 ]++ )
            for( index[ 0 ] = begin[ 0 ]; index[ 0 ] < end[ 0 ]; index[ 0 ]++ )
               f( index, args... );
      }

      if( dim == 3 ) {
         for( index[ 2 ] = begin[ 2 ]; index[ 2 ] < end[ 2 ]; index[ 2 ]++ )
            for( index[ 1 ] = begin[ 1 ]; index[ 1 ] < end[ 1 ]; index[ 1 ]++ )
               for( index[ 0 ] = begin[ 0 ]; index[ 0 ] < end[ 0 ]; index[ 0 ]++ )
                  f( index, args... );
      }
   }

public:
   static bool
   Cut( MeshFunctionType& inputMeshFunction,
        OutMesh& outMesh,
        OutDof& outData,
        Containers::StaticVector< outDimension, int > savedDimensions,
        Containers::StaticVector< codimension, int > reducedDimensions,
        Containers::StaticVector< codimension, typename MeshFunctionType::IndexType > fixedIndexes )
   {
      bool inCut;
      Containers::StaticVector< codimension, typename MeshFunctionType::IndexType > localFixedIndexes;

      auto fromData = inputMeshFunction.getData().getData();
      auto fromMesh = inputMeshFunction.getMesh();

      // Set-up Grid
      auto fromDistributedGrid = fromMesh.getDistributedMesh();
      if( fromDistributedGrid != nullptr ) {
         auto toDistributedGrid = outMesh.getDistributedMesh();
         if( toDistributedGrid == nullptr )
            throw std::logic_error(
               "You are trying cut distributed meshfunction, but output grid is not set up for distribution" );

         inCut = toDistributedGrid->SetupByCut( *fromDistributedGrid, savedDimensions, reducedDimensions, fixedIndexes );
         if( inCut ) {
            toDistributedGrid->setupGrid( outMesh );
            for( int i = 0; i < codimension; i++ )
               localFixedIndexes[ i ] = fixedIndexes[ i ] - fromDistributedGrid->getGlobalBegin()[ reducedDimensions[ i ] ];
         }
      }
      else {
         typename OutMesh::PointType outOrigin;
         typename OutMesh::PointType outProportions;
         typename OutMesh::CoordinatesType outDimensions;

         for( int i = 0; i < outDimension; i++ ) {
            outOrigin[ i ] = fromMesh.getOrigin()[ savedDimensions[ i ] ];
            outProportions[ i ] = fromMesh.getProportions()[ savedDimensions[ i ] ];
            outDimensions[ i ] = fromMesh.getSizes()[ savedDimensions[ i ] ];
         }

         outMesh.setSizes( outDimensions );
         outMesh.setDomain( outOrigin, outProportions );

         inCut = true;
         localFixedIndexes = fixedIndexes;
      }

      // copy data
      if( inCut ) {
         outData.setSize( outMesh.template getEntitiesCount< typename OutMesh::Cell >() );
         auto kernel = [ &fromData, &fromMesh, &outData, &outMesh, &savedDimensions, &localFixedIndexes, &reducedDimensions ](
                          typename OutMesh::CoordinatesType index )
         {
            typename MeshFunctionType::MeshType::Cell fromEntity( fromMesh );
            typename OutMesh::Cell outEntity( outMesh );

            for( int j = 0; j < outDimension; j++ ) {
               fromEntity.getCoordinates()[ savedDimensions[ j ] ] = index[ j ];
               outEntity.getCoordinates()[ j ] = index[ j ];
            }

            for( int j = 0; j < codimension; j++ )
               fromEntity.getCoordinates()[ reducedDimensions[ j ] ] = localFixedIndexes[ j ];

            fromEntity.refresh();
            outEntity.refresh();
            outData[ outEntity.getIndex() ] = fromData[ fromEntity.getIndex() ];
         };

         typename OutMesh::CoordinatesType starts;
         starts.setValue( 0 );
         staticVectorFor( starts, outMesh.getSizes(), kernel );
      }

      return inCut;
   }
};

}  // namespace TNL::Functions
