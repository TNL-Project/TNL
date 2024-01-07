// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>

namespace TNL::Meshes::DistributedMeshes {

template< typename MeshFunctionType,
          typename PeriodicBoundariesMaskPointer,
          int dim,
          typename RealType = typename MeshFunctionType::MeshType::RealType,
          typename Device = typename MeshFunctionType::MeshType::DeviceType,
          typename Index = typename MeshFunctionType::MeshType::GlobalIndexType >
class BufferEntitiesHelper;

template< typename MeshFunctionType, typename MaskPointer, typename RealType, typename Device, typename Index >
class BufferEntitiesHelper< MeshFunctionType, MaskPointer, 1, RealType, Device, Index >
{
public:
   static void
   BufferEntities( MeshFunctionType& meshFunction,
                   const MaskPointer& maskPointer,
                   RealType* buffer,
                   bool isBoundary,
                   const Containers::StaticArray< 1, Index >& begin,
                   const Containers::StaticArray< 1, Index >& size,
                   bool tobuffer )
   {
      Index beginx = begin.x();
      Index sizex = size.x();

      auto* mesh = &meshFunction.getMeshPointer().template getData< Device >();
      RealType* meshFunctionData = meshFunction.getData().getData();
      const typename MaskPointer::ObjectType* mask( nullptr );
      if( maskPointer )
         mask = &maskPointer.template getData< Device >();
      auto kernel = [ tobuffer, mesh, buffer, isBoundary, meshFunctionData, mask, beginx ] __cuda_callable__( Index j )
      {
         typename MeshFunctionType::MeshType::Cell entity( *mesh );
         entity.getCoordinates().x() = beginx + j;
         entity.refresh();
         if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] ) {
            if( tobuffer )
               buffer[ j ] = meshFunctionData[ entity.getIndex() ];
            else
               meshFunctionData[ entity.getIndex() ] = buffer[ j ];
         }
      };
      Algorithms::parallelFor< Device >( 0, sizex, kernel );
   }
};

template< typename MeshFunctionType, typename MaskPointer, typename RealType, typename Device, typename Index >
class BufferEntitiesHelper< MeshFunctionType, MaskPointer, 2, RealType, Device, Index >
{
public:
   static void
   BufferEntities( MeshFunctionType& meshFunction,
                   const MaskPointer& maskPointer,
                   RealType* buffer,
                   bool isBoundary,
                   const Containers::StaticArray< 2, Index >& begin,
                   const Containers::StaticArray< 2, Index >& size,
                   bool tobuffer )
   {
      auto* mesh = &meshFunction.getMeshPointer().template getData< Device >();
      RealType* meshFunctionData = meshFunction.getData().getData();
      const typename MaskPointer::ObjectType* mask( nullptr );
      if( maskPointer )
         mask = &maskPointer.template getData< Device >();

      auto kernel = [ tobuffer, mask, mesh, buffer, isBoundary, meshFunctionData, begin, size ] __cuda_callable__(
                       const Containers::StaticArray< 2, Index >& i )
      {
         typename MeshFunctionType::MeshType::Cell entity( *mesh );
         entity.getCoordinates().x() = begin.x() + i.x();
         entity.getCoordinates().y() = begin.y() + i.y();
         entity.refresh();
         if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] ) {
            if( tobuffer )
               buffer[ i.y() * size.x() + i.x() ] = meshFunctionData[ entity.getIndex() ];
            else
               meshFunctionData[ entity.getIndex() ] = buffer[ i.y() * size.x() + i.x() ];
         }
      };
      Algorithms::parallelFor< Device >( Containers::StaticArray< 2, Index >{ 0, 0 }, size, kernel );
   }
};

template< typename MeshFunctionType, typename MaskPointer, typename RealType, typename Device, typename Index >
class BufferEntitiesHelper< MeshFunctionType, MaskPointer, 3, RealType, Device, Index >
{
public:
   static void
   BufferEntities( MeshFunctionType& meshFunction,
                   const MaskPointer& maskPointer,
                   RealType* buffer,
                   bool isBoundary,
                   const Containers::StaticArray< 3, Index >& begin,
                   const Containers::StaticArray< 3, Index >& size,
                   bool tobuffer )
   {
      auto* mesh = &meshFunction.getMeshPointer().template getData< Device >();
      RealType* meshFunctionData = meshFunction.getData().getData();
      const typename MaskPointer::ObjectType* mask( nullptr );
      if( maskPointer )
         mask = &maskPointer.template getData< Device >();
      auto kernel = [ tobuffer, mesh, mask, buffer, isBoundary, meshFunctionData, begin, size ] __cuda_callable__(
                       const Containers::StaticArray< 3, Index >& i )
      {
         typename MeshFunctionType::MeshType::Cell entity( *mesh );
         entity.getCoordinates().x() = begin.x() + i.x();
         entity.getCoordinates().y() = begin.y() + i.y();
         entity.getCoordinates().z() = begin.z() + i.z();
         entity.refresh();
         if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] ) {
            if( tobuffer )
               buffer[ i.z() * size.x() * size.y() + i.y() * size.x() + i.x() ] = meshFunctionData[ entity.getIndex() ];
            else
               meshFunctionData[ entity.getIndex() ] = buffer[ i.z() * size.x() * size.y() + i.y() * size.x() + i.x() ];
         }
      };
      Algorithms::parallelFor< Device >( Containers::StaticArray< 3, Index >{ 0, 0, 0 }, size, kernel );
   }
};

}  // namespace TNL::Meshes::DistributedMeshes
