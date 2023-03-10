// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename MeshFunctionType, int dim = MeshFunctionType::getMeshDimension() >
class CopyEntitiesHelper;

template< typename MeshFunctionType >
class CopyEntitiesHelper< MeshFunctionType, 1 >
{
public:
   using CoordinatesType = typename MeshFunctionType::MeshType::CoordinatesType;
   using Cell = typename MeshFunctionType::MeshType::Cell;
   using Index = typename MeshFunctionType::MeshType::GlobalIndexType;

   template< typename FromFunction >
   static void
   Copy( FromFunction& from, MeshFunctionType& to, CoordinatesType& fromBegin, CoordinatesType& toBegin, CoordinatesType& size )
   {
      auto toData = to.getData().getData();
      auto fromData = from.getData().getData();
      auto* fromMesh = &from.getMeshPointer().template getData< typename MeshFunctionType::MeshType::DeviceType >();
      auto* toMesh = &to.getMeshPointer().template getData< typename MeshFunctionType::MeshType::DeviceType >();
      auto kernel = [ fromData, toData, fromMesh, toMesh, fromBegin, toBegin ] __cuda_callable__( Index i )
      {
         Cell fromEntity( *fromMesh );
         Cell toEntity( *toMesh );
         toEntity.getCoordinates().x() = toBegin.x() + i;
         toEntity.refresh();
         fromEntity.getCoordinates().x() = fromBegin.x() + i;
         fromEntity.refresh();
         toData[ toEntity.getIndex() ] = fromData[ fromEntity.getIndex() ];
      };
      Algorithms::parallelFor< typename MeshFunctionType::MeshType::DeviceType >( 0, size.x(), kernel );
   }
};

template< typename MeshFunctionType >

class CopyEntitiesHelper< MeshFunctionType, 2 >
{
public:
   using CoordinatesType = typename MeshFunctionType::MeshType::CoordinatesType;
   using Cell = typename MeshFunctionType::MeshType::Cell;
   using Index = typename MeshFunctionType::MeshType::GlobalIndexType;

   template< typename FromFunction >
   static void
   Copy( FromFunction& from, MeshFunctionType& to, CoordinatesType& fromBegin, CoordinatesType& toBegin, CoordinatesType& size )
   {
      auto toData = to.getData().getData();
      auto fromData = from.getData().getData();
      auto* fromMesh = &from.getMeshPointer().template getData< typename MeshFunctionType::MeshType::DeviceType >();
      auto* toMesh = &to.getMeshPointer().template getData< typename MeshFunctionType::MeshType::DeviceType >();
      auto kernel = [ fromData, toData, fromMesh, toMesh, fromBegin, toBegin ] __cuda_callable__(
                       const Containers::StaticArray< 2, Index >& i )
      {
         Cell fromEntity( *fromMesh );
         Cell toEntity( *toMesh );
         toEntity.getCoordinates().x() = toBegin.x() + i.x();
         toEntity.getCoordinates().y() = toBegin.y() + i.y();
         toEntity.refresh();
         fromEntity.getCoordinates().x() = fromBegin.x() + i.x();
         fromEntity.getCoordinates().y() = fromBegin.y() + i.y();
         fromEntity.refresh();
         toData[ toEntity.getIndex() ] = fromData[ fromEntity.getIndex() ];
      };
      Algorithms::parallelFor< typename MeshFunctionType::MeshType::DeviceType >(
         Containers::StaticArray< 2, Index >{ 0, 0 }, size, kernel );
   }
};

template< typename MeshFunctionType >
class CopyEntitiesHelper< MeshFunctionType, 3 >
{
public:
   using CoordinatesType = typename MeshFunctionType::MeshType::CoordinatesType;
   using Cell = typename MeshFunctionType::MeshType::Cell;
   using Index = typename MeshFunctionType::MeshType::GlobalIndexType;

   template< typename FromFunction >
   static void
   Copy( FromFunction& from, MeshFunctionType& to, CoordinatesType& fromBegin, CoordinatesType& toBegin, CoordinatesType& size )
   {
      auto toData = to.getData().getData();
      auto fromData = from.getData().getData();
      auto* fromMesh = &from.getMeshPointer().template getData< typename MeshFunctionType::MeshType::DeviceType >();
      auto* toMesh = &to.getMeshPointer().template getData< typename MeshFunctionType::MeshType::DeviceType >();
      auto kernel = [ fromData, toData, fromMesh, toMesh, fromBegin, toBegin ] __cuda_callable__(
                       const Containers::StaticArray< 3, Index >& i )
      {
         Cell fromEntity( *fromMesh );
         Cell toEntity( *toMesh );
         toEntity.getCoordinates().x() = toBegin.x() + i.x();
         toEntity.getCoordinates().y() = toBegin.y() + i.y();
         toEntity.getCoordinates().z() = toBegin.z() + i.z();
         toEntity.refresh();
         fromEntity.getCoordinates().x() = fromBegin.x() + i.x();
         fromEntity.getCoordinates().y() = fromBegin.y() + i.y();
         fromEntity.getCoordinates().z() = fromBegin.z() + i.z();
         fromEntity.refresh();
         toData[ toEntity.getIndex() ] = fromData[ fromEntity.getIndex() ];
      };
      Algorithms::parallelFor< typename MeshFunctionType::MeshType::DeviceType >(
         Containers::StaticArray< 3, Index >{ 0, 0, 0 }, size, kernel );
   }
};

}  // namespace DistributedMeshes
}  // namespace Meshes
}  // namespace TNL
