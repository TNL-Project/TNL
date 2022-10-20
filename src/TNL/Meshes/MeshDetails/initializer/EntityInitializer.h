// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/initializer/SubentitySeedsCreator.h>
#include <TNL/Atomic.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/staticFor.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class Initializer;

template< int subdimension, int superdimension, typename MeshConfig >
void
initializeSuperentities( Initializer< MeshConfig >& meshInitializer, typename Initializer< MeshConfig >::MeshType& mesh )
{
   using MeshType = typename Initializer< MeshConfig >::MeshType;
   using GlobalIndexType = typename MeshType::GlobalIndexType;
   using LocalIndexType = typename MeshType::LocalIndexType;
   using SuperentityMatrixType = typename MeshType::MeshTraitsType::SuperentityMatrixType;
   using NeighborCountsArray = typename MeshType::MeshTraitsType::NeighborCountsArray;
   using SuperentityTopology = typename MeshType::MeshTraitsType::template EntityTraits< superdimension >::EntityTopology;
   using SubentitySeedsCreatorType = SubentitySeedsCreator< MeshType, SuperentityTopology, DimensionTag< subdimension > >;
   using SeedType = typename SubentitySeedsCreatorType::SubentitySeed;

   static constexpr bool subentityStorage = MeshConfig::subentityStorage( superdimension, subdimension );
   static constexpr bool superentityStorage = MeshConfig::superentityStorage( subdimension, superdimension );

   // std::cout << "   Initiating superentities with dimension " << superdimension << " for subentities with
   // dimension " << subdimension << " ... " << std::endl;

   const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< subdimension >();
   const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< superdimension >();

   if constexpr( subentityStorage && (subdimension > 0 || std::is_same_v< SuperentityTopology, Topologies::Polyhedron >) ) {
      NeighborCountsArray capacities( superentitiesCount );

      Algorithms::ParallelFor< Devices::Host >::exec(
         GlobalIndexType{ 0 },
         superentitiesCount,
         [ & ]( GlobalIndexType superentityIndex )
         {
            capacities[ superentityIndex ] = SubentitySeedsCreatorType::getSubentitiesCount( mesh, superentityIndex );
         } );

      meshInitializer.template initSubentityMatrix< superdimension, subdimension >( capacities, subentitiesCount );
   }

   typename NeighborCountsArray::ViewType superentitiesCountsView;
   if constexpr( superentityStorage ) {
      // counter for superentities of each subentity
      NeighborCountsArray& superentitiesCounts =
         meshInitializer.template getSuperentitiesCountsArray< subdimension, superdimension >();
      superentitiesCounts.setSize( subentitiesCount );
      superentitiesCounts.setValue( 0 );
      superentitiesCountsView.bind( superentitiesCounts );
   }

   if constexpr( subentityStorage || superentityStorage ) {
      Algorithms::ParallelFor< Devices::Host >::exec(
         GlobalIndexType{ 0 },
         superentitiesCount,
         [ & ]( GlobalIndexType superentityIndex )
         {
            LocalIndexType i = 0;
            SubentitySeedsCreatorType::iterate(
               mesh,
               superentityIndex,
               [ & ]( SeedType& seed )
               {
                  const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( seed );

                  // Subentity indices for SubdimensionTag::value == 0 of non-polyhedral meshes were already set up from
                  // seeds
                  if constexpr( subentityStorage
                                && (subdimension > 0 || std::is_same_v< SuperentityTopology, Topologies::Polyhedron >) )
                     meshInitializer.template setSubentityIndex< superdimension, subdimension >(
                        superentityIndex, i++, subentityIndex );

                  if constexpr( superentityStorage ) {
                     Algorithms::AtomicOperations< Devices::Host >::add( superentitiesCountsView[ subentityIndex ],
                                                                         LocalIndexType{ 1 } );
                  }
               } );
         } );
   }

   if constexpr( superentityStorage ) {
      // allocate superentities storage
      SuperentityMatrixType& matrix = meshInitializer.template getSuperentitiesMatrix< subdimension, superdimension >();
      matrix.setDimensions( subentitiesCount, superentitiesCount );
      matrix.setRowCapacities( superentitiesCountsView );
      superentitiesCountsView.setValue( 0 );

      // initialize superentities storage
      if constexpr( subentityStorage ) {
         for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
            for( LocalIndexType i = 0;
                 i < mesh.template getSubentitiesCount< superdimension, subdimension >( superentityIndex );
                 i++ ) {
               const GlobalIndexType subentityIndex =
                  mesh.template getSubentityIndex< superdimension, subdimension >( superentityIndex, i );
               auto row = matrix.getRow( subentityIndex );
               row.setElement( superentitiesCountsView[ subentityIndex ]++, superentityIndex, true );
            }
         }
      }
      else {
         for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
            SubentitySeedsCreatorType::iterate(
               mesh,
               superentityIndex,
               [ & ]( SeedType& seed )
               {
                  const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( seed );
                  auto row = matrix.getRow( subentityIndex );
                  row.setElement( superentitiesCountsView[ subentityIndex ]++, superentityIndex, true );
               } );
         }
      }
   }
}

template< typename MeshConfig >
void
initializeFacesOfPolyhedrons( Initializer< MeshConfig >& meshInitializer, typename Initializer< MeshConfig >::MeshType& mesh )
{
   using MeshType = typename Initializer< MeshConfig >::MeshType;
   using GlobalIndexType = typename MeshType::GlobalIndexType;
   using LocalIndexType = typename MeshType::LocalIndexType;
   using SuperentityMatrixType = typename MeshType::MeshTraitsType::SuperentityMatrixType;
   using NeighborCountsArray = typename MeshType::MeshTraitsType::NeighborCountsArray;

   static constexpr int subdimension = 2;
   static constexpr int superdimension = 3;
   static constexpr bool subentityStorage = MeshConfig::subentityStorage( superdimension, subdimension );
   static constexpr bool superentityStorage = MeshConfig::superentityStorage( subdimension, superdimension );
   static_assert( subentityStorage );

   // std::cout << "   Initiating superentities with dimension " << superdimension << " for subentities with
   // dimension " << subdimension << " ... " << std::endl;

   const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< subdimension >();
   const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< superdimension >();

   auto& cellSeeds = meshInitializer.getCellSeeds();

   typename NeighborCountsArray::ViewType superentitiesCountsView;
   if constexpr( superentityStorage ) {
      // counter for superentities of each subentity
      NeighborCountsArray& superentitiesCounts =
         meshInitializer.template getSuperentitiesCountsArray< subdimension, superdimension >();
      superentitiesCounts.setSize( subentitiesCount );
      superentitiesCounts.setValue( 0 );
      superentitiesCountsView.bind( superentitiesCounts );

      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
         const auto cellSeed = cellSeeds.getSeed( superentityIndex );
         for( LocalIndexType i = 0; i < cellSeed.getCornersCount(); i++ ) {
            const GlobalIndexType subentityIndex = cellSeed.getCornerId( i );
            superentitiesCountsView[ subentityIndex ]++;
         }
      }
   }

   auto& subvertexMatrix = meshInitializer.template getSubentitiesMatrix< superdimension, subdimension >();
   subvertexMatrix = std::move( cellSeeds.getMatrix() );
   meshInitializer.template initSubentitiesCounts< superdimension, subdimension >( cellSeeds.getEntityCornerCounts() );

   if constexpr( superentityStorage ) {
      // allocate superentities storage
      SuperentityMatrixType& matrix = meshInitializer.template getSuperentitiesMatrix< subdimension, superdimension >();
      matrix.setDimensions( subentitiesCount, superentitiesCount );
      matrix.setRowCapacities( superentitiesCountsView );
      superentitiesCountsView.setValue( 0 );

      // initialize superentities storage
      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
         for( LocalIndexType i = 0; i < mesh.template getSubentitiesCount< superdimension, subdimension >( superentityIndex );
              i++ ) {
            const GlobalIndexType subentityIndex =
               mesh.template getSubentityIndex< superdimension, subdimension >( superentityIndex, i );
            auto row = matrix.getRow( subentityIndex );
            row.setElement( superentitiesCountsView[ subentityIndex ]++, superentityIndex, true );
         }
      }
   }
}

template< typename MeshConfig, int EntityDimension >
class EntityInitializer
{
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< EntityDimension >;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;

   using SeedType = EntitySeed< MeshConfig, typename EntityTraitsType::EntityTopology >;
   using InitializerType = Initializer< MeshConfig >;
   using MeshType = typename InitializerType::MeshType;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SeedMatrixType = typename EntityTraitsType::SeedMatrixType;

   static constexpr bool subvertexStorage = MeshConfig::subentityStorage( EntityDimension, 0 );

public:
   static void
   initSubvertexMatrix( NeighborCountsArray& capacities, InitializerType& initializer )
   {
      if constexpr( subvertexStorage )
         initializer.template initSubentityMatrix< EntityDimension, 0 >( capacities );
   }

   static void
   initSubvertexMatrix( SeedMatrixType& seeds, InitializerType& initializer )
   {
      if constexpr( subvertexStorage ) {
         auto& subvertexMatrix = initializer.template getSubentitiesMatrix< EntityDimension, 0 >();
         subvertexMatrix = std::move( seeds.getMatrix() );
         initializer.template initSubentitiesCounts< EntityDimension, 0 >( seeds.getEntityCornerCounts() );
      }
   }

   static void
   initEntity( const GlobalIndexType entityIndex, const SeedType& entitySeed, InitializerType& initializer )
   {
      if constexpr( subvertexStorage ) {
         // this is necessary if we want to use existing entities instead of intermediate seeds to create subentity seeds
         for( LocalIndexType i = 0; i < entitySeed.getCornerIds().getSize(); i++ )
            initializer.template setSubentityIndex< EntityDimension, 0 >( entityIndex, i, entitySeed.getCornerIds()[ i ] );
      }
   }

   static void
   initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      Algorithms::staticFor< int, EntityDimension + 1, MeshType::getMeshDimension() + 1 >(
         [ & ]( auto dim )
         {
            // transform dim to ensure decrementing steps in the loop
            constexpr int superdimension = MeshType::getMeshDimension() + EntityDimension + 1 - dim;

            using SuperentityTopology =
               typename MeshType::MeshTraitsType::template EntityTraits< superdimension >::EntityTopology;

            if constexpr( EntityDimension == 2 && superdimension == 3
                          && std::is_same_v< SuperentityTopology, Topologies::Polyhedron > )
               initializeFacesOfPolyhedrons( meshInitializer, mesh );
            else
               initializeSuperentities< EntityDimension, superdimension >( meshInitializer, mesh );
         } );
   }
};

}  // namespace Meshes
}  // namespace TNL
