// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

//#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/Topologies/IsDynamicTopology.h>
#include <TNL/Devices/Host.h>

namespace TNL::Meshes {

template< typename MeshConfig, typename Device >
class MeshTraits;

template< typename MeshConfig,
          typename EntityTopology,
          bool IsDynamicTopology = Topologies::IsDynamicTopology< EntityTopology >::value >
class EntitySeedMatrix;

template< typename MeshConfig, typename EntityTopology >
class EntitySeedMatrix< MeshConfig, EntityTopology, false >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Devices::Host >;

public:
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using SubentityTraitsType = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityMatrixType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension >;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;

   EntitySeedMatrix() = default;

   EntitySeedMatrix( const EntitySeedMatrix& other ) = default;

   EntitySeedMatrix( EntitySeedMatrix&& other ) noexcept = default;

   EntitySeedMatrix&
   operator=( const EntitySeedMatrix& other ) = default;

   EntitySeedMatrix&
   operator=( EntitySeedMatrix&& other ) noexcept( false ) = default;

   class EntitySeedMatrixSeed
   {
      using RowView = typename SubentityMatrixType::RowView;

   public:
      EntitySeedMatrixSeed( const RowView& matrixRow ) : row( matrixRow ) {}

      [[nodiscard]] static constexpr LocalIndexType
      getCornersCount()
      {
         return SubentityTraitsType::count;
      }

      void
      setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
      {
         TNL_ASSERT_GE( cornerIndex, 0, "corner index must be non-negative" );
         TNL_ASSERT_LT( cornerIndex, getCornersCount(), "corner index is out of bounds" );
         TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );
         this->row.setColumnIndex( cornerIndex, pointIndex );
      }

      template< typename... IndexTypes >
      void
      setCornerIds( const IndexTypes&... pointIndices )
      {
         static_assert( sizeof...( pointIndices ) == getCornersCount(), "invalid number of indices" );
         setCornerIds_impl( 0, pointIndices... );
      }

      [[nodiscard]] GlobalIndexType
      getCornerId( const LocalIndexType& cornerIndex ) const
      {
         return this->row.getColumnIndex( cornerIndex );
      }

   private:
      RowView row;

      // empty overload to terminate recursion
      void
      setCornerIds_impl( const LocalIndexType& cornerIndex )
      {}

      template< typename... IndexTypes >
      void
      setCornerIds_impl( const LocalIndexType& cornerIndex,
                         const GlobalIndexType& pointIndex,
                         const IndexTypes&... pointIndices )
      {
         setCornerId( cornerIndex, pointIndex );
         setCornerIds_impl( cornerIndex + 1, pointIndices... );
      }
   };

   class ConstEntitySeedMatrixSeed
   {
      using ConstRowView = typename SubentityMatrixType::ConstRowView;

   public:
      ConstEntitySeedMatrixSeed( const ConstRowView& matrixRow ) : row( matrixRow ) {}

      [[nodiscard]] static constexpr LocalIndexType
      getCornersCount()
      {
         return SubentityTraitsType::count;
      }

      [[nodiscard]] GlobalIndexType
      getCornerId( const LocalIndexType& cornerIndex ) const
      {
         return this->row.getColumnIndex( cornerIndex );
      }

   private:
      ConstRowView row;
   };

   void
   setDimensions( const GlobalIndexType& entitiesCount, const GlobalIndexType& pointsCount )
   {
      matrix.setDimensions( entitiesCount, pointsCount );

      NeighborCountsArray capacities( entitiesCount );
      capacities.setValue( SubentityTraitsType::count );
      matrix.setRowCapacities( capacities );
   }

   // This method is only here for compatibility with specialization for dynamic entity topologies
   void
   setEntityCornersCounts( const NeighborCountsArray& counts )
   {}

   // This method is only here for compatibility with specialization for dynamic entity topologies
   void
   setEntityCornersCounts( NeighborCountsArray&& counts )
   {}

   void
   reset()
   {
      matrix.reset();
   }

   [[nodiscard]] GlobalIndexType
   getEntitiesCount() const
   {
      return matrix.getRows();
   }

   [[nodiscard]] SubentityMatrixType&
   getMatrix()
   {
      return matrix;
   }

   [[nodiscard]] const SubentityMatrixType&
   getMatrix() const
   {
      return matrix;
   }

   [[nodiscard]] NeighborCountsArray
   getEntityCornerCounts() const
   {
      NeighborCountsArray counts( getEntitiesCount() );
      counts.setValue( SubentityTraitsType::count );
      return counts;
   }

   [[nodiscard]] bool
   empty() const
   {
      return getEntitiesCount() == 0;
   }

   [[nodiscard]] EntitySeedMatrixSeed
   getSeed( const GlobalIndexType& entityIndex )
   {
      return EntitySeedMatrixSeed( matrix.getRow( entityIndex ) );
   }

   [[nodiscard]] ConstEntitySeedMatrixSeed
   getSeed( const GlobalIndexType& entityIndex ) const
   {
      return ConstEntitySeedMatrixSeed( matrix.getRow( entityIndex ) );
   }

private:
   SubentityMatrixType matrix;
};

template< typename MeshConfig >
class EntitySeedMatrix< MeshConfig, Topologies::Vertex, false >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Devices::Host >;

public:
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using SubentityMatrixType = typename MeshTraitsType::template SubentityMatrixType< 0 >;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;

   EntitySeedMatrix() = default;

   EntitySeedMatrix( const EntitySeedMatrix& other ) = default;

   EntitySeedMatrix( EntitySeedMatrix&& other ) noexcept = default;

   EntitySeedMatrix&
   operator=( const EntitySeedMatrix& other ) = default;

   EntitySeedMatrix&
   operator=( EntitySeedMatrix&& other ) noexcept( false ) = default;

   class EntitySeedMatrixSeed
   {
      using RowView = typename SubentityMatrixType::RowView;

   public:
      EntitySeedMatrixSeed( const RowView& matrixRow ) : row( matrixRow ) {}

      [[nodiscard]] static constexpr LocalIndexType
      getCornersCount()
      {
         return 1;
      }

      void
      setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
      {
         TNL_ASSERT_GE( cornerIndex, 0, "corner index must be non-negative" );
         TNL_ASSERT_LT( cornerIndex, getCornersCount(), "corner index is out of bounds" );
         TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );
         this->row.setColumnIndex( cornerIndex, pointIndex );
      }

      [[nodiscard]] GlobalIndexType
      getCornerId( const LocalIndexType& cornerIndex ) const
      {
         return this->row.getColumnIndex( cornerIndex );
      }

   private:
      RowView row;
   };

   class ConstEntitySeedMatrixSeed
   {
      using ConstRowView = typename SubentityMatrixType::ConstRowView;

   public:
      ConstEntitySeedMatrixSeed( const ConstRowView& matrixRow ) : row( matrixRow ) {}

      [[nodiscard]] static constexpr LocalIndexType
      getCornersCount()
      {
         return 1;
      }

      [[nodiscard]] GlobalIndexType
      getCornerId( const LocalIndexType& cornerIndex ) const
      {
         return this->row.getColumnIndex( cornerIndex );
      }

   private:
      ConstRowView row;
   };

   void
   setDimensions( const GlobalIndexType& entitiesCount, const GlobalIndexType& pointsCount )
   {
      matrix.setDimensions( entitiesCount, pointsCount );

      NeighborCountsArray capacities( entitiesCount );
      capacities.setValue( 1 );
      matrix.setRowCapacities( capacities );
   }

   // This method is only here for compatibility with specialization for dynamic entity topologies
   void
   setEntityCornersCounts( const NeighborCountsArray& counts )
   {}

   // This method is only here for compatibility with specialization for dynamic entity topologies
   void
   setEntityCornersCounts( NeighborCountsArray&& counts )
   {}

   void
   reset()
   {
      matrix.reset();
   }

   [[nodiscard]] GlobalIndexType
   getEntitiesCount() const
   {
      return matrix.getRows();
   }

   [[nodiscard]] SubentityMatrixType&
   getMatrix()
   {
      return matrix;
   }

   [[nodiscard]] const SubentityMatrixType&
   getMatrix() const
   {
      return matrix;
   }

   [[nodiscard]] NeighborCountsArray
   getEntityCornerCounts() const
   {
      NeighborCountsArray counts( getEntitiesCount() );
      counts.setValue( 1 );
      return counts;
   }

   [[nodiscard]] bool
   empty() const
   {
      return getEntitiesCount() == 0;
   }

   [[nodiscard]] EntitySeedMatrixSeed
   getSeed( const GlobalIndexType& entityIndex )
   {
      return EntitySeedMatrixSeed( matrix.getRow( entityIndex ) );
   }

   [[nodiscard]] ConstEntitySeedMatrixSeed
   getSeed( const GlobalIndexType& entityIndex ) const
   {
      return ConstEntitySeedMatrixSeed( matrix.getRow( entityIndex ) );
   }

private:
   SubentityMatrixType matrix;
};

template< typename MeshConfig, typename EntityTopology >
class EntitySeedMatrix< MeshConfig, EntityTopology, true >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Devices::Host >;

public:
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using SubentityMatrixType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension >;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;

   EntitySeedMatrix() = default;

   EntitySeedMatrix( const EntitySeedMatrix& other ) = default;

   EntitySeedMatrix( EntitySeedMatrix&& other ) noexcept = default;

   EntitySeedMatrix&
   operator=( const EntitySeedMatrix& other ) = default;

   EntitySeedMatrix&
   operator=( EntitySeedMatrix&& other ) noexcept( false ) = default;

   class EntitySeedMatrixSeed
   {
      using RowView = typename SubentityMatrixType::RowView;

   public:
      EntitySeedMatrixSeed( const RowView& matrixRow, const LocalIndexType& corners )
      : row( matrixRow ), cornersCount( corners )
      {}

      [[nodiscard]] LocalIndexType
      getCornersCount() const
      {
         return cornersCount;
      }

      void
      setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
      {
         TNL_ASSERT_GE( cornerIndex, 0, "corner index must be non-negative" );
         TNL_ASSERT_LT( cornerIndex, getCornersCount(), "corner index is out of bounds" );
         TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );
         this->row.setColumnIndex( cornerIndex, pointIndex );
      }

      [[nodiscard]] GlobalIndexType
      getCornerId( const LocalIndexType& cornerIndex ) const
      {
         return this->row.getColumnIndex( cornerIndex );
      }

   private:
      RowView row;
      LocalIndexType cornersCount;
   };

   class ConstEntitySeedMatrixSeed
   {
      using ConstRowView = typename SubentityMatrixType::ConstRowView;

   public:
      ConstEntitySeedMatrixSeed( const ConstRowView& matrixRow, const LocalIndexType& corners )
      : row( matrixRow ), cornersCount( corners )
      {}

      [[nodiscard]] LocalIndexType
      getCornersCount() const
      {
         return cornersCount;
      }

      [[nodiscard]] GlobalIndexType
      getCornerId( const LocalIndexType& cornerIndex ) const
      {
         return this->row.getColumnIndex( cornerIndex );
      }

   private:
      ConstRowView row;
      LocalIndexType cornersCount;
   };

   void
   setDimensions( const GlobalIndexType& entitiesCount, const GlobalIndexType& pointsCount )
   {
      counts.setSize( entitiesCount );
      matrix.setDimensions( entitiesCount, pointsCount );
   }

   void
   setEntityCornersCounts( const NeighborCountsArray& counts_ )
   {
      this->counts = counts_;
      matrix.setRowCapacities( this->counts );
   }

   void
   setEntityCornersCounts( NeighborCountsArray&& counts_ )
   {
      this->counts = std::move( counts_ );
      matrix.setRowCapacities( this->counts );
   }

   void
   reset()
   {
      matrix.reset();
      counts.reset();
   }

   [[nodiscard]] GlobalIndexType
   getEntitiesCount() const
   {
      return matrix.getRows();
   }

   [[nodiscard]] SubentityMatrixType&
   getMatrix()
   {
      return matrix;
   }

   [[nodiscard]] const SubentityMatrixType&
   getMatrix() const
   {
      return matrix;
   }

   [[nodiscard]] NeighborCountsArray&
   getEntityCornerCounts()
   {
      return counts;
   }

   [[nodiscard]] const NeighborCountsArray&
   getEntityCornerCounts() const
   {
      return counts;
   }

   [[nodiscard]] bool
   empty() const
   {
      return getEntitiesCount() == 0;
   }

   [[nodiscard]] EntitySeedMatrixSeed
   getSeed( const GlobalIndexType& entityIndex )
   {
      return EntitySeedMatrixSeed( matrix.getRow( entityIndex ), counts[ entityIndex ] );
   }

   [[nodiscard]] ConstEntitySeedMatrixSeed
   getSeed( const GlobalIndexType& entityIndex ) const
   {
      return ConstEntitySeedMatrixSeed( matrix.getRow( entityIndex ), counts[ entityIndex ] );
   }

private:
   SubentityMatrixType matrix;
   NeighborCountsArray counts;
};

}  // namespace TNL::Meshes
