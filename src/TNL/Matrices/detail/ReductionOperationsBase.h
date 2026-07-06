// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/compress.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Matrices::detail {

template< typename Matrix >
struct ReductionOperationsBase
{
   using IndexType = typename Matrix::IndexType;
   using DeviceType = typename Matrix::DeviceType;
   using ConstMatrixView = typename Matrix::ConstViewType;

   // ===================== reduceRowsIf (range) =====================

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsIf(
      Matrix& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return 0;
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndexes = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndexes.getSize() == 0 )
         return 0;
      filteredIndexes += begin;
      ReductionOperations< Matrix >::reduceRows(
         matrix,
         filteredIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return filteredIndexes.getSize();
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsIf(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return 0;
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndexes = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndexes.getSize() == 0 )
         return 0;
      filteredIndexes += begin;
      ReductionOperations< Matrix >::reduceRows(
         matrix,
         filteredIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return filteredIndexes.getSize();
   }

   // ===================== reduceRowsIf (array) =====================

   template<
      typename Array,
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsIf(
      Matrix& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return 0;
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      auto rowIndexes_view = rowIndexes.getConstView();

      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndices = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndices.getSize() == 0 )
         return 0;

      VectorType filteredRowIndexes( filteredIndices.getSize() );
      auto filteredRowIndexes_view = filteredRowIndexes.getView();
      auto filteredIndices_view = filteredIndices.getConstView();
      Algorithms::parallelFor< DeviceType >(
         (IndexType) 0,
         filteredIndices.getSize(),
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            filteredRowIndexes_view[ i ] = rowIndexes_view[ filteredIndices_view[ i ] + begin ];
         } );

      ReductionOperations< Matrix >::reduceRows(
         matrix,
         filteredRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return filteredRowIndexes.getSize();
   }

   template<
      typename Array,
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsIf(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return 0;
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      auto rowIndexes_view = rowIndexes.getConstView();

      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndices = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndices.getSize() == 0 )
         return 0;

      VectorType filteredRowIndexes( filteredIndices.getSize() );
      auto filteredRowIndexes_view = filteredRowIndexes.getView();
      auto filteredIndices_view = filteredIndices.getConstView();
      Algorithms::parallelFor< DeviceType >(
         (IndexType) 0,
         filteredIndices.getSize(),
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            filteredRowIndexes_view[ i ] = rowIndexes_view[ filteredIndices_view[ i ] + begin ];
         } );

      ReductionOperations< Matrix >::reduceRows(
         matrix,
         filteredRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return filteredRowIndexes.getSize();
   }

   // ===================== reduceRowsWithArgumentIf (range) =====================

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf(
      Matrix& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return 0;
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndexes = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndexes.getSize() == 0 )
         return 0;
      filteredIndexes += begin;
      ReductionOperations< Matrix >::reduceRowsWithArgument(
         matrix,
         filteredIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return filteredIndexes.getSize();
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return 0;
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndexes = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndexes.getSize() == 0 )
         return 0;
      filteredIndexes += begin;
      ReductionOperations< Matrix >::reduceRowsWithArgument(
         matrix,
         filteredIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return filteredIndexes.getSize();
   }

   // ===================== reduceRowsWithArgumentIf (array) =====================

   template<
      typename Array,
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf(
      Matrix& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return 0;
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      auto rowIndexes_view = rowIndexes.getConstView();

      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndices = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndices.getSize() == 0 )
         return 0;

      VectorType filteredRowIndexes( filteredIndices.getSize() );
      auto filteredRowIndexes_view = filteredRowIndexes.getView();
      auto filteredIndices_view = filteredIndices.getConstView();
      Algorithms::parallelFor< DeviceType >(
         (IndexType) 0,
         filteredIndices.getSize(),
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            filteredRowIndexes_view[ i ] = rowIndexes_view[ filteredIndices_view[ i ] + begin ];
         } );

      ReductionOperations< Matrix >::reduceRowsWithArgument(
         matrix,
         filteredRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return filteredRowIndexes.getSize();
   }

   template<
      typename Array,
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return 0;
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      auto rowIndexes_view = rowIndexes.getConstView();

      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndices = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndices.getSize() == 0 )
         return 0;

      VectorType filteredRowIndexes( filteredIndices.getSize() );
      auto filteredRowIndexes_view = filteredRowIndexes.getView();
      auto filteredIndices_view = filteredIndices.getConstView();
      Algorithms::parallelFor< DeviceType >(
         (IndexType) 0,
         filteredIndices.getSize(),
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            filteredRowIndexes_view[ i ] = rowIndexes_view[ filteredIndices_view[ i ] + begin ];
         } );

      ReductionOperations< Matrix >::reduceRowsWithArgument(
         matrix,
         filteredRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return filteredRowIndexes.getSize();
   }
};

}  // namespace TNL::Matrices::detail
