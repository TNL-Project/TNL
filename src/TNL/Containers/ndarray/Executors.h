// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>

#include <TNL/Containers/ndarray/Meta.h>
#include <TNL/Containers/ndarray/SizesHolder.h>

namespace TNL::Containers::detail {

template< typename Permutation, typename Device2 >
struct Functor_call_with_unpermuted_arguments
{
   template< typename MultiIndex, typename Func >
   void
   operator()( MultiIndex i, Func f ) const
   {
      static_assert( 2 <= MultiIndex::getSize() && MultiIndex::getSize() <= 3 );
      if constexpr( MultiIndex::getSize() == 2 )
         call_with_unpermuted_arguments< Permutation >( f, i[ 1 ], i[ 0 ] );
      if constexpr( MultiIndex::getSize() == 3 )
         call_with_unpermuted_arguments< Permutation >( f, i[ 2 ], i[ 1 ], i[ 0 ] );
   }
};

// stupid specialization to avoid a shitpile of nvcc warnings
// (nvcc does not like nested __cuda_callable__ and normal lambdas...)
template< typename Permutation >
struct Functor_call_with_unpermuted_arguments< Permutation, Devices::Cuda >
{
   template< typename MultiIndex, typename Func >
   __cuda_callable__
   void
   operator()( MultiIndex i, Func f ) const
   {
      static_assert( 2 <= MultiIndex::getSize() && MultiIndex::getSize() <= 3 );
      if constexpr( MultiIndex::getSize() == 2 )
         call_with_unpermuted_arguments< Permutation >( f, i[ 1 ], i[ 0 ] );
      if constexpr( MultiIndex::getSize() == 3 )
         call_with_unpermuted_arguments< Permutation >( f, i[ 2 ], i[ 1 ], i[ 0 ] );
   }
};

template< typename Permutation, typename LevelTag = IndexTag< 0 > >
struct SequentialExecutor
{
   template< typename Begins, typename Ends, typename Func, typename... Indices >
   __cuda_callable__
   void
   operator()( const Begins& begins, const Ends& ends, Func f, Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      SequentialExecutor< Permutation, IndexTag< LevelTag::value + 1 > > exec;
      const auto begin = begins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto end = ends.template getSize< get< LevelTag::value >( Permutation{} ) >();
      for( auto i = begin; i < end; i++ )
         exec( begins, ends, f, std::forward< Indices >( indices )..., i );
   }
};

template< typename Permutation >
struct SequentialExecutor< Permutation, IndexTag< Permutation::size() - 1 > >
{
   template< typename Begins, typename Ends, typename Func, typename... Indices >
   __cuda_callable__
   void
   operator()( const Begins& begins, const Ends& ends, Func f, Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );
      static_assert( sizeof...( indices ) == Begins::getDimension() - 1,
                     "invalid number of indices in the final step of the SequentialExecutor" );

      using LevelTag = IndexTag< Permutation::size() - 1 >;

      const auto begin = begins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto end = ends.template getSize< get< LevelTag::value >( Permutation{} ) >();
      for( auto i = begin; i < end; i++ )
         call_with_unpermuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
   }
};

template< typename Permutation, typename LevelTag = IndexTag< Permutation::size() - 1 > >
struct SequentialExecutorRTL
{
   template< typename Begins, typename Ends, typename Func, typename... Indices >
   __cuda_callable__
   void
   operator()( const Begins& begins, const Ends& ends, Func f, Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      SequentialExecutorRTL< Permutation, IndexTag< LevelTag::value - 1 > > exec;
      const auto begin = begins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto end = ends.template getSize< get< LevelTag::value >( Permutation{} ) >();
      for( auto i = begin; i < end; i++ )
         exec( begins, ends, f, i, std::forward< Indices >( indices )... );
   }
};

template< typename Permutation >
struct SequentialExecutorRTL< Permutation, IndexTag< 0 > >
{
   template< typename Begins, typename Ends, typename Func, typename... Indices >
   __cuda_callable__
   void
   operator()( const Begins& begins, const Ends& ends, Func f, Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );
      static_assert( sizeof...( indices ) == Begins::getDimension() - 1,
                     "invalid number of indices in the final step of the SequentialExecutorRTL" );

      const auto begin = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto end = ends.template getSize< get< 0 >( Permutation{} ) >();
      for( auto i = begin; i < end; i++ )
         call_with_unpermuted_arguments< Permutation >( f, i, std::forward< Indices >( indices )... );
   }
};

template< typename Permutation, typename Device >
struct ParallelExecutorDeviceDispatch
{
   template< typename Begins, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const Ends& ends,
               const typename Device::LaunchConfiguration& launch_configuration,
               Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      using Index = typename Ends::IndexType;
      using MultiIndex = Containers::StaticArray< 3, Index >;

      auto kernel = [ = ]( const MultiIndex& i )
      {
         SequentialExecutor< Permutation, IndexTag< 3 > > exec;
         exec( begins, ends, f, i[ 2 ], i[ 1 ], i[ 0 ] );
      };

      const MultiIndex begin = { begins.template getSize< get< 2 >( Permutation{} ) >(),
                                 begins.template getSize< get< 1 >( Permutation{} ) >(),
                                 begins.template getSize< get< 0 >( Permutation{} ) >() };
      const MultiIndex end = { ends.template getSize< get< 2 >( Permutation{} ) >(),
                               ends.template getSize< get< 1 >( Permutation{} ) >(),
                               ends.template getSize< get< 0 >( Permutation{} ) >() };
      Algorithms::parallelFor< Device >( begin, end, launch_configuration, kernel );
   }
};

template< typename Permutation >
struct ParallelExecutorDeviceDispatch< Permutation, Devices::Cuda >
{
   template< typename Begins, typename Ends, typename Func >
   void
   operator()( const Begins& begins, const Ends& ends, const Devices::Cuda::LaunchConfiguration& launch_configuration, Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      using Index = typename Ends::IndexType;
      using MultiIndex = Containers::StaticArray< 3, Index >;

      auto kernel = [ = ] __cuda_callable__( const MultiIndex& i )
      {
         SequentialExecutorRTL< Permutation, IndexTag< Begins::getDimension() - 4 > > exec;
         exec( begins, ends, f, i[ 2 ], i[ 1 ], i[ 0 ] );
      };

      const MultiIndex begin = { begins.template getSize< get< Begins::getDimension() - 1 >( Permutation{} ) >(),
                                 begins.template getSize< get< Begins::getDimension() - 2 >( Permutation{} ) >(),
                                 begins.template getSize< get< Begins::getDimension() - 3 >( Permutation{} ) >() };
      const MultiIndex end = { ends.template getSize< get< Ends::getDimension() - 1 >( Permutation{} ) >(),
                               ends.template getSize< get< Ends::getDimension() - 2 >( Permutation{} ) >(),
                               ends.template getSize< get< Ends::getDimension() - 3 >( Permutation{} ) >() };
      Algorithms::parallelFor< Devices::Cuda >( begin, end, launch_configuration, kernel );
   }
};

template< typename Permutation, typename Device, typename DimTag = IndexTag< Permutation::size() > >
struct ParallelExecutor
{
   template< typename Begins, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const Ends& ends,
               const typename Device::LaunchConfiguration& launch_configuration,
               Func f )
   {
      ParallelExecutorDeviceDispatch< Permutation, Device > dispatch;
      dispatch( begins, ends, launch_configuration, f );
   }
};

template< typename Permutation, typename Device >
struct ParallelExecutor< Permutation, Device, IndexTag< 3 > >
{
   template< typename Begins, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const Ends& ends,
               const typename Device::LaunchConfiguration& launch_configuration,
               Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      using Index = typename Ends::IndexType;
      using MultiIndex = Containers::StaticArray< 3, Index >;

      // nvcc does not like nested __cuda_callable__ and normal lambdas...
      Functor_call_with_unpermuted_arguments< Permutation, Device > kernel;

      const MultiIndex begin = { begins.template getSize< get< 2 >( Permutation{} ) >(),
                                 begins.template getSize< get< 1 >( Permutation{} ) >(),
                                 begins.template getSize< get< 0 >( Permutation{} ) >() };
      const MultiIndex end = { ends.template getSize< get< 2 >( Permutation{} ) >(),
                               ends.template getSize< get< 1 >( Permutation{} ) >(),
                               ends.template getSize< get< 0 >( Permutation{} ) >() };
      Algorithms::parallelFor< Device >( begin, end, launch_configuration, kernel, f );
   }
};

template< typename Permutation, typename Device >
struct ParallelExecutor< Permutation, Device, IndexTag< 2 > >
{
   template< typename Begins, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const Ends& ends,
               const typename Device::LaunchConfiguration& launch_configuration,
               Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      using Index = typename Ends::IndexType;
      using MultiIndex = Containers::StaticArray< 2, Index >;

      // nvcc does not like nested __cuda_callable__ and normal lambdas...
      Functor_call_with_unpermuted_arguments< Permutation, Device > kernel;

      const MultiIndex begin = { begins.template getSize< get< 1 >( Permutation{} ) >(),
                                 begins.template getSize< get< 0 >( Permutation{} ) >() };
      const MultiIndex end = { ends.template getSize< get< 1 >( Permutation{} ) >(),
                               ends.template getSize< get< 0 >( Permutation{} ) >() };
      Algorithms::parallelFor< Device >( begin, end, launch_configuration, kernel, f );
   }
};

template< typename Permutation, typename Device >
struct ParallelExecutor< Permutation, Device, IndexTag< 1 > >
{
   template< typename Begins, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const Ends& ends,
               const typename Device::LaunchConfiguration& launch_configuration,
               Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      using Index = typename Ends::IndexType;

      const Index begin = begins.template getSize< get< 0 >( Permutation{} ) >();
      const Index end = ends.template getSize< get< 0 >( Permutation{} ) >();
      Algorithms::parallelFor< Device >( begin, end, launch_configuration, f );
   }
};

// Device may be void which stands for StaticNDArray
template< typename Permutation, typename Device >
struct ExecutorDispatcher
{
   template< typename Begins, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const Ends& ends,
               const typename Device::LaunchConfiguration& launch_configuration,
               Func f )
   {
      SequentialExecutor< Permutation >()( begins, ends, f );
   }
};

template< typename Permutation >
struct ExecutorDispatcher< Permutation, Devices::Host >
{
   template< typename Begins, typename Ends, typename Func >
   void
   operator()( const Begins& begins, const Ends& ends, const Devices::Host::LaunchConfiguration& launch_configuration, Func f )
   {
      if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 )
         ParallelExecutor< Permutation, Devices::Host >()( begins, ends, launch_configuration, f );
      else
         SequentialExecutor< Permutation >()( begins, ends, f );
   }
};

template< typename Permutation >
struct ExecutorDispatcher< Permutation, Devices::Cuda >
{
   template< typename Begins, typename Ends, typename Func >
   void
   operator()( const Begins& begins, const Ends& ends, const Devices::Cuda::LaunchConfiguration& launch_configuration, Func f )
   {
      ParallelExecutor< Permutation, Devices::Cuda >()( begins, ends, launch_configuration, f );
   }
};

}  // namespace TNL::Containers::detail
