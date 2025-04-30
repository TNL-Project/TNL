#pragma once

namespace TNL {
namespace ParticleSystem {

//note:
//row-major - std::index_sequence< 0, 1 >
//column-major - std::index_sequence< 1, 0 >

template< int Dimension >
struct DefaultPermutation;

template <>
struct DefaultPermutation< 2 >
{
   using value = std::index_sequence< 0, 1 >;
};

template <>
struct DefaultPermutation< 3 >
{
   using value = std::index_sequence< 0, 1, 2 >;
};

template< int Dimension, typename Permutation = typename DefaultPermutation< Dimension >::value >
class SimpleCellIndex
{};

template< typename Permutation  >
class SimpleCellIndex< 2, Permutation >
{
public:

   template< typename IndexType, typename IndexVectorType >
   __cuda_callable__
   static uint32_t
   EvaluateCellIndex( const IndexType& i, const IndexType& j, const IndexVectorType& gridSize )
   {
      if constexpr( std::is_same_v< Permutation, std::index_sequence< 0, 1 > > )
         return j * gridSize[ 0 ] + i;

      if constexpr( std::is_same_v< Permutation, std::index_sequence< 1, 0 > > )
         return i * gridSize[ 1 ] + j;
   }

   template< typename IndexVectorType >
   __cuda_callable__
   static uint32_t
   EvaluateCellIndex( const IndexVectorType& i, const IndexVectorType& gridSize )
   {
      if constexpr( std::is_same_v< Permutation, std::index_sequence< 0, 1 > > )
         return i[ 1 ] * gridSize[ 0 ] + i[ 0 ];

      if constexpr( std::is_same_v< Permutation, std::index_sequence< 1, 0 > > )
         return i[ 0 ] * gridSize[ 1 ] + i[ 1 ];
   }

   template< typename PointType, typename IndexVectorType, typename RealType >
   __cuda_callable__
   static uint32_t
   EvaluateCellIndex( const PointType& r,
                      const PointType& gridOrigin,
                      const IndexVectorType& gridDimension,
                      const RealType& searchRadius )
   {
      if constexpr( std::is_same_v< Permutation, std::index_sequence< 0, 1 > > )
         return TNL::floor( ( r[ 0 ] - gridOrigin[ 0 ] ) / searchRadius ) + \
                TNL::floor( ( r[ 1 ] - gridOrigin[ 1 ] ) / searchRadius ) * gridDimension[ 0 ];

      if constexpr( std::is_same_v< Permutation, std::index_sequence< 1, 0 > > )
         return TNL::floor( ( r[ 1 ] - gridOrigin[ 1 ] ) / searchRadius ) + \
                TNL::floor( ( r[ 0 ] - gridOrigin[ 0 ] ) / searchRadius ) * gridDimension[ 1 ];
   }
};

template< typename Permutation >
class SimpleCellIndex< 3, Permutation >
{
public:

   template< typename IndexType, typename IndexVectorType >
   __cuda_callable__
   static uint32_t
   EvaluateCellIndex( const IndexType& i, const IndexType& j, const IndexType& k, const IndexVectorType& gridSize )
   {
      if constexpr( std::is_same_v< Permutation, std::index_sequence< 0, 1, 2 > > )
         return k * gridSize[ 0 ] * gridSize[ 1 ] + j * gridSize[ 0 ] + i;

      if constexpr( std::is_same_v< Permutation, std::index_sequence< 2, 1, 0 > > )
         return i * gridSize[ 1 ] * gridSize[ 2 ] + j * gridSize[ 1 ] + k;
   }

   template< typename IndexVectorType >
   __cuda_callable__
   static uint32_t
   EvaluateCellIndex( const IndexVectorType& i, const IndexVectorType& gridSize )
   {
      if constexpr( std::is_same_v< Permutation, std::index_sequence< 0, 1, 2 > > )
         return i[ 2 ] * gridSize[ 0 ] * gridSize[ 1 ] + i[ 1 ] * gridSize[ 0 ] + i[ 0 ];

      if constexpr( std::is_same_v< Permutation, std::index_sequence< 2, 1, 0 > > )
         return i[ 0 ] * gridSize[ 1 ] * gridSize[ 2 ] + i[ 1 ] * gridSize[ 1 ] + i[ 2 ];
   }

   template< typename PointType, typename IndexVectorType, typename RealType >
   __cuda_callable__
   static uint32_t
   EvaluateCellIndex( const PointType& r,
                      const PointType& gridOrigin,
                      const IndexVectorType& gridDimension,
                      const RealType& searchRadius )
   {
      if constexpr( std::is_same_v< Permutation, std::index_sequence< 0, 1, 2 > > )
         return TNL::floor( ( r[ 0 ] - gridOrigin[ 0 ] ) / searchRadius ) +
                TNL::floor( ( r[ 1 ] - gridOrigin[ 1 ] ) / searchRadius ) * gridDimension[ 0 ] +
                TNL::floor( ( r[ 2 ] - gridOrigin[ 2 ] ) / searchRadius ) * gridDimension[ 0 ] * gridDimension[ 1 ];

      if constexpr( std::is_same_v< Permutation, std::index_sequence< 2, 1, 0 > > )
         return TNL::floor( ( r[ 2 ] - gridOrigin[ 2 ] ) / searchRadius ) +
                TNL::floor( ( r[ 1 ] - gridOrigin[ 1 ] ) / searchRadius ) * gridDimension[ 1 ] +
                TNL::floor( ( r[ 0 ] - gridOrigin[ 0 ] ) / searchRadius ) * gridDimension[ 1 ] * gridDimension[ 2 ];
   }
};

}  //namespace ParticleSystem
}  //namespace TNL

