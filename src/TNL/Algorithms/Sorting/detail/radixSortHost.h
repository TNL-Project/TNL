// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#ifdef HAVE_OPENMP
   #include <omp.h>
#endif
#if defined( __SSE2__ )
   #include <immintrin.h>
#endif

#include <TNL/Containers/Array.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Algorithms/Sorting/detail/radixSort.h>

namespace TNL::Algorithms::Sorting::detail {

//! \brief Arrays of at most this many elements are sorted by the stable
//! insertion sort instead of the radix sort on the host.
constexpr int radixSortHostInsertionLimit = 64;

//! \brief Minimal number of elements per thread of the parallel radix sort
//! on \ref Devices::Host; smaller arrays use fewer threads (down to the
//! sequential sort), since the per-pass synchronization and the histogram
//! exchange do not pay off for short chunks.
constexpr int radixSortHostMinChunk = 1 << 15;

//! \brief Minimal number of elements of a chunk scattered with the software
//! write-combining buffers (see \ref radixSortHostScatterBuffered).
constexpr int radixSortHostBufferedLimit = 1 << 12;

//! \brief Minimal size in bytes of the sorted array for which the scatter
//! uses the write-combining buffers with the non-temporal stores; smaller
//! arrays stay in the caches between the passes, where the plain scatter is
//! faster.
constexpr std::size_t radixSortHostNontemporalLimit = std::size_t( 32 ) << 20;

//! \brief Trait telling if the device is one of the host devices supported
//! by \ref radixSortHost.
template< typename Device >
inline constexpr bool isRadixSortHostDevice_v =
   std::is_same_v< Device, Devices::Sequential > || std::is_same_v< Device, Devices::Host >;

/**
 * \brief Workspace of the temporary buffers of the radix sort on the host
 * devices (\ref Devices::Sequential and \ref Devices::Host).
 *
 * The buffers have the same semantics as the device workspace (see
 * \ref RadixSortWorkspace): they are grown on demand and never shrunk, so a
 * workspace reused across sorts performs no allocation once the buffers fit
 * the sorted array.
 *
 * \note The sorts sharing one workspace must not run concurrently.
 */
template< typename Value, typename Device, typename Index >
struct RadixSortHostWorkspace
{
   static_assert( isRadixSortHostDevice_v< Device >, "RadixSortHostWorkspace is meant only for the host devices" );
   static_assert( std::is_arithmetic_v< Value >, "RadixSort supports only arithmetic value types" );
   static_assert( sizeof( Value ) <= 8, "RadixSort supports value types with at most 8 bytes" );

   //! \brief Number of the radix passes (one per byte of the value).
   static constexpr int passes = static_cast< int >( sizeof( Value ) );

   /**
    * \brief Grows the buffers so that an array of `size` elements can be
    * sorted by `threads` threads.
    */
   void
   setSize( const Index size, const int threads = 1 )
   {
      if( aux.getSize() < size )
         aux.setSize( size );
      // the parallel sort needs also the next-pass histograms of all chunks
      // (see radixSortHostParallel)
      const std::size_t requiredCounts =
         static_cast< std::size_t >( threads ) * ( passes + ( threads > 1 ? threads : 0 ) ) * radixSortBuckets;
      if( counts.size() < requiredCounts )
         counts.resize( requiredCounts );
   }

   //! \brief Auxiliary (ping-pong) array of at least the sorted size.
   Containers::Array< Value, Device, Index > aux;

   //! \brief Digit histograms of the threads: `threads` x (`passes` +
   //! `threads`) x `radixSortBuckets` entries for the parallel sort,
   //! `passes` x `radixSortBuckets` entries for the sequential sort.
   std::vector< Index > counts;
};

//! \brief Specialization of the radix sort workspace for \ref Devices::Sequential.
template< typename Value, typename Index >
struct RadixSortWorkspace< Value, Devices::Sequential, Index >
: public RadixSortHostWorkspace< Value, Devices::Sequential, Index >
{};

//! \brief Specialization of the radix sort workspace for \ref Devices::Host.
template< typename Value, typename Index >
struct RadixSortWorkspace< Value, Devices::Host, Index > : public RadixSortHostWorkspace< Value, Devices::Host, Index >
{};

/**
 * \brief Stable insertion sort by the radix sort keys.
 *
 * The keys are compared instead of the values, so that the order of the
 * special floating-point values (signed zeros, NaNs) matches the order
 * produced by the radix sort.
 */
template< typename Value, typename Index, bool descending >
void
radixSortHostInsertion( Value* data, const Index n )
{
   for( Index i = 1; i < n; i++ ) {
      const Value value = data[ i ];
      const auto key = radixSortKey< Value, descending >( value );
      Index j = i;
      while( j > 0 && radixSortKey< Value, descending >( data[ j - 1 ] ) > key ) {
         data[ j ] = data[ j - 1 ];
         j--;
      }
      data[ j ] = value;
   }
}

//! \brief Extracts the digit of the given pass from a radix sort key.
template< typename Value >
inline unsigned
radixSortHostDigit( const RadixSortKey< Value > key, const int pass )
{
   return static_cast< unsigned >( key >> ( radixSortBits * pass ) ) & ( radixSortBuckets - 1 );
}

/**
 * \brief Accumulates the digit histograms of all passes of the elements
 * `data[ begin, end )` into `counts` (`passes` x `radixSortBuckets` entries).
 */
template< typename Value, typename Index, bool descending >
void
radixSortHostCountAll( const Value* data, const Index begin, const Index end, Index* counts )
{
   constexpr int passes = static_cast< int >( sizeof( Value ) );
   for( Index i = begin; i < end; i++ ) {
      const auto key = radixSortKey< Value, descending >( data[ i ] );
      for( int pass = 0; pass < passes; pass++ )
         counts[ pass * radixSortBuckets + radixSortHostDigit< Value >( key, pass ) ]++;
   }
}

//! \brief Counter of the scatter which does nothing.
struct RadixSortHostNoCounter
{
   template< typename Key, typename Index >
   void
   operator()( const Key& /* key */, const unsigned /* digit */, const Index /* position */ )
   {}
};

/**
 * \brief Counter of the scatter accumulating the digit histograms of the
 * next pass for the chunks of the target array.
 *
 * The parallel sort splits the array into the same chunks in every pass, so
 * the digit histogram of a chunk for the next pass can be counted while the
 * elements are being scattered into the chunk: each thread counts the
 * elements it writes into each target chunk in its private table with
 * `numThreads` x `radixSortBuckets` entries, and the owner of the chunk sums
 * the tables of all threads afterwards. This saves one read of the whole
 * array per pass. The target chunk of each digit run is tracked
 * incrementally, since the positions of a digit run only grow.
 */
template< typename Value, typename Index >
struct RadixSortHostNextPassCounter
{
   //! \brief Initializes the counter for the scatter with the given
   //! `offsets` (the start positions of the digit runs).
   RadixSortHostNextPassCounter(
      Index* counts,
      const int nextPass,
      const Index chunk,
      const Index remainder,
      const int numThreads,
      const Index* offsets )
   : counts( counts ),
     nextPass( nextPass ),
     chunk( chunk ),
     remainder( remainder ),
     numThreads( numThreads )
   {
      std::fill( counts, counts + static_cast< std::size_t >( numThreads ) * radixSortBuckets, Index( 0 ) );
      for( int digit = 0; digit < radixSortBuckets; digit++ ) {
         const int owner = ownerOf( offsets[ digit ] );
         chunkEnd[ digit ] = chunkBegin( owner + 1 );
         ownerCounts[ digit ] = counts + static_cast< std::size_t >( owner ) * radixSortBuckets;
      }
   }

   void
   operator()( const RadixSortKey< Value > key, const unsigned digit, const Index position )
   {
      if( position >= chunkEnd[ digit ] ) {
         // the run continues in the next chunk(s)
         const int owner = ownerOf( position );
         chunkEnd[ digit ] = chunkBegin( owner + 1 );
         ownerCounts[ digit ] = counts + static_cast< std::size_t >( owner ) * radixSortBuckets;
      }
      ownerCounts[ digit ][ radixSortHostDigit< Value >( key, nextPass ) ]++;
   }

private:
   Index
   chunkBegin( const int t ) const
   {
      return chunk * t + TNL::min( static_cast< Index >( t ), remainder );
   }

   int
   ownerOf( const Index position ) const
   {
      // the first `remainder` chunks have `chunk + 1` elements
      const Index bigChunksEnd = ( chunk + 1 ) * remainder;
      int owner = 0;
      if( position < bigChunksEnd )
         owner = static_cast< int >( position / ( chunk + 1 ) );
      else
         owner = static_cast< int >( remainder + ( position - bigChunksEnd ) / chunk );
      return TNL::min( owner, numThreads - 1 );
   }

   Index* counts;
   int nextPass;
   Index chunk;
   Index remainder;
   int numThreads;
   std::array< Index, radixSortBuckets > chunkEnd;
   std::array< Index*, radixSortBuckets > ownerCounts;
};

/**
 * \brief Stably scatters the elements `src[ begin, end )` to `dst` by the
 * digit of the given pass; `offsets` holds the target position of the next
 * element of each digit and it is advanced by the scatter.
 *
 * The `counter` is called with the sort key, the digit and the target
 * position of each scattered element (see \ref RadixSortHostNextPassCounter).
 */
template< typename Value, typename Index, bool descending, typename Counter >
void
radixSortHostScatter(
   const Value* src,
   Value* dst,
   const Index begin,
   const Index end,
   const int pass,
   Index* offsets,
   Counter& counter )
{
   for( Index i = begin; i < end; i++ ) {
      const Value value = src[ i ];
      const auto key = radixSortKey< Value, descending >( value );
      const unsigned digit = radixSortHostDigit< Value >( key, pass );
      const Index position = offsets[ digit ]++;
      dst[ position ] = value;
      counter( key, digit, position );
   }
}

#if defined( __SSE2__ )
//! \brief Size of the cache line assumed by the buffered scatter.
constexpr int radixSortHostCacheLine = 64;

/**
 * \brief Scatter with software write-combining buffers and non-temporal
 * stores.
 *
 * Behaves like \ref radixSortHostScatter, but the elements of each digit are
 * first collected in a cache-line-sized buffer, which is written to the
 * target array once the corresponding cache line of the target is complete.
 * The 256 buffers fit in the L1 cache and the full lines are written by
 * non-temporal (streaming) stores, which bypass the cache hierarchy and
 * avoid reading the target line before the write (read-for-ownership). This
 * reduces the memory traffic of the scatter by a third, which pays off when
 * the scatter is limited by the memory bandwidth, i.e., for large arrays
 * sorted by many threads. The buffers are aligned to the cache lines of the
 * *target* array, so that a full line never overlaps the elements written
 * by other threads or with other digits.
 *
 * The non-temporal stores are available only on x86, the other platforms
 * use the plain scatter.
 */
template< typename Value, typename Index, bool descending, typename Counter >
void
radixSortHostScatterBuffered(
   const Value* src,
   Value* dst,
   const Index begin,
   const Index end,
   const int pass,
   Index* offsets,
   Counter& counter )
{
   constexpr unsigned lineSize = radixSortHostCacheLine / sizeof( Value );
   static_assert( lineSize * sizeof( Value ) == radixSortHostCacheLine );

   alignas( radixSortHostCacheLine ) Value buffer[ radixSortBuckets * lineSize ];
   // the first slot of the current line owned by each digit (non-zero only
   // for the first, partial line of the digit run)
   unsigned char lineStart[ radixSortBuckets ];
   // shift of the element indices aligning the lines of the buffer with the
   // cache lines of the target array
   const std::size_t alignShift = reinterpret_cast< std::uintptr_t >( dst ) / sizeof( Value );
   const auto slotOf = [ alignShift ]( const Index position ) -> unsigned
   {
      return static_cast< unsigned >( ( static_cast< std::size_t >( position ) + alignShift ) % lineSize );
   };
   for( int digit = 0; digit < radixSortBuckets; digit++ )
      lineStart[ digit ] = static_cast< unsigned char >( slotOf( offsets[ digit ] ) );

   for( Index i = begin; i < end; i++ ) {
      const Value value = src[ i ];
      const auto key = radixSortKey< Value, descending >( value );
      const unsigned digit = radixSortHostDigit< Value >( key, pass );
      const Index position = offsets[ digit ]++;
      counter( key, digit, position );
      const unsigned slot = slotOf( position );
      Value* line = buffer + digit * lineSize;
      line[ slot ] = value;
      if( slot == lineSize - 1 ) {
         const unsigned first = lineStart[ digit ];
         Value* target = dst + ( position - static_cast< Index >( slot - first ) );
         if( first == 0 ) {
            auto* out = reinterpret_cast< __m128i* >( target );
            const auto* in = reinterpret_cast< const __m128i* >( line );
            for( int k = 0; k < radixSortHostCacheLine / 16; k++ )
               _mm_stream_si128( out + k, _mm_load_si128( in + k ) );
         }
         else {
            std::memcpy(
               static_cast< void* >( target ),
               static_cast< const void* >( line + first ),
               ( lineSize - first ) * sizeof( Value ) );
            lineStart[ digit ] = 0;
         }
      }
   }

   // flush the incomplete lines
   for( int digit = 0; digit < radixSortBuckets; digit++ ) {
      const unsigned first = lineStart[ digit ];
      const unsigned last = slotOf( offsets[ digit ] );
      if( last > first )
         std::memcpy(
            static_cast< void* >( dst + ( offsets[ digit ] - static_cast< Index >( last - first ) ) ),
            static_cast< const void* >( buffer + digit * lineSize + first ),
            ( last - first ) * sizeof( Value ) );
   }

   // make the streaming stores visible to the other threads
   _mm_sfence();
}
#endif

/**
 * \brief Scatter selecting the variant for the given chunk.
 *
 * \param buffered tells whether the buffered scatter with the non-temporal
 *                 stores should be used (see \ref radixSortHostUseBuffered).
 */
template< typename Value, typename Index, bool descending, typename Counter >
void
radixSortHostScatterAuto(
   const Value* src,
   Value* dst,
   const Index begin,
   const Index end,
   const int pass,
   Index* offsets,
   [[maybe_unused]] const bool buffered,
   Counter& counter )
{
#if defined( __SSE2__ )
   if( buffered && end - begin >= static_cast< Index >( radixSortHostBufferedLimit ) ) {
      radixSortHostScatterBuffered< Value, Index, descending >( src, dst, begin, end, pass, offsets, counter );
      return;
   }
#endif
   radixSortHostScatter< Value, Index, descending >( src, dst, begin, end, pass, offsets, counter );
}

/**
 * \brief Tells if the scatter of an array of `n` values sorted by `threads`
 * threads should use the buffered scatter with the non-temporal stores.
 *
 * The non-temporal stores pay off only when the array does not fit in the
 * caches and the scatter is limited by the memory bandwidth. A single
 * thread is rather limited by the latency of the scattered stores, where
 * the buffered scatter was measured to be faster only for the values of at
 * most 4 bytes (on AMD Zen 3).
 */
template< typename Value, typename Index >
bool
radixSortHostUseBuffered( const Index n, const int threads )
{
   return static_cast< std::size_t >( n ) * sizeof( Value ) >= radixSortHostNontemporalLimit
       && ( threads > 1 || sizeof( Value ) <= 4 );
}

/**
 * \brief Single-threaded LSD radix sort of `data[ 0, n )` using the
 * auxiliary array `aux` and the histogram buffer `counts` with
 * `passes` x `radixSortBuckets` entries.
 */
template< typename Value, typename Index, bool descending >
void
radixSortHostSequential( Value* data, Value* aux, const Index n, Index* counts )
{
   constexpr int passes = static_cast< int >( sizeof( Value ) );

   // the digit counts do not depend on the order of the elements, so the
   // histograms of all passes are computed by a single read of the input
   std::fill( counts, counts + passes * radixSortBuckets, Index( 0 ) );
   radixSortHostCountAll< Value, Index, descending >( data, Index( 0 ), n, counts );

   Value* src = data;
   Value* dst = aux;
   for( int pass = 0; pass < passes; pass++ ) {
      Index* passCounts = counts + pass * radixSortBuckets;
      // a pass with a single non-zero bucket is the identity permutation
      if( passCounts[ radixSortDigit< Value, descending >( src[ 0 ], pass ) ] == n )
         continue;
      Index offset = 0;
      for( int digit = 0; digit < radixSortBuckets; digit++ ) {
         const Index count = passCounts[ digit ];
         passCounts[ digit ] = offset;
         offset += count;
      }
      RadixSortHostNoCounter counter;
      radixSortHostScatterAuto< Value, Index, descending >(
         src, dst, Index( 0 ), n, pass, passCounts, radixSortHostUseBuffered< Value >( n, 1 ), counter );
      std::swap( src, dst );
   }

   if( src != data )
      std::memcpy(
         static_cast< void* >( data ), static_cast< const void* >( src ), static_cast< std::size_t >( n ) * sizeof( Value ) );
}

#ifdef HAVE_OPENMP
/**
 * \brief Size of the histogram buffer (in entries) of one thread of the
 * parallel radix sort: the digit histograms of all passes of its chunk and
 * the next-pass histograms of all target chunks (see
 * \ref RadixSortHostNextPassCounter).
 */
template< typename Value >
constexpr std::size_t
radixSortHostThreadCountsSize( const int threads )
{
   return ( sizeof( Value ) + static_cast< std::size_t >( threads ) ) * radixSortBuckets;
}

/**
 * \brief Multi-threaded LSD radix sort of `data[ 0, n )` using OpenMP.
 *
 * The array is split into one contiguous chunk per thread, the chunks are
 * the same in all passes. Each pass computes the target offset of each
 * (thread, digit) pair from the digit histograms of the chunks (the digit
 * runs are ordered by the digit and, within a digit, by the thread, which
 * preserves the stability) and scatters the chunks. The histograms of the
 * first executed pass are taken from the upfront count of all passes, which
 * also detects the skippable passes; the histograms of the following passes
 * are counted during the scatter of the preceding pass (see
 * \ref RadixSortHostNextPassCounter), so each pass reads and writes the
 * array only once.
 *
 * \param counts is the histogram buffer with `threads` x
 *               \ref radixSortHostThreadCountsSize entries.
 */
template< typename Value, typename Index, bool descending >
void
radixSortHostParallel( Value* data, Value* aux, const Index n, Index* counts, const int threads )
{
   constexpr int passes = static_cast< int >( sizeof( Value ) );
   constexpr int histogramSize = passes * radixSortBuckets;
   const std::size_t threadCountsSize = radixSortHostThreadCountsSize< Value >( threads );
   const bool buffered = radixSortHostUseBuffered< Value >( n, threads );

   // the passes which are not skipped
   std::array< int, passes > executedPasses{};
   int executed = 0;

   #pragma omp parallel num_threads( threads )
   {
      const int thread = omp_get_thread_num();
      const int numThreads = omp_get_num_threads();
      // balanced chunks without overflowing the index type
      const Index chunk = n / numThreads;
      const Index remainder = n % numThreads;
      const auto chunkBegin = [ chunk, remainder ]( const int t ) -> Index
      {
         return chunk * t + TNL::min( static_cast< Index >( t ), remainder );
      };
      const Index begin = chunkBegin( thread );
      const Index end = chunkBegin( thread + 1 );
      const auto threadCounts = [ counts, threadCountsSize ]( const int t ) -> Index*
      {
         return counts + static_cast< std::size_t >( t ) * threadCountsSize;
      };
      // histograms of the chunk of this thread for all passes
      Index* localCounts = threadCounts( thread );
      // histograms of the next pass of all chunks counted by this thread
      Index* nextCounts = localCounts + histogramSize;

      std::fill( localCounts, localCounts + histogramSize, Index( 0 ) );
      radixSortHostCountAll< Value, Index, descending >( data, begin, end, localCounts );
      #pragma omp barrier

      // detect the passes with a single non-zero bucket (identity permutations)
      #pragma omp single
      {
         for( int pass = 0; pass < passes; pass++ ) {
            const unsigned digit = radixSortDigit< Value, descending >( data[ 0 ], pass );
            Index total = 0;
            for( int t = 0; t < numThreads; t++ )
               total += threadCounts( t )[ pass * radixSortBuckets + digit ];
            if( total != n )
               executedPasses[ executed++ ] = pass;
         }
      }  // implicit barrier

      Value* src = data;
      Value* dst = aux;
      for( int k = 0; k < executed; k++ ) {
         const int pass = executedPasses[ k ];
         // the skipped passes do not change the order, so the upfront counts
         // of the chunks are valid for the first executed pass; the counts of
         // the other passes are summed from the next-pass histograms
         if( k > 0 ) {
            Index* passCounts = localCounts + pass * radixSortBuckets;
            std::fill( passCounts, passCounts + radixSortBuckets, Index( 0 ) );
            for( int t = 0; t < numThreads; t++ ) {
               const Index* threadNextCounts = threadCounts( t ) + histogramSize + thread * radixSortBuckets;
               for( int digit = 0; digit < radixSortBuckets; digit++ )
                  passCounts[ digit ] += threadNextCounts[ digit ];
            }
            #pragma omp barrier
         }

         // target offsets of this thread: all elements with smaller digits
         // plus the elements with the same digit in the preceding chunks
         std::array< Index, radixSortBuckets > offsets;
         Index offset = 0;
         for( int digit = 0; digit < radixSortBuckets; digit++ ) {
            for( int t = 0; t < numThreads; t++ ) {
               if( t == thread )
                  offsets[ digit ] = offset;
               offset += threadCounts( t )[ pass * radixSortBuckets + digit ];
            }
         }

         if( k + 1 < executed ) {
            RadixSortHostNextPassCounter< Value, Index > counter(
               nextCounts, executedPasses[ k + 1 ], chunk, remainder, numThreads, offsets.data() );
            radixSortHostScatterAuto< Value, Index, descending >(
               src, dst, begin, end, pass, offsets.data(), buffered, counter );
         }
         else {
            RadixSortHostNoCounter counter;
            radixSortHostScatterAuto< Value, Index, descending >(
               src, dst, begin, end, pass, offsets.data(), buffered, counter );
         }
         std::swap( src, dst );
         // the scatter of the other threads must finish before the next pass
         // reads the target array and the next-pass histograms
         #pragma omp barrier
      }

      if( executed % 2 == 1 )
         std::memcpy(
            static_cast< void* >( data + begin ),
            static_cast< const void* >( aux + begin ),
            static_cast< std::size_t >( end - begin ) * sizeof( Value ) );
   }
}
#endif

/**
 * \brief Sorts the given view on a host device using LSD radix sort,
 * reusing the given workspace of temporary buffers.
 *
 * The algorithm processes the bytes of the order-preserving sort key (see
 * \ref valueToRadixSortKey) from the least significant to the most
 * significant one, so it produces the same order as the GPU implementation.
 * The passes whose histogram has a single non-zero bucket are skipped.
 *
 * On \ref Devices::Host, the sort runs in parallel with OpenMP (when it is
 * enabled) with at most \ref Devices::Host::getMaxThreadsCount threads and
 * at least \ref radixSortHostMinChunk elements per thread. Arrays of at most
 * \ref radixSortHostInsertionLimit elements are sorted by the insertion sort
 * and the workspace is not touched.
 *
 * \tparam descending tells whether to sort in descending order.
 */
template< typename Value, typename Device, typename Index, bool descending >
void
radixSortHost( Containers::ArrayView< Value, Device, Index > view, RadixSortWorkspace< Value, Device, Index >& workspace )
{
   static_assert( isRadixSortHostDevice_v< Device >, "radixSortHost supports only the host devices" );

   const Index n = view.getSize();
   if( n <= static_cast< Index >( radixSortHostInsertionLimit ) ) {
      radixSortHostInsertion< Value, Index, descending >( view.getData(), n );
      return;
   }

   int threads = 1;
#ifdef HAVE_OPENMP
   if constexpr( std::is_same_v< Device, Devices::Host > ) {
      if( Devices::Host::isOMPEnabled() ) {
         const Index maxThreads = n / static_cast< Index >( radixSortHostMinChunk );
         threads = static_cast< int >( TNL::min( static_cast< Index >( Devices::Host::getMaxThreadsCount() ), maxThreads ) );
         threads = TNL::max( threads, 1 );
      }
   }
#endif

   workspace.setSize( n, threads );
#ifdef HAVE_OPENMP
   if( threads > 1 ) {
      radixSortHostParallel< Value, Index, descending >(
         view.getData(), workspace.aux.getData(), n, workspace.counts.data(), threads );
      return;
   }
#endif
   radixSortHostSequential< Value, Index, descending >( view.getData(), workspace.aux.getData(), n, workspace.counts.data() );
}

/**
 * \brief One-shot overload of \ref radixSortHost which allocates a fresh
 * workspace per call.
 */
template< typename Value, typename Device, typename Index, bool descending >
void
radixSortHost( Containers::ArrayView< Value, Device, Index > view )
{
   RadixSortWorkspace< Value, Device, Index > workspace;
   radixSortHost< Value, Device, Index, descending >( view, workspace );
}

}  // namespace TNL::Algorithms::Sorting::detail
