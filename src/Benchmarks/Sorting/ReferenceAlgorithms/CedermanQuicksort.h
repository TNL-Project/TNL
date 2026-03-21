#pragma once

#include <algorithm>
#include <stdexcept>

#include <TNL/Backend/Macros.h>
#include <TNL/Containers/ArrayView.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/Sorting/detail/blockBitonicSort.h>

#define MAXTHREADS 256
#define MAXBLOCKS 2048

/**
 * The main sort function
 * @param data		Data to be sorted
 * @param size		The length of the data
 */
void
gpuqsort(
   unsigned int* data,
   unsigned int size,
   unsigned int blockscount = 0,
   unsigned int threads = 0,
   unsigned int sbsize = 0,
   unsigned int phase = 0 );

// Keep tracks of the data blocks in phase one
template< typename element >
struct BlockSize
{
   unsigned int beg;
   unsigned int end;
   unsigned int orgbeg;
   unsigned int orgend;
   element rmaxpiv;
   element lmaxpiv;
   element rminpiv;
   element lminpiv;

   bool altered;
   bool flip;
   element pivot;
};

// Holds parameters to the kernel in phase one
template< typename element >
struct Params
{
   unsigned int from;
   unsigned int end;
   element pivot;
   unsigned int ptr;
   bool last;
};

// Used to perform a cumulative sum between blocks.
// Unnecessary for cards with atomic operations.
// Will be removed when these becomes more common
template< typename element >
struct Length
{
   element maxpiv[ MAXBLOCKS ];
   element minpiv[ MAXBLOCKS ];

   unsigned int left[ MAXBLOCKS ];
   unsigned int right[ MAXBLOCKS ];
};

// Since we have divided up the kernel in to three
// we need to remember the result of the cumulative sum
// Unnecessary for cards with atomic operations.
// Will be removed when these becomes more common
struct Hist
{
   unsigned int left[ MAXTHREADS * MAXBLOCKS ];
   unsigned int right[ MAXTHREADS * MAXBLOCKS ];
};

struct LQSortParams
{
   unsigned int beg;
   unsigned int end;
   bool flip;
   unsigned int sbsize;
};

template< typename element >
class GPUQSort
{
   element* data2 = nullptr;
   Params< element >* params = nullptr;
   Params< element >* dparams = nullptr;

   LQSortParams* lqparams = nullptr;
   LQSortParams* dlqparams = nullptr;

   Hist* dhists = nullptr;
   Length< element >* dlength = nullptr;
   Length< element >* length = nullptr;
   BlockSize< element >* workset = nullptr;

   float TK, TM, MK, MM, SM, SK;

   bool init = false;

public:
   GPUQSort();
   ~GPUQSort();

   void
   sort(
      element* data,
      unsigned int size,
      unsigned int blockscount = 0,
      unsigned int threads = 0,
      unsigned int sbsize = 0,
      unsigned int phase = 0 );
};

#undef THREADS

#define THREADS blockDim.x

extern __shared__ unsigned int sarray[];

#ifdef HASATOMICS
__device__
unsigned int ohtotal = 0;
#endif

/**
 * Perform a cumulative count on two arrays
 * @param lblock Array one
 * @param rblock Array two
 */
__device__
inline void
cumcount( unsigned int* lblock, unsigned int* rblock )
{
   int tx = threadIdx.x;

   int offset = 1;

   __syncthreads();

   for( int d = THREADS >> 1; d > 0; d >>= 1 )  // build sum in place up the tree
   {
      __syncthreads();

      if( tx < d ) {
         int ai = offset * ( 2 * tx + 1 ) - 1;
         int bi = offset * ( 2 * tx + 2 ) - 1;
         lblock[ bi ] += lblock[ ai ];
         rblock[ bi ] += rblock[ ai ];
      }
      offset *= 2;
   }
   __syncthreads();
   if( tx == 0 ) {
      lblock[ THREADS ] = lblock[ THREADS - 1 ];
      rblock[ THREADS ] = rblock[ THREADS - 1 ];
      lblock[ THREADS - 1 ] = 0;
      rblock[ THREADS - 1 ] = 0;
   }  // clear the last unsigned int */
   __syncthreads();

   for( int d = 1; d < THREADS; d *= 2 )  // traverse down tree & build scan
   {
      offset >>= 1;
      __syncthreads();

      if( tx < d ) {
         int ai = offset * ( 2 * tx + 1 ) - 1;
         int bi = offset * ( 2 * tx + 2 ) - 1;

         int t = lblock[ ai ];
         lblock[ ai ] = lblock[ bi ];
         lblock[ bi ] += t;

         t = rblock[ ai ];
         rblock[ ai ] = rblock[ bi ];
         rblock[ bi ] += t;
      }
   }
}

/**
 * Part One - Counts the number of unsigned ints larger or smaller than the pivot. It then
 * performs a cumulative sum so that each thread knows where to write
 * @param data   unsigned ints to be counted
 * @param params Specifies which data each thread block is responsible for
 * @param hist   The cumulative sum for each thread is stored here
 * @param lengths The total sum for each thread block is stored here
 */
//template <typename unsigned int>
__global__
void
part1( const unsigned int* data, Params< unsigned int >* params, Hist* hist, Length< unsigned int >* lengths )
{
   const int tx = threadIdx.x;

   auto* lblock = static_cast< unsigned int* >( sarray );
   auto* rblock = &lblock[ blockDim.x + 1 ];
   auto* minpiv = &rblock[ blockDim.x + 1 ];
   auto* maxpiv = &minpiv[ blockDim.x ];

   // Where should we read?
   unsigned int start = params[ blockIdx.x ].from;
   unsigned int end = params[ blockIdx.x ].end;
   unsigned int pivot = params[ blockIdx.x ].pivot;

   // Stores the max and min value of the data. Used to decide a new pivot
   minpiv[ tx ] = data[ start + tx ];
   maxpiv[ tx ] = data[ start + tx ];

   __syncthreads();
   int ll = 0;
   int lr = 0;

   __syncthreads();

   int coal = start & 0xf;
   start = start - coal;

   // Go through the data
   if( tx + start < end ) {
      unsigned int d = data[ tx + start ];

      if( ! ( tx < coal ) ) {
         // Counting unsigned ints smaller...
         if( d < pivot )
            ll++;
         else
            // or larger than the pivot
            if( d > pivot )
               lr++;

         // Store the max and min unsigned int
         minpiv[ tx ] = TNL::min( minpiv[ tx ], d );
         maxpiv[ tx ] = TNL::max( maxpiv[ tx ], d );
      }
   }

   // Go through the data
   for( unsigned int i = tx + start + THREADS; i < end; i += THREADS ) {
      unsigned int d = data[ i ];

      // Counting unsigned ints smaller...
      if( d < pivot )
         ll++;
      else
         // or larger than the pivot
         if( d > pivot )
            lr++;

      // Store the max and min unsigned int
      minpiv[ tx ] = TNL::min( minpiv[ tx ], d );
      maxpiv[ tx ] = TNL::max( maxpiv[ tx ], d );
   }

   lblock[ tx ] = ll;
   rblock[ tx ] = lr;

   __syncthreads();

   // Perform a cumulative sum
   cumcount( lblock, rblock );

   if( tx == 0 ) {
      // Decide on max and min unsigned int
      for( int i = 0; i < THREADS; i++ ) {
         minpiv[ 0 ] = TNL::min( minpiv[ 0 ], minpiv[ i ] );
         maxpiv[ 0 ] = TNL::max( maxpiv[ 0 ], maxpiv[ i ] );
      }
   }
   __syncthreads();

   // Store each threads part of the cumulative count
   hist->left[ blockIdx.x * ( THREADS ) + threadIdx.x ] = lblock[ threadIdx.x + 1 ];
   hist->right[ blockIdx.x * ( THREADS ) + threadIdx.x ] = rblock[ threadIdx.x + 1 ];

   // Store the total sum
   lengths->left[ blockIdx.x ] = lblock[ THREADS ];
   lengths->right[ blockIdx.x ] = rblock[ THREADS ];

   // Store the max and min unsigned int
   lengths->minpiv[ blockIdx.x ] = minpiv[ 0 ];
   lengths->maxpiv[ blockIdx.x ] = maxpiv[ 0 ];
}

/**
 * Part Two - Move unsigned ints to their correct position in the auxiliary array
 * @param data   unsigned ints to be moved
 * @param data2  Destination for unsigned ints
 * @param params Specifies which data each thread block is responsible for
 * @param hist   The cumulative sum for each thread is stored here
 * @param lengths The total sum for each thread block is stored here
 */
//template <typename unsigned int>
__global__
void
part2(
   const unsigned int* data,
   unsigned int* data2,
   Params< unsigned int >* params,
   Hist* hist,
   Length< unsigned int >* lengths )
{
   const int tx = threadIdx.x;
   const int bx = blockIdx.x;

   // Each thread uses the cumulative sum to know where to write
   unsigned int x = lengths->left[ bx ] + hist->left[ bx * ( THREADS ) + tx ] - 1;  // - 1;
   unsigned int y = lengths->right[ bx ] - hist->right[ bx * ( THREADS ) + tx ];

   // Where should we read?
   unsigned int start = params[ bx ].from;
   unsigned int end = params[ bx ].end;
   unsigned int pivot = params[ bx ].pivot;

   __syncthreads();

   int coal = start & 0xf;
   start = start - coal;

   // Go through all the assigned data
   if( tx + start < end ) {
      // Reading unsigned ints...
      unsigned int d = data[ tx + start ];

      if( ! ( tx < coal ) ) {
         // and writing them to auxiliary array
         if( d < pivot ) {
            if( x > 0 )
               data2[ x-- ] = d;
            else
               data2[ x ] = d;
         }
         else if( d > pivot )
            data2[ y++ ] = d;
      }
   }

   __syncthreads();

   // Go through all the assigned data
   for( unsigned int i = start + tx + THREADS; i < end; i += THREADS ) {
      // Reading unsigned ints...
      unsigned int d = data[ i ];

      // and writing them to auxiliary array
      if( d < pivot ) {
         if( x > 0 )
            data2[ x-- ] = d;
         else
            data2[ x ] = d;
      }
      else if( d > pivot )
         data2[ y++ ] = d;
   }
}

/**
 * Part Three - Write the pivot value
 * @param data   Destination for pivot
 * @param params Specifies which data each thread block is responsible for
 * @param hist   The cumulative sum for each thread is stored here
 * @param lengths The total sum for each thread block is stored here
 */
//template <typename unsigned int>
__global__
void
part3( unsigned int* data, Params< unsigned int >* params, Hist* hist, Length< unsigned int >* lengths )
{
   const int tx = threadIdx.x;
   const int bx = blockIdx.x;

   // If we are the "last" thread block that is assigned to the same data sequence
   // we write the pivot between the left and right block
   if( params[ bx ].last ) {
      // Get destination position
      unsigned int x = lengths->left[ bx ] + hist->left[ bx * THREADS + THREADS - 1 ] + tx;
      unsigned int y = lengths->right[ bx ] - hist->right[ bx * THREADS + THREADS - 1 ];
      unsigned int pivot = params[ bx ].pivot;

      // Write the pivot values
      for( ; x < y; x += THREADS )
         data[ x ] = pivot;
   }
}

/**
 * The local quicksort - sorts a block of data with no inter-block synchronization
 * @param adata  Contains some of the blocks to be sorted and also acts as the final
 *               destination for sorted data
 * @param adata2 Contains some of the blocks to be sorted
 * @param bs     List of blocks to be sorted and a pointer telling if a specific block is
 *               in \a adata or \a adata2
 */
//template <typename unsigned int>
__global__
void
lqsort( unsigned int* adata, unsigned int* adata2, LQSortParams* bs, unsigned int phase )
{
   __shared__ unsigned int lphase;
   lphase = phase;

   // Shorthand for the threadid
   int tx = threadIdx.x;

   // Stack pointer
   __shared__ int bi;

   // Stack unsigned ints
   __shared__ unsigned int beg[ 32 ];
   __shared__ unsigned int end[ 32 ];
   __shared__ bool flip[ 32 ];

   auto* lblock = static_cast< unsigned int* >( sarray );
   auto* rblock = &lblock[ blockDim.x + 1 ];

   // The current pivot
   __shared__ unsigned int pivot;

   // The sequence to be sorted
   __shared__ unsigned int from;
   __shared__ unsigned int to;

   // Since we switch between the primary and the auxiliary buffer,
   // these variables are required to keep track on which role
   // a buffer currently has
   __shared__ unsigned int* data;
   __shared__ unsigned int* data2;
   __shared__ unsigned int sbsize;

   __shared__ unsigned int bx;
   if( threadIdx.x == 0 )
#ifdef HASATOMICS
      bx = atomicInc( &ohtotal, 50000 );
#else
      bx = blockIdx.x;
#endif

   __syncthreads();

   while( bx < gridDim.x ) {
      // Thread 0 is in charge of the stack operations
      if( tx == 0 ) {
         // We push our first block on the stack
         // This is the block given by the bs parameter
         beg[ 0 ] = bs[ bx ].beg;
         end[ 0 ] = bs[ bx ].end;
         flip[ 0 ] = bs[ bx ].flip;
         sbsize = bs[ bx ].sbsize;

         bi = 0;
      }

      __syncthreads();

      // If we were given an empty block there is no need to continue
      if( end[ 0 ] == beg[ 0 ] )
         return;

      // While there are items left on the stack to sort
      while( bi >= 0 ) {
         __syncthreads();
         // Thread 0 pops a fresh sequence from the stack
         if( tx == 0 ) {
            from = beg[ bi ];
            to = end[ bi ];

            // Check which buffer the sequence is in
            if( ! flip[ bi ] ) {
               data = adata2;
               data2 = adata;
            }
            else {
               data = adata;
               data2 = adata2;
            }
         }

         __syncthreads();

         // If the sequence is smaller than SBSIZE we sort it using
         // an alternative sort. Otherwise each thread would sort just one
         // or two unsigned ints and that wouldn't be efficient
         if( ( to - from ) < ( sbsize - 16 ) ) {
            // Sort it using bitonic sort. This could be changed to some other
            // sorting method. Store the result in the final destination buffer
            if( ( to - from >= 1 ) && ( lphase != 2 ) ) {
               // Create ArrayViews for the data to sort
               TNL::Containers::ArrayView< unsigned int, TNL::Devices::Cuda > view_data( data + from, to - from );
               TNL::Containers::ArrayView< unsigned int, TNL::Devices::Cuda > view_adata( adata + from, to - from );

               // Use TNL's bitonicSort_Block with shared memory
               // The comparator uses standard less-than for ascending sort
               TNL::Algorithms::Sorting::detail::bitonicSort_Block(
                  view_data,
                  view_adata,
                  sarray,
                  []( unsigned int a, unsigned int b )
                  {
                     return a < b;
                  } );
            }
            __syncthreads();

            // Decrement the stack pointer
            if( tx == 0 )
               bi--;
            __syncthreads();
            // and continue with the next sequence
            continue;
         }

         if( tx == 0 ) {
            // Create a new pivot for the sequence
            // Try to optimize this for your input distribution
            // if you have some information about it
            unsigned int mip = TNL::min( data[ from ], data[ to - 1 ], data[ ( from + to ) / 2 ] );
            unsigned int map = TNL::max( data[ from ], data[ to - 1 ], data[ ( from + to ) / 2 ] );
            pivot = TNL::min( TNL::max( mip / 2 + map / 2, mip ), map );
         }

         unsigned int ll = 0;
         unsigned int lr = 0;

         __syncthreads();

         unsigned int coal = from & 0xf;

         if( tx + from - coal < to ) {
            unsigned int d = data[ tx + from - coal ];

            if( ! ( tx < coal ) ) {
               // Counting unsigned ints that have a higher value than the pivot
               if( d < pivot )
                  ll++;
               else
                  // or a lower
                  if( d > pivot )
                     lr++;
            }
         }

         // Go through the current sequence
         for( int i = from + tx + THREADS - coal; i < to; i += THREADS ) {
            unsigned int d = data[ i ];

            // Counting unsigned ints that have a higher value than the pivot
            if( d < pivot )
               ll++;
            else
               // or a lower
               if( d > pivot )
                  lr++;
         }

         // Store the result in a shared array so that we can calculate a
         // cumulative sum
         lblock[ tx ] = ll;
         rblock[ tx ] = lr;

         __syncthreads();

         // Calculate the cumulative sum
         cumcount( lblock, rblock );

         __syncthreads();

         // Let thread 0 add the new resulting subsequences to the stack
         if( tx == 0 ) {
            // The sequences are in the other buffer now
            flip[ bi + 1 ] = ! flip[ bi ];
            flip[ bi ] = ! flip[ bi ];

            // We need to place the smallest object on top of the stack
            // to ensure that we don't run out of stack space
            if( lblock[ THREADS ] < rblock[ THREADS ] ) {
               beg[ bi + 1 ] = beg[ bi ];
               beg[ bi ] = to - rblock[ THREADS ];
               end[ bi + 1 ] = from + lblock[ THREADS ];
            }
            else {
               end[ bi + 1 ] = end[ bi ];
               end[ bi ] = from + lblock[ THREADS ];
               beg[ bi + 1 ] = to - rblock[ THREADS ];
            }
            // Increment the stack pointer
            bi++;
         }

         __syncthreads();

         unsigned int x = from + lblock[ tx + 1 ] - 1;
         unsigned int y = to - rblock[ tx + 1 ];

         coal = from & 0xf;

         if( tx + from - coal < to ) {
            unsigned int d = data[ tx + from - coal ];

            if( ! ( tx < coal ) ) {
               if( d < pivot ) {
                  if( x > 0 )
                     data2[ x-- ] = d;
                  else
                     data2[ x ] = d;
               }
               else if( d > pivot )
                  data2[ y++ ] = d;
            }
         }

         // Go through the data once again
         // writing it to its correct position
         for( unsigned int i = from + tx + THREADS - coal; i < to; i += THREADS ) {
            unsigned int d = data[ i ];

            if( d < pivot ) {
               if( x > 0 )
                  data2[ x-- ] = d;
               else
                  data2[ x ] = d;
            }
            else if( d > pivot )
               data2[ y++ ] = d;
         }

         __syncthreads();

         // As a final step, write the pivot value between the right and left
         // subsequence. Write it to the final destination since this pivot
         // is always correctly sorted
         for( unsigned int i = from + lblock[ THREADS ] + tx; i < to - rblock[ THREADS ]; i += THREADS ) {
            adata[ i ] = pivot;
         }

         __syncthreads();
      }
#ifdef HASATOMICS
      if( threadIdx.x == 0 )
         bx = atomicInc( &ohtotal, 50000 );
      __syncthreads();
#else
      break;
#endif
   }

   __syncthreads();
}

#undef THREADS
#define THREADS threads

/**
 * The main sort function
 * @param data		Data to be sorted - allocated on the device
 * @param size		The length of the data
 */
template< typename element >
void
GPUQSort< element >::sort(
   element* data,
   unsigned int size,
   unsigned int blockscount,
   unsigned int threads,
   unsigned int sbsize,
   unsigned int phase )
{
   if( threads == 0 || blockscount == 0 || sbsize == 0 ) {
      threads = 1 << static_cast< int >( std::round( std::log2( size * TK + TM ) ) );
      blockscount = 1 << static_cast< int >( std::round( std::log2( size * MK + MM ) ) );
      sbsize = 1 << static_cast< int >( std::round( std::log2( size * SK + SM ) ) );
   }

#ifdef HASATOMICS
   unsigned int* doh;
   unsigned int oh;

   cudaGetSymbolAddress( (void**) &doh, "ohtotal" );
   oh = 0;
   TNL_BACKEND_SAFE_CALL( cudaMemcpy( doh, &oh, 4, cudaMemcpyHostToDevice ) );
#endif

   if( threads > MAXTHREADS )
      throw std::runtime_error( "Too many threads" );

   if( blockscount > MAXBLOCKS )
      throw std::runtime_error( "Too many blocks" );

   // Create an auxiallary array
   data2 = nullptr;
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &data2, ( size ) * sizeof( element ) ) );

   // We start with a set containing only the sequence to be sorted
   // This will grow as we partition the data
   workset[ 0 ].beg = 0;
   workset[ 0 ].end = size;
   workset[ 0 ].orgbeg = 0;
   workset[ 0 ].orgend = size;
   workset[ 0 ].altered = false;
   workset[ 0 ].flip = false;

   // Get a starting pivot - copy elements to host for comparison
   element pivot_vals[ 3 ];
   TNL_BACKEND_SAFE_CALL( cudaMemcpy( pivot_vals, data, 3 * sizeof( element ), cudaMemcpyDeviceToHost ) );
   workset[ 0 ].pivot = ( TNL::min( pivot_vals[ 0 ], pivot_vals[ 1 ], pivot_vals[ 2 ] )
                          + TNL::max( pivot_vals[ 0 ], pivot_vals[ 1 ], pivot_vals[ 2 ] ) )
                      / 2;
   unsigned int worksize = 1;

   unsigned int blocks = blockscount / 2;
   if( blocks == 0 )
      blocks = 1;  // Prevent division by zero
   unsigned totsize = size;
   unsigned int maxlength = ( size / blocks ) / 4;
   if( maxlength == 0 )
      maxlength = 1;  // Prevent division by zero

   bool flip = true;

   // Partition the sequences until we have enough
   while( worksize < blocks ) {
      unsigned int ws = totsize / blocks;
      if( ws == 0 )
         ws = 1;  // Prevent division by zero
      unsigned int paramsize = 0;

      // Go through the sequences we have and divide them into sections
      // and assign thread blocks according to their size
      for( unsigned int i = 0; i < worksize; i++ ) {
         if( ( workset[ i ].end - workset[ i ].beg ) < maxlength )
            continue;

         // Larger sequences gets more thread blocks assigned to them
         unsigned int blocksassigned = TNL::max( ( workset[ i ].end - workset[ i ].beg ) / ws, 1 );
         for( unsigned int q = 0; q < blocksassigned; q++ ) {
            params[ paramsize ].from = workset[ i ].beg + ws * q;
            params[ paramsize ].end = params[ paramsize ].from + ws;
            params[ paramsize ].pivot = workset[ i ].pivot;
            params[ paramsize ].ptr = i;
            params[ paramsize ].last = false;
            paramsize++;
         }
         params[ paramsize - 1 ].last = true;
         params[ paramsize - 1 ].end = workset[ i ].end;

         workset[ i ].lmaxpiv = 0;
         workset[ i ].lminpiv = 0xffffffff;
         workset[ i ].rmaxpiv = 0;
         workset[ i ].rminpiv = 0xffffffff;
      }

      if( paramsize == 0 )
         break;

      // Copy the block assignment to the GPU
      TNL_BACKEND_SAFE_CALL( cudaMemcpy( dparams, params, paramsize * sizeof( Params< element > ), cudaMemcpyHostToDevice ) );

      // Do the cumulative sum
      if( flip ) {
         // clang-format off
         part1<<< paramsize, THREADS, ( THREADS + 1 ) * 2 * 4 + THREADS * 2 * 4 >>>( data, dparams, dhists, dlength );
         // clang-format on
      }
      else {
         // clang-format off
         part1<<< paramsize, THREADS, ( THREADS + 1 ) * 2 * 4 + THREADS * 2 * 4 >>>( data2, dparams, dhists, dlength );
         // clang-format on
      }
      TNL_BACKEND_SAFE_CALL( cudaMemcpy( length, dlength, sizeof( Length< element > ), cudaMemcpyDeviceToHost ) );

      // Do the block cumulative sum. Done on the CPU since not all cards have support for
      // atomic operations yet.
      for( unsigned int i = 0; i < paramsize; i++ ) {
         unsigned int l = length->left[ i ];
         unsigned int r = length->right[ i ];

         length->left[ i ] = workset[ params[ i ].ptr ].beg;
         length->right[ i ] = workset[ params[ i ].ptr ].end;

         workset[ params[ i ].ptr ].beg += l;
         workset[ params[ i ].ptr ].end -= r;
         workset[ params[ i ].ptr ].altered = true;

         workset[ params[ i ].ptr ].rmaxpiv = TNL::max( length->maxpiv[ i ], workset[ params[ i ].ptr ].rmaxpiv );
         workset[ params[ i ].ptr ].lminpiv = TNL::min( length->minpiv[ i ], workset[ params[ i ].ptr ].lminpiv );

         workset[ params[ i ].ptr ].lmaxpiv = TNL::min( workset[ params[ i ].ptr ].pivot, workset[ params[ i ].ptr ].rmaxpiv );
         workset[ params[ i ].ptr ].rminpiv = TNL::max( workset[ params[ i ].ptr ].pivot, workset[ params[ i ].ptr ].lminpiv );
      }

      // Copy the result of the block cumulative sum to the GPU
      TNL_BACKEND_SAFE_CALL( cudaMemcpy( dlength, length, sizeof( Length< element > ), cudaMemcpyHostToDevice ) );

      // Move the elements to their correct position
      if( flip ) {
         // clang-format off
         part2<<< paramsize, THREADS >>>( data, data2, dparams, dhists, dlength );
         // clang-format on
      }
      else {
         // clang-format off
         part2<<< paramsize, THREADS >>>( data2, data, dparams, dhists, dlength );
         // clang-format on
      }

      // Fill in the pivot value between the left and right blocks
      // clang-format off
      part3<<< paramsize, THREADS >>>( data, dparams, dhists, dlength );
      // clang-format on

      flip = ! flip;

      // Add the sequences resulting from the partitioning
      // to set
      unsigned int oldworksize = worksize;
      totsize = 0;
      for( unsigned int i = 0; i < oldworksize; i++ ) {
         if( workset[ i ].altered ) {
            if( workset[ i ].beg - workset[ i ].orgbeg >= maxlength )
               totsize += workset[ i ].beg - workset[ i ].orgbeg;
            if( workset[ i ].orgend - workset[ i ].end >= maxlength )
               totsize += workset[ i ].orgend - workset[ i ].end;

            workset[ worksize ].beg = workset[ worksize ].orgbeg = workset[ i ].orgbeg;
            workset[ worksize ].end = workset[ worksize ].orgend = workset[ i ].beg;
            workset[ worksize ].flip = flip;
            workset[ worksize ].altered = false;
            workset[ worksize ].pivot = ( workset[ i ].lminpiv / 2 + workset[ i ].lmaxpiv / 2 );

            worksize++;

            workset[ i ].orgbeg = workset[ i ].beg = workset[ i ].end;
            workset[ i ].end = workset[ i ].orgend;
            workset[ i ].flip = flip;
            workset[ i ].pivot = ( workset[ i ].rminpiv / 2 + workset[ i ].rmaxpiv / 2 );
            workset[ i ].altered = false;
         }
      }
   }

   // Due to the poor scheduler on some graphics card
   // we need to sort the order in which the blocks
   // are sorted to avoid poor scheduling decisions
   unsigned int sortblocks[ MAXBLOCKS * 2 ];
   for( unsigned int i = 0; i < worksize; i++ )
      sortblocks[ i ] =
         ( ( workset[ i ].end - workset[ i ].beg ) << static_cast< int >( std::round( std::log2( MAXBLOCKS * 4 ) ) ) ) + i;
   std::sort( &sortblocks[ 0 ], &sortblocks[ worksize ] );

   if( worksize != 0 ) {
      // Copy the block assignments to the GPU
      for( unsigned int i = 0; i < worksize; i++ ) {
         unsigned int q = ( worksize - 1 ) - ( sortblocks[ i ] & ( MAXBLOCKS * 4 - 1 ) );

         lqparams[ i ].beg = workset[ q ].beg;
         lqparams[ i ].end = workset[ q ].end;
         lqparams[ i ].flip = workset[ q ].flip;
         lqparams[ i ].sbsize = sbsize;
      }

      TNL_BACKEND_SAFE_CALL( cudaMemcpy( dlqparams, lqparams, worksize * sizeof( LQSortParams ), cudaMemcpyHostToDevice ) );

      // Run the local quicksort, the one that doesn't need inter-block synchronization
      if( phase != 1 ) {
         // clang-format off
         lqsort<<< worksize, THREADS, TNL::max( ( THREADS + 1 ) * 2 * 4, sbsize * 4 ) >>>( data, data2, dlqparams, phase );
         // clang-format on
      }
   }

   TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );

   cudaFree( data2 );
}

template< typename element >
GPUQSort< element >::GPUQSort()
{
   // Example for GeForce 8800 GTX:
   // TK = 1.17125033316e-005f;
   // TM = 52.855721393f;
   // MK = 3.7480010661e-005f;
   // MM = 476.338308458f;
   // SK = 4.68500133262e-005f;
   // SM = 211.422885572f;
   TK = 0;
   TM = 128;
   MK = 0;
   MM = 512;
   SK = 0;
   SM = 512;

   TNL_BACKEND_SAFE_CALL( cudaMallocHost( (void**) &workset, MAXBLOCKS * 2 * sizeof( BlockSize< element > ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMallocHost( (void**) &params, MAXBLOCKS * sizeof( Params< element > ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMallocHost( (void**) &length, sizeof( Length< element > ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMallocHost( (void**) &lqparams, MAXBLOCKS * sizeof( LQSortParams ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &dlqparams, MAXBLOCKS * sizeof( LQSortParams ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &dhists, sizeof( Hist ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &dlength, sizeof( Length< element > ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &dparams, MAXBLOCKS * sizeof( Params< element > ) ) );
}

template< typename element >
GPUQSort< element >::~GPUQSort()
{
   cudaFreeHost( workset );
   cudaFreeHost( params );
   cudaFreeHost( length );
   cudaFreeHost( lqparams );
   cudaFree( dparams );
   cudaFree( dlqparams );
   cudaFree( dhists );
   cudaFree( dlength );
}

void
gpuqsort(
   unsigned int* data,
   unsigned int size,
   unsigned int blockscount,
   unsigned int threads,
   unsigned int sbsize,
   unsigned int phase )
{
   GPUQSort< unsigned int > s;
   s.sort( data, size, blockscount, threads, sbsize, phase );
}

struct CedermanQuicksort
{
   static void
   sort( TNL::Containers::ArrayView< int, TNL::Devices::Cuda >& array )
   {
      gpuqsort( (unsigned int*) array.getData(), (unsigned int) array.getSize() );
   }
};
