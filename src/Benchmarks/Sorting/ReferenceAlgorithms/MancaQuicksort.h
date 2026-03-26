#pragma once

#include <limits>
#include <stdexcept>

#include <TNL/Atomic.h>
#include <TNL/Backend.h>
#include <TNL/Containers/ArrayView.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/detail/CudaScanKernel.h>

//defines the shared memory size
#define SHARED_LIMIT 1024

#define LOG2_WARP_SIZE 5U
#define WARP_SIZE ( 1U << LOG2_WARP_SIZE )

/*
 * division of the vector to be sorted in buckets
 * the attributes of the object Block are the parameters of each bucket
 */
template< typename Type >
struct Block
{
   unsigned int begin;
   unsigned int end;

   unsigned int nextbegin;
   unsigned int nextend;

   Type pivot;

   //max of the bucket items
   Type maxPiv;
   //min of the bucket items
   Type minPiv;
   //done indicates that a bucket has been analyzed
   short done;
   short select;
};

template< typename Type >
struct Partition
{
   unsigned int ibucket;
   unsigned int from;
   unsigned int end;
   Type pivot;
};

using uint = unsigned int;

template< typename Type >
inline __device__
void
warpCompareInclusive( Type& idata, Type& idata2, volatile Type* s_Data, uint size )
{
   volatile Type* s_Data2;
   s_Data2 = s_Data + blockDim.x * 2;
   uint pos = 2 * threadIdx.x - ( threadIdx.x & ( size - 1 ) );
   s_Data[ pos ] = 0;
   s_Data2[ pos ] = 0;
   pos += size;
   s_Data[ pos ] = idata;
   s_Data2[ pos ] = idata2;

   for( uint offset = 1; offset < size; offset <<= 1 ) {
      s_Data[ pos ] = max( s_Data[ pos ], s_Data[ pos - offset ] );
      s_Data2[ pos ] = min( s_Data2[ pos ], s_Data2[ pos - offset ] );
   }

   idata = s_Data[ pos ];
   idata2 = s_Data2[ pos ];
}

template< typename Type >
inline __device__
void
compareInclusive( Type& idata, Type& idata2, volatile Type* s_Data, uint size )
{
   volatile Type* s_Data2;
   s_Data2 = s_Data + blockDim.x * 2;
   //Bottom-level inclusive warp scan
   warpCompareInclusive( idata, idata2, s_Data, WARP_SIZE );

   //Save top Types of each warp for exclusive warp scan
   //sync to wait for warp scans to complete (because s_Data is being overwritten)
   __syncthreads();
   if( ( threadIdx.x & ( WARP_SIZE - 1 ) ) == ( WARP_SIZE - 1 ) ) {
      s_Data[ threadIdx.x >> LOG2_WARP_SIZE ] = idata;
      s_Data2[ threadIdx.x >> LOG2_WARP_SIZE ] = idata2;
   }

   //wait for warp scans to complete
   __syncthreads();
   if( threadIdx.x < ( blockDim.x / WARP_SIZE ) ) {
      //grab top warp Types
      Type val = s_Data[ threadIdx.x ];
      Type val2 = s_Data2[ threadIdx.x ];
      //calculate exclsive scan and write back to shared memory
      warpCompareInclusive( val, val2, s_Data, size >> LOG2_WARP_SIZE );
      s_Data[ threadIdx.x ] = val;
      s_Data2[ threadIdx.x ] = val2;
   }

   //return updated warp scans with exclusive scan results
   __syncthreads();
   idata = max( idata, s_Data[ threadIdx.x >> LOG2_WARP_SIZE ] );
   idata2 = min( idata2, s_Data2[ threadIdx.x >> LOG2_WARP_SIZE ] );
}

template< typename Type >
__device__
inline void
Comparator( Type& valA, Type& valB, uint dir )
{
   if( ( valA > valB ) == dir )
      TNL::swap( valA, valB );
}

static __device__
__forceinline__ unsigned int
__qsflo( unsigned int word )
{
   unsigned int ret;
   asm volatile( "bfind.u32 %0, %1;" : "=r"( ret ) : "r"( word ) );
   return ret;
}

// Helper function to get the next representable value after x
// For integers: x + 1
// For floating-point: next value towards +infinity
template< typename Type >
__device__
inline Type
nextValue( Type x )
{
   if constexpr( std::is_floating_point_v< Type > )
      return std::nextafter( x, std::numeric_limits< Type >::infinity() );
   else
      return x + Type( 1 );
}

template< typename Type >
__global__
void
globalBitonicSort( Type* indata, Type* outdata, Block< Type >* bucket, bool inputSelect )
{
   __shared__ Type shared[ 1024 ];

   Type* data;

   Block< Type > cord = bucket[ blockIdx.x ];

   uint size = cord.end - cord.begin;
   bool select = ! ( cord.select );

   if( cord.end - cord.begin > 1024 || cord.end - cord.begin == 0 )
      return;

   unsigned int bitonicSize = 1U << ( __qsflo( size - 1U ) + 1U );

   if( select )
      data = indata;
   else
      data = outdata;

   //__syncthreads();

   for( int i = threadIdx.x; i < size; i += blockDim.x )
      shared[ i ] = data[ i + cord.begin ];

   for( int i = threadIdx.x + size; i < bitonicSize; i += blockDim.x )
      shared[ i ] = std::numeric_limits< Type >::max();

   __syncthreads();

   for( uint size = 2; size < bitonicSize; size <<= 1 ) {
      //Bitonic merge
      uint ddd = 1 ^ ( ( threadIdx.x & ( size / 2 ) ) != 0 );
      for( uint stride = size / 2; stride > 0; stride >>= 1 ) {
         __syncthreads();
         uint pos = 2 * threadIdx.x - ( threadIdx.x & ( stride - 1 ) );
         //if(pos <bitonicSize){
         Comparator( shared[ pos + 0 ], shared[ pos + stride ], ddd );
         // }
      }
   }

   //ddd == dir for the last bitonic merge step

   for( uint stride = bitonicSize / 2; stride > 0; stride >>= 1 ) {
      __syncthreads();
      uint pos = 2 * threadIdx.x - ( threadIdx.x & ( stride - 1 ) );
      // if(pos <bitonicSize){
      Comparator( shared[ pos + 0 ], shared[ pos + stride ], 1 );
      // }
   }

   __syncthreads();

   // Write back the sorted data to its correct position
   for( int i = threadIdx.x; i < size; i += blockDim.x )
      indata[ i + cord.begin ] = shared[ i ];
}

template< int blockSize, typename Type >
__global__
void
quick( Type* indata, Type* buffer, Partition< Type >* partition, Block< Type >* bucket )
{
   using namespace TNL::Algorithms::detail;
   struct Uint2Plus
   {
      __device__
      uint2
      operator()( uint2 a, uint2 b ) const
      {
         uint2 result;
         result.x = a.x + b.x;
         result.y = a.y + b.y;
         return result;
      }
   };
   using BlockScan = CudaBlockScan< ScanType::Inclusive, blockSize, Uint2Plus, uint2 >;
   // storage to be allocated in shared memory
   union Shared
   {
      Type data[ 1024 ];
      typename BlockScan::Storage blockScanStorage;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__
      Shared() {}
   };
   __shared__ Shared storage;

   __shared__ uint start1, end1;
   __shared__ uint left, right;

   int tix = threadIdx.x;

   uint start = partition[ blockIdx.x ].from;
   uint end = partition[ blockIdx.x ].end;
   Type pivot = partition[ blockIdx.x ].pivot;
   uint nseq = partition[ blockIdx.x ].ibucket;

   uint lo = 0;
   uint hi = 0;

   Type lmin = std::numeric_limits< Type >::max();
   Type rmax = std::numeric_limits< Type >::lowest();

   Type d;

   // start read on 1° tile and store the coordinates of the items that must
   // be moved on the left or on the right of the pivot

   if( tix + start < end ) {
      d = indata[ tix + start ];

      //count items smaller or bigger than the pivot
      // if d<pivot then ll++ else ll
      lo = ( d < pivot ) * ( lo + 1 ) + ( d >= pivot ) * lo;
      // if d>pivot then lr++ else lr
      hi = ( d <= pivot ) * ( hi ) + ( d > pivot ) * ( hi + 1 );

      lmin = d;
      rmax = d;
   }

   //read and store the coordinates on next tiles for each block
   for( uint i = tix + start + blockDim.x; i < end; i += blockDim.x ) {
      Type d = indata[ i ];

      //count items smaller or bigger than the pivot
      lo = ( d < pivot ) * ( lo + 1 ) + ( d >= pivot ) * lo;
      hi = ( d <= pivot ) * ( hi ) + ( d > pivot ) * ( hi + 1 );

      //compute max and min of tile items
      lmin = min( lmin, d );
      rmax = max( rmax, d );
   }

   //compute max and min of every partition
   compareInclusive( rmax, lmin, storage.data, blockDim.x );

   __syncthreads();

   if( tix == blockDim.x - 1 ) {
      //compute absolute max and min for the bucket
      atomicMax( &bucket[ nseq ].maxPiv, rmax );
      atomicMin( &bucket[ nseq ].minPiv, lmin );
   }

   __syncthreads();

   // calculate the coordinates of its assigned item to each thread,
   // which are necessary to known in which subsequences the item must be copied
   uint2 _result = BlockScan::scan( Uint2Plus{}, uint2{ 0, 0 }, uint2{ lo, hi }, threadIdx.x, storage.blockScanStorage );
   lo = _result.x;
   hi = _result.y;

   lo = lo - 1;
   hi = SHARED_LIMIT - hi;

   if( tix == blockDim.x - 1 ) {
      left = lo + 1;
      right = SHARED_LIMIT - hi;

      start1 = atomicAdd( &bucket[ nseq ].nextbegin, left );
      end1 = atomicSub( &bucket[ nseq ].nextend, right );
   }

   __syncthreads();

   //thread blocks write on the shared memory the items smaller and bigger than the first tile's pivot
   if( tix + start < end ) {
      //items smaller than pivot
      if( d < pivot ) {
         storage.data[ lo ] = d;
         lo--;
      }

      //items bigger than pivot
      if( d > pivot ) {
         storage.data[ hi ] = d;
         hi++;
      }
   }

   //thread blocks write on the shared memory the items smaller and bigger than next tiles' pivot
   for( uint i = start + tix + blockDim.x; i < end; i += blockDim.x ) {
      Type d = indata[ i ];
      //items smaller than the pivot
      if( d < pivot ) {
         storage.data[ lo ] = d;
         lo--;
      }

      //items bigger than the pivot
      if( d > pivot ) {
         storage.data[ hi ] = d;
         hi++;
      }
   }

   __syncthreads();

   //items smaller and bigger than the pivot already sorted in the shared memory are coalesced written on the global memory
   //partial results of each thread block stored on the shared memory are merged together in two subsequences within the global
   //memory coalesced writing of next tiles on the global memory
   for( uint i = tix; i < SHARED_LIMIT; i += blockDim.x ) {
      if( i < left )
         buffer[ start1 + i ] = storage.data[ i ];

      if( i >= SHARED_LIMIT - right )
         buffer[ end1 + i - SHARED_LIMIT ] = storage.data[ i ];
   }
}

//this function assigns the attributes to each partition of each bucket
//a thread block is assigned to a specific partition
template< typename Type >
__global__
void
partitionAssign( Block< Type >* bucket, const uint* npartitions, Partition< Type >* partition )
{
   int tx = threadIdx.x;
   int bx = blockIdx.x;

   uint beg = bucket[ bx ].nextbegin;
   uint end = bucket[ bx ].nextend;
   Type pivot = bucket[ bx ].pivot;

   uint from;
   uint to;

   if( bx > 0 )
      from = npartitions[ bx - 1 ];
   else
      from = 0;
   to = npartitions[ bx ];

   uint i = tx + from;

   if( i < to ) {
      uint begin = beg + SHARED_LIMIT * tx;
      partition[ i ].from = begin;
      partition[ i ].end = begin + SHARED_LIMIT;
      partition[ i ].pivot = pivot;
      partition[ i ].ibucket = bx;
   }

   for( uint i = tx + from + blockDim.x; i < to; i += blockDim.x ) {
      uint begin = beg + SHARED_LIMIT * ( i - from );
      partition[ i ].from = begin;
      partition[ i ].end = begin + SHARED_LIMIT;
      partition[ i ].pivot = pivot;
      partition[ i ].ibucket = bx;
   }
   __syncthreads();
   if( tx == 0 && to - from > 0 )
      partition[ to - 1 ].end = end;
}

//this function enters the pivot value in the central bucket's items
template< typename Type >
__global__
void
insertPivot( Type* data, Block< Type >* bucket, int nbucket )
{
   Type pivot = bucket[ blockIdx.x ].pivot;
   uint start = bucket[ blockIdx.x ].nextbegin;
   uint end = bucket[ blockIdx.x ].nextend;
   bool is_altered = bucket[ blockIdx.x ].done;

   if( is_altered && blockIdx.x < nbucket )
      for( uint j = start + threadIdx.x; j < end; j += blockDim.x )
         data[ j ] = pivot;
}

//this function assigns the new attributes of each bucket
template< typename Type >
__global__
void
bucketAssign( Block< Type >* bucket, uint* npartitions, int nbucket, int select )
{
   uint i = blockIdx.x * blockDim.x + threadIdx.x;

   if( i < nbucket ) {
      bool is_altered = bucket[ i ].done;
      if( is_altered ) {
         //read on i node
         uint orgbeg = bucket[ i ].begin;
         uint from = bucket[ i ].nextbegin;
         uint orgend = bucket[ i ].end;
         uint end = bucket[ i ].nextend;
         Type pivot = bucket[ i ].pivot;
         Type minPiv = bucket[ i ].minPiv;
         Type maxPiv = bucket[ i ].maxPiv;

         //compare each bucket's max and min to the pivot
         Type lmaxpiv = min( pivot, maxPiv );
         Type rminpiv = max( pivot, minPiv );

         //write on i+nbucket node
         bucket[ i + nbucket ].begin = orgbeg;
         bucket[ i + nbucket ].nextbegin = orgbeg;
         bucket[ i + nbucket ].nextend = from;
         bucket[ i + nbucket ].end = from;
         // Use safe average to avoid overflow for large unsigned values
         bucket[ i + nbucket ].pivot = minPiv / 2 + lmaxpiv / 2;

         //if(select)
         //	bucket[i+nbucket].done   = (from-orgbeg)>1024;// && (minPiv!=maxPiv);
         //else
         bucket[ i + nbucket ].done = ( from - orgbeg ) > 1024 && ( minPiv != maxPiv );
         bucket[ i + nbucket ].select = select;
         bucket[ i + nbucket ].minPiv = std::numeric_limits< Type >::max();
         bucket[ i + nbucket ].maxPiv = std::numeric_limits< Type >::lowest();
         //bucket[i+nbucket].finish=false;

         //calculate the number of partitions (npartitions) necessary to the i+nbucket bucket
         if( ! bucket[ i + nbucket ].done )
            npartitions[ i + nbucket ] = 0;
         else
            npartitions[ i + nbucket ] = ( from - orgbeg + SHARED_LIMIT - 1 ) / SHARED_LIMIT;

         //write on i node
         bucket[ i ].begin = end;
         bucket[ i ].nextbegin = end;
         bucket[ i ].nextend = orgend;
         // Use safe average to avoid overflow for large unsigned values
         // For floating-point, use nextValue to get the next representable value after the average
         bucket[ i ].pivot = nextValue( rminpiv / 2 + maxPiv / 2 );

         //if(select)
         //bucket[i].done   = (orgend-end)>1024;// && (minPiv!=maxPiv);
         //	else
         bucket[ i ].done = ( orgend - end ) > 1024 && ( minPiv != maxPiv );
         bucket[ i ].select = select;
         bucket[ i ].minPiv = std::numeric_limits< Type >::max();
         bucket[ i ].maxPiv = std::numeric_limits< Type >::lowest();
         //bucket[i].finish=false;

         //calculate the number of partitions (npartitions) necessary to the i-bucket bucket
         if( ! bucket[ i ].done )
            npartitions[ i ] = 0;
         else
            npartitions[ i ] = ( orgend - end + SHARED_LIMIT - 1 ) / SHARED_LIMIT;
      }
   }
}

template< typename Type >
__global__
void
init( Type* data, Block< Type >* bucket, uint* npartitions, int size, int nblocks )
{
   uint i = blockIdx.x * blockDim.x + threadIdx.x;

   if( i < nblocks ) {
      bucket[ i ].nextbegin = 0;
      bucket[ i ].begin = 0;

      bucket[ i ].nextend = 0 + size * ( i == 0 );
      bucket[ i ].end = 0 + size * ( i == 0 );
      npartitions[ i ] = 0;
      bucket[ i ].done = false + i == 0;
      bucket[ i ].select = false;
      bucket[ i ].minPiv = std::numeric_limits< Type >::max();
      bucket[ i ].maxPiv = std::numeric_limits< Type >::lowest();
      //bucket[i].pivot  = 0+ (i==0)*((min(min(data[0],data[size/2]),data[size-1]) +
      //max(max(data[0],data[size/2]),data[size-1]))/2);
      bucket[ i ].pivot = data[ size / 2 ];
   }
}

template< int blockSize, typename Type >
void
manca_qsort( Type* ddata, uint size )
{
   dim3 cudaBlocks;

   uint blocks = ( size + SHARED_LIMIT - 1 ) / SHARED_LIMIT;
   uint nblock = 10 * blocks;
   uint nblock_max = TNL::max( nblock, SHARED_LIMIT );
   uint partition_max = nblock;  //TNL::min( 262144, size );

   // Allocate CUDA arrays
   TNL::Containers::Array< Type, TNL::Devices::Cuda, uint > dbuffer( size );
   TNL::Containers::Array< Block< Type >, TNL::Devices::Cuda, uint > dbucket( partition_max );
   TNL::Containers::Array< Partition< Type >, TNL::Devices::Cuda, uint > partition( nblock_max );
   TNL::Containers::Array< uint, TNL::Devices::Cuda, uint > npartitions1( partition_max );
   TNL::Containers::Array< uint, TNL::Devices::Cuda, uint > npartitions2( partition_max );
   npartitions1.setValue( 0 );

   //setting GPU Cache
   cudaFuncSetCacheConfig( init< Type >, cudaFuncCachePreferL1 );
   cudaFuncSetCacheConfig( insertPivot< Type >, cudaFuncCachePreferL1 );
   cudaFuncSetCacheConfig( bucketAssign< Type >, cudaFuncCachePreferL1 );
   cudaFuncSetCacheConfig( partitionAssign< Type >, cudaFuncCachePreferL1 );
   cudaFuncSetCacheConfig( quick< blockSize, Type >, cudaFuncCachePreferShared );
   cudaFuncSetCacheConfig( globalBitonicSort< Type >, cudaFuncCachePreferShared );

   TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );

   // Initialize the bucket array: initial attributes for each bucket
   // clang-format off
   init<Type><<<(nblock + 255) / 256, 256>>>(ddata, dbucket.getData(), npartitions1.getData(), size, partition_max);
   // clang-format on
   TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );

   uint nbucket = 1;
   uint numIterations = 0;
   bool inputSelect = true;

   cudaBlocks.x = blocks;
   TNL_BACKEND_SAFE_CALL( cudaMemcpy( npartitions2.getData(), &cudaBlocks.x, sizeof( uint ), cudaMemcpyHostToDevice ) );

   // beginning of the first phase
   // this phase goes on until the size of the buckets is comparable to the SHARED_LIMIT size
   while( true ) {
      /*
       *       	---------------------    Pre-processing: Partitioning    ---------------------
       *
       * buckets are further divided in partitions based on their size
       * the number of partitions needed for each subsequence is determined by the number of elements which can be
       * processed by each thread block.
       *
       * the number of partitions (npartitions) for each block will depend on the shared memory size (SHARED_LIMIT)
       */
      if( numIterations > 0 ) {
         // Use TNL's inclusive scan (handles all sizes correctly)
         auto view1 = npartitions1.getView( 0, nbucket );
         auto view2 = npartitions2.getView( 0, nbucket );
         TNL::Algorithms::inclusiveScan( view1, view2 );
         TNL_BACKEND_SAFE_CALL(
            cudaMemcpy( &cudaBlocks.x, npartitions2.getData() + nbucket - 1, sizeof( uint ), cudaMemcpyDeviceToHost ) );
      }

      if( cudaBlocks.x == 0 )
         break;

      /*
       *  ---------------------     step 1    ---------------------
       *
       * 	A thread block is assigned to each different partition
       * 	each partition is assigned coordinates, pivot and ....
       */
      // clang-format off
      partitionAssign<Type><<<nbucket, 1024>>>(dbucket.getData(), npartitions2.getData(), partition.getData());
      // clang-format on
      TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );

      /*
       *  ---------------------    step 2a    ---------------------
       *
       *    In this function each thread block creates two subsequences
       *    to divide the items in the partition whose value is lower than
       *    the pivot value, from the items whose value is higher than the pivot value
       */
      if( inputSelect ) {
         // clang-format off
         quick<blockSize, Type><<<cudaBlocks, blockSize>>>(ddata, dbuffer.getData(), partition.getData(), dbucket.getData());
         // clang-format on
      }
      else {
         // clang-format off
         quick<blockSize, Type><<<cudaBlocks, blockSize>>>(dbuffer.getData(), ddata, partition.getData(), dbucket.getData());
         // clang-format on
      }
      TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );

      //step 2b: this function enters the pivot value in the central bucket's items
      // clang-format off
      insertPivot<Type><<<nbucket, 512>>>(ddata, dbucket.getData(), nbucket);
      // clang-format on
      TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );

      //step 3: parameters are assigned, linked to the two new buckets created in step 2
      // clang-format off
      bucketAssign<Type><<<(nbucket + 255) / 256, 256>>>(dbucket.getData(), npartitions1.getData(), nbucket, inputSelect);
      // clang-format on
      TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );

      nbucket *= 2;

      inputSelect = ! inputSelect;
      numIterations++;

      // bucketAssign writes to bucket[i + nbucket], so we need nbucket <= partition_max / 2
      if( nbucket > partition_max / 2 )
         break;
   }

   /*
    * start second phase:
    * now the size of the buckets is such that they can be entirely processed by a thread block
    */
   if( nbucket > TNL::Backend::getMaxGridXSize() ) {
      throw std::runtime_error(
         "MancaQuicksort can't terminate sorting as the block threads needed to finish it are more than the Maximum "
         "x-dimension of GPU thread blocks." );
   }
   else {
      // clang-format off
      globalBitonicSort<Type><<<nbucket, 512, 0>>>(ddata, dbuffer.getData(), dbucket.getData(), inputSelect);
      // clang-format on
      TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );
   }
}

struct MancaQuicksort
{
   template< typename Array >
   static void
   sort( Array& array )
   {
      using DeviceType = typename Array::DeviceType;

      static_assert( std::is_same_v< DeviceType, TNL::Devices::Cuda >, "MancaQuicksort requires Devices::Cuda" );

      if( array.getSize() <= 1 )
         return;

      auto view = array.getView();
      manca_qsort< 256 >( view.getData(), view.getSize() );
   }
};
