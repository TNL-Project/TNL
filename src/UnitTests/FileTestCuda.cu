#include <TNL/Allocators/Cuda.h>
#include <TNL/Backend/Macros.h>
#include <TNL/File.h>

#include <gtest/gtest.h>

using namespace TNL;

static const char* TEST_FILE_NAME = "test_FileTestCuda.tnl";

TEST( FileTestCuda, WriteAndReadCUDA )
{
   int intData( 5 );
   float floatData[ 3 ] = { 1.0, 2.0, 3.0 };
   const double constDoubleData = 3.14;

   int* cudaIntData;
   float* cudaFloatData;
   const double* cudaConstDoubleData;
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &cudaIntData, sizeof( int ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &cudaFloatData, 3 * sizeof( float ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &cudaConstDoubleData, sizeof( double ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMemcpy( cudaIntData, &intData, sizeof( int ), cudaMemcpyHostToDevice ) );
   TNL_BACKEND_SAFE_CALL( cudaMemcpy( cudaFloatData, floatData, 3 * sizeof( float ), cudaMemcpyHostToDevice ) );
   TNL_BACKEND_SAFE_CALL(
      cudaMemcpy( (void*) cudaConstDoubleData, &constDoubleData, sizeof( double ), cudaMemcpyHostToDevice ) );

   File file;
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::out ) );

   file.save< int, int, Allocators::Cuda< int > >( cudaIntData );
   file.save< float, float, Allocators::Cuda< float > >( cudaFloatData, 3 );
   file.save< const double, double, Allocators::Cuda< const double > >( cudaConstDoubleData );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   int newIntData;
   float newFloatData[ 3 ];
   double newDoubleData;
   int* newCudaIntData;
   float* newCudaFloatData;
   double* newCudaDoubleData;
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &newCudaIntData, sizeof( int ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &newCudaFloatData, 3 * sizeof( float ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &newCudaDoubleData, sizeof( double ) ) );

   file.load< int, int, Allocators::Cuda< int > >( newCudaIntData, 1 );
   file.load< float, float, Allocators::Cuda< float > >( newCudaFloatData, 3 );
   file.load< double, double, Allocators::Cuda< double > >( newCudaDoubleData, 1 );
   ASSERT_NO_THROW( file.close() );

   TNL_BACKEND_SAFE_CALL( cudaMemcpy( &newIntData, newCudaIntData, sizeof( int ), cudaMemcpyDeviceToHost ) );
   TNL_BACKEND_SAFE_CALL( cudaMemcpy( newFloatData, newCudaFloatData, 3 * sizeof( float ), cudaMemcpyDeviceToHost ) );
   TNL_BACKEND_SAFE_CALL( cudaMemcpy( &newDoubleData, newCudaDoubleData, sizeof( double ), cudaMemcpyDeviceToHost ) );

   EXPECT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i++ )
      EXPECT_EQ( newFloatData[ i ], floatData[ i ] );
   EXPECT_EQ( newDoubleData, constDoubleData );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

TEST( FileTestCuda, WriteAndReadCUDAWithConversion )
{
   const double constDoubleData[ 3 ] = { 3.1415926535897932384626433,
                                         2.7182818284590452353602874,
                                         1.6180339887498948482045868 };
   float floatData[ 3 ];
   int intData[ 3 ];

   int* cudaIntData;
   float* cudaFloatData;
   const double* cudaConstDoubleData;
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &cudaIntData, 3 * sizeof( int ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &cudaFloatData, 3 * sizeof( float ) ) );
   TNL_BACKEND_SAFE_CALL( cudaMalloc( (void**) &cudaConstDoubleData, 3 * sizeof( double ) ) );
   TNL_BACKEND_SAFE_CALL(
      cudaMemcpy( (void*) cudaConstDoubleData, &constDoubleData, 3 * sizeof( double ), cudaMemcpyHostToDevice ) );

   File file;
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::out | std::ios_base::trunc ) );
   file.save< double, float, Allocators::Cuda< double > >( cudaConstDoubleData, 3 );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   file.load< float, float, Allocators::Cuda< float > >( cudaFloatData, 3 );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   file.load< int, float, Allocators::Cuda< int > >( cudaIntData, 3 );
   ASSERT_NO_THROW( file.close() );

   TNL_BACKEND_SAFE_CALL( cudaMemcpy( floatData, cudaFloatData, 3 * sizeof( float ), cudaMemcpyDeviceToHost ) );
   TNL_BACKEND_SAFE_CALL( cudaMemcpy( &intData, cudaIntData, 3 * sizeof( int ), cudaMemcpyDeviceToHost ) );

   EXPECT_NEAR( floatData[ 0 ], 3.14159, 0.0001 );
   EXPECT_NEAR( floatData[ 1 ], 2.71828, 0.0001 );
   EXPECT_NEAR( floatData[ 2 ], 1.61803, 0.0001 );

   EXPECT_EQ( intData[ 0 ], 3 );
   EXPECT_EQ( intData[ 1 ], 2 );
   EXPECT_EQ( intData[ 2 ], 1 );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

#include "main.h"
