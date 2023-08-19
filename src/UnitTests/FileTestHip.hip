#include <TNL/Allocators/Hip.h>
#include <TNL/Backend/Macros.h>
#include <TNL/File.h>

#include <gtest/gtest.h>

using namespace TNL;

static const char* TEST_FILE_NAME = "test_FileTestHip.tnl";

TEST( FileTestHip, WriteAndReadHIP )
{
   int intData( 5 );
   float floatData[ 3 ] = { 1.0, 2.0, 3.0 };
   const double constDoubleData = 3.14;

   int* hipIntData;
   float* hipFloatData;
   const double* hipConstDoubleData;
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &hipIntData, sizeof( int ) ) );
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &hipFloatData, 3 * sizeof( float ) ) );
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &hipConstDoubleData, sizeof( double ) ) );
   TNL_BACKEND_SAFE_CALL( hipMemcpy( hipIntData, &intData, sizeof( int ), hipMemcpyHostToDevice ) );
   TNL_BACKEND_SAFE_CALL( hipMemcpy( hipFloatData, floatData, 3 * sizeof( float ), hipMemcpyHostToDevice ) );
   TNL_BACKEND_SAFE_CALL( hipMemcpy( (void*) hipConstDoubleData, &constDoubleData, sizeof( double ), hipMemcpyHostToDevice ) );

   File file;
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::out ) );

   file.save< int, int, Allocators::Hip< int > >( hipIntData );
   file.save< float, float, Allocators::Hip< float > >( hipFloatData, 3 );
   file.save< const double, double, Allocators::Hip< const double > >( hipConstDoubleData );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   int newIntData;
   float newFloatData[ 3 ];
   double newDoubleData;
   int* newHipIntData;
   float* newHipFloatData;
   double* newHipDoubleData;
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &newHipIntData, sizeof( int ) ) );
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &newHipFloatData, 3 * sizeof( float ) ) );
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &newHipDoubleData, sizeof( double ) ) );

   file.load< int, int, Allocators::Hip< int > >( newHipIntData, 1 );
   file.load< float, float, Allocators::Hip< float > >( newHipFloatData, 3 );
   file.load< double, double, Allocators::Hip< double > >( newHipDoubleData, 1 );
   ASSERT_NO_THROW( file.close() );

   TNL_BACKEND_SAFE_CALL( hipMemcpy( &newIntData, newHipIntData, sizeof( int ), hipMemcpyDeviceToHost ) );
   TNL_BACKEND_SAFE_CALL( hipMemcpy( newFloatData, newHipFloatData, 3 * sizeof( float ), hipMemcpyDeviceToHost ) );
   TNL_BACKEND_SAFE_CALL( hipMemcpy( &newDoubleData, newHipDoubleData, sizeof( double ), hipMemcpyDeviceToHost ) );

   EXPECT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i++ )
      EXPECT_EQ( newFloatData[ i ], floatData[ i ] );
   EXPECT_EQ( newDoubleData, constDoubleData );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

TEST( FileTestHip, WriteAndReadHIPWithConversion )
{
   const double constDoubleData[ 3 ] = { 3.1415926535897932384626433,
                                         2.7182818284590452353602874,
                                         1.6180339887498948482045868 };
   float floatData[ 3 ];
   int intData[ 3 ];

   int* hipIntData;
   float* hipFloatData;
   const double* hipConstDoubleData;
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &hipIntData, 3 * sizeof( int ) ) );
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &hipFloatData, 3 * sizeof( float ) ) );
   TNL_BACKEND_SAFE_CALL( hipMalloc( (void**) &hipConstDoubleData, 3 * sizeof( double ) ) );
   TNL_BACKEND_SAFE_CALL(
      hipMemcpy( (void*) hipConstDoubleData, &constDoubleData, 3 * sizeof( double ), hipMemcpyHostToDevice ) );

   File file;
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::out | std::ios_base::trunc ) );
   file.save< double, float, Allocators::Hip< double > >( hipConstDoubleData, 3 );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   file.load< float, float, Allocators::Hip< float > >( hipFloatData, 3 );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   file.load< int, float, Allocators::Hip< int > >( hipIntData, 3 );
   ASSERT_NO_THROW( file.close() );

   TNL_BACKEND_SAFE_CALL( hipMemcpy( floatData, hipFloatData, 3 * sizeof( float ), hipMemcpyDeviceToHost ) );
   TNL_BACKEND_SAFE_CALL( hipMemcpy( &intData, hipIntData, 3 * sizeof( int ), hipMemcpyDeviceToHost ) );

   EXPECT_NEAR( floatData[ 0 ], 3.14159, 0.0001 );
   EXPECT_NEAR( floatData[ 1 ], 2.71828, 0.0001 );
   EXPECT_NEAR( floatData[ 2 ], 1.61803, 0.0001 );

   EXPECT_EQ( intData[ 0 ], 3 );
   EXPECT_EQ( intData[ 1 ], 2 );
   EXPECT_EQ( intData[ 2 ], 1 );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

#include "main.h"
