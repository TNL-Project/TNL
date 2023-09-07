// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <iostream>
#include <ios>
#include <sstream>

#include <TNL/File.h>
#include <TNL/Assert.h>
#include <TNL/Cuda/CheckDevice.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/FileSerializationError.h>
#include <TNL/Exceptions/FileDeserializationError.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {

inline File::File( const std::string& fileName, std::ios_base::openmode mode )
{
   open( fileName, mode );
}

inline void
File::open( const std::string& fileName, std::ios_base::openmode mode )
{
   // enable exceptions
   file.exceptions( std::fstream::failbit | std::fstream::badbit | std::fstream::eofbit );

   close();

   mode |= std::ios::binary;
   try {
      file.open( fileName, mode );
   }
   catch( std::ios_base::failure& ) {
      std::stringstream msg;
      msg << "Unable to open file " << fileName << " ";
      if( ( mode & std::ios_base::in ) != 0 )
         msg << " for reading.";
      if( ( mode & std::ios_base::out ) != 0 )
         msg << " for writing.";

      throw std::ios_base::failure( msg.str() );
   }

   this->fileName = fileName;
}

inline void
File::close()
{
   if( file.is_open() ) {
      try {
         file.close();
      }
      catch( std::ios_base::failure& ) {
         std::stringstream msg;
         msg << "Unable to close file " << fileName << ".";

         throw std::ios_base::failure( msg.str() );
      }
   }
   // reset file name
   fileName = "";
}

template< typename Type, typename SourceType, typename Allocator >
void
File::load( Type* buffer, std::streamsize elements )
{
   static_assert( std::is_same< Type, typename Allocator::value_type >::value,
                  "Allocator::value_type must be the same as Type." );
   if( elements < 0 )
      throw std::invalid_argument( "File::load: number of elements to load must be non-negative." );
   else if( elements > 0 )
      load_impl< Type, SourceType, Allocator >( buffer, elements );
}

// Host allocators
template< typename Type, typename SourceType, typename Allocator, typename >
void
File::load_impl( Type* buffer, std::streamsize elements )
{
   if constexpr( std::is_same< Type, SourceType >::value )
      file.read( reinterpret_cast< char* >( buffer ), sizeof( Type ) * elements );
   else {
      const std::streamsize cast_buffer_size =
         std::min( Cuda::getTransferBufferSize() / (std::streamsize) sizeof( SourceType ), elements );
      using BaseType = typename std::remove_cv< SourceType >::type;
      std::unique_ptr< BaseType[] > cast_buffer{ new BaseType[ cast_buffer_size ] };
      std::streamsize readElements = 0;
      while( readElements < elements ) {
         const std::streamsize transfer = std::min( elements - readElements, cast_buffer_size );
         file.read( reinterpret_cast< char* >( cast_buffer.get() ), sizeof( SourceType ) * transfer );
         for( std::streamsize i = 0; i < transfer; i++ )
            buffer[ readElements++ ] = static_cast< Type >( cast_buffer[ i ] );
      }
   }
}

// Allocators::Cuda
template< typename Type, typename SourceType, typename Allocator, typename, typename >
void
File::load_impl( Type* buffer, std::streamsize elements )
{
#ifdef __CUDACC__
   const std::streamsize host_buffer_size =
      std::min( Cuda::getTransferBufferSize() / (std::streamsize) sizeof( Type ), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   std::unique_ptr< BaseType[] > host_buffer{ new BaseType[ host_buffer_size ] };

   std::streamsize readElements = 0;
   if constexpr( std::is_same< Type, SourceType >::value ) {
      while( readElements < elements ) {
         const std::streamsize transfer = std::min( elements - readElements, host_buffer_size );
         file.read( reinterpret_cast< char* >( host_buffer.get() ), sizeof( Type ) * transfer );
         cudaMemcpy(
            (void*) &buffer[ readElements ], (void*) host_buffer.get(), transfer * sizeof( Type ), cudaMemcpyHostToDevice );
         TNL_CHECK_CUDA_DEVICE;
         readElements += transfer;
      }
   }
   else {
      const std::streamsize cast_buffer_size =
         std::min( Cuda::getTransferBufferSize() / (std::streamsize) sizeof( SourceType ), elements );
      using BaseType = typename std::remove_cv< SourceType >::type;
      std::unique_ptr< BaseType[] > cast_buffer{ new BaseType[ cast_buffer_size ] };

      while( readElements < elements ) {
         const std::streamsize transfer = std::min( elements - readElements, cast_buffer_size );
         file.read( reinterpret_cast< char* >( cast_buffer.get() ), sizeof( SourceType ) * transfer );
         for( std::streamsize i = 0; i < transfer; i++ )
            host_buffer[ i ] = static_cast< Type >( cast_buffer[ i ] );
         cudaMemcpy(
            (void*) &buffer[ readElements ], (void*) host_buffer.get(), transfer * sizeof( Type ), cudaMemcpyHostToDevice );
         TNL_CHECK_CUDA_DEVICE;
         readElements += transfer;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename Type, typename TargetType, typename Allocator >
void
File::save( const Type* buffer, std::streamsize elements )
{
   static_assert( std::is_same< std::remove_cv_t< Type >, std::remove_cv_t< typename Allocator::value_type > >::value,
                  "Allocator::value_type must be the same as Type." );
   if( elements < 0 )
      throw std::invalid_argument( "File::save: number of elements to save must be non-negative." );
   else if( elements > 0 )
      save_impl< Type, TargetType, Allocator >( buffer, elements );
}

// Host allocators
template< typename Type, typename TargetType, typename Allocator, typename >
void
File::save_impl( const Type* buffer, std::streamsize elements )
{
   if constexpr( std::is_same< Type, TargetType >::value )
      file.write( reinterpret_cast< const char* >( buffer ), sizeof( Type ) * elements );
   else {
      const std::streamsize cast_buffer_size =
         std::min( Cuda::getTransferBufferSize() / (std::streamsize) sizeof( TargetType ), elements );
      using BaseType = typename std::remove_cv< TargetType >::type;
      std::unique_ptr< BaseType[] > cast_buffer{ new BaseType[ cast_buffer_size ] };
      std::streamsize writtenElements = 0;
      while( writtenElements < elements ) {
         const std::streamsize transfer = std::min( elements - writtenElements, cast_buffer_size );
         for( std::streamsize i = 0; i < transfer; i++ )
            cast_buffer[ i ] = static_cast< TargetType >( buffer[ writtenElements++ ] );
         file.write( reinterpret_cast< char* >( cast_buffer.get() ), sizeof( TargetType ) * transfer );
      }
   }
}

// Allocators::Cuda
template< typename Type, typename TargetType, typename Allocator, typename, typename >
void
File::save_impl( const Type* buffer, std::streamsize elements )
{
#ifdef __CUDACC__
   const std::streamsize host_buffer_size =
      std::min( Cuda::getTransferBufferSize() / (std::streamsize) sizeof( Type ), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   std::unique_ptr< BaseType[] > host_buffer{ new BaseType[ host_buffer_size ] };

   std::streamsize writtenElements = 0;
   if constexpr( std::is_same< Type, TargetType >::value ) {
      while( writtenElements < elements ) {
         const std::streamsize transfer = std::min( elements - writtenElements, host_buffer_size );
         cudaMemcpy(
            (void*) host_buffer.get(), (void*) &buffer[ writtenElements ], transfer * sizeof( Type ), cudaMemcpyDeviceToHost );
         TNL_CHECK_CUDA_DEVICE;
         file.write( reinterpret_cast< const char* >( host_buffer.get() ), sizeof( Type ) * transfer );
         writtenElements += transfer;
      }
   }
   else {
      const std::streamsize cast_buffer_size =
         std::min( Cuda::getTransferBufferSize() / (std::streamsize) sizeof( TargetType ), elements );
      using BaseType = typename std::remove_cv< TargetType >::type;
      std::unique_ptr< BaseType[] > cast_buffer{ new BaseType[ cast_buffer_size ] };

      while( writtenElements < elements ) {
         const std::streamsize transfer = std::min( elements - writtenElements, host_buffer_size );
         cudaMemcpy(
            (void*) host_buffer.get(), (void*) &buffer[ writtenElements ], transfer * sizeof( Type ), cudaMemcpyDeviceToHost );
         TNL_CHECK_CUDA_DEVICE;
         for( std::streamsize i = 0; i < transfer; i++ )
            cast_buffer[ i ] = static_cast< TargetType >( host_buffer[ i ] );

         file.write( reinterpret_cast< const char* >( cast_buffer.get() ), sizeof( TargetType ) * transfer );
         writtenElements += transfer;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename SourceType >
void
File::ignore( std::streamsize elements )
{
   // use seekg instead of ignore for efficiency
   // https://stackoverflow.com/a/31246560
   file.seekg( sizeof( SourceType ) * elements, std::ios_base::cur );
}

// serialization of strings
inline File&
operator<<( File& file, const std::string& str )
{
   const int len = str.size();
   try {
      file.save( &len );
   }
   catch( ... ) {
      throw Exceptions::FileSerializationError( file.getFileName(), "unable to write string length." );
   }
   try {
      file.save( str.c_str(), len );
   }
   catch( ... ) {
      throw Exceptions::FileSerializationError( file.getFileName(), "unable to write a C-string." );
   }
   return file;
}

// deserialization of strings
inline File&
operator>>( File& file, std::string& str )
{
   int length;
   try {
      file.load( &length );
   }
   catch( ... ) {
      throw Exceptions::FileDeserializationError( file.getFileName(), "unable to read string length." );
   }
   if( length > 0 ) {
      std::unique_ptr< char[] > buffer{ new char[ length ] };
      try {
         file.load( buffer.get(), length );
      }
      catch( ... ) {
         throw Exceptions::FileDeserializationError( file.getFileName(), "unable to read a C-string." );
      }
      str.assign( buffer.get(), length );
   }
   return file;
}

}  // namespace TNL
