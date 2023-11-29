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
#include <TNL/Backend.h>
#include <TNL/Exceptions/FileSerializationError.h>
#include <TNL/Exceptions/FileDeserializationError.h>

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
File::load( Type* destination, std::streamsize elements )
{
   static_assert( std::is_same_v< std::remove_cv_t< Type >, std::remove_cv_t< typename Allocator::value_type > >,
                  "Allocator::value_type must be the same as Type." );
   if( elements < 0 )
      throw std::invalid_argument( "File::load: number of elements to load must be non-negative." );
   else if( elements > 0 )
      load_impl< Type, SourceType, Allocator >( destination, elements );
}

// Host allocators
template< typename Type, typename SourceType, typename Allocator, typename >
void
File::load_impl( Type* destination, std::streamsize elements )
{
   if constexpr( std::is_same_v< std::remove_cv_t< Type >, std::remove_cv_t< SourceType > > )
      file.read( reinterpret_cast< char* >( destination ), sizeof( Type ) * elements );
   else {
      using BaseType = std::remove_cv_t< SourceType >;
      auto fill = [ = ]( std::size_t offset, BaseType* buffer, std::size_t buffer_size )
      {
         TNL_ASSERT_LE(
            offset + buffer_size, (std::size_t) elements, "bufferedTransferToDevice supplied wrong offset or buffer size" );
         this->file.read( reinterpret_cast< char* >( buffer ), sizeof( BaseType ) * buffer_size );
      };
      auto push = [ destination ]( std::size_t offset, const BaseType* buffer, std::size_t buffer_size, bool& next_iter )
      {
         for( std::size_t i = 0; i < buffer_size; i++ )
            destination[ offset + i ] = static_cast< Type >( buffer[ i ] );
      };
      Backend::bufferedTransfer< BaseType >( elements, fill, push );
   }
}

// Allocators::Cuda
template< typename Type, typename SourceType, typename Allocator, typename, typename >
void
File::load_impl( Type* destination, std::streamsize elements )
{
   using BaseType = std::remove_cv_t< SourceType >;
   std::unique_ptr< BaseType[] > cast_buffer = nullptr;
   std::size_t cast_buffer_size = 0;

   auto fill = [ &, this ]( std::size_t offset, Type* buffer, std::size_t buffer_size )
   {
      if constexpr( std::is_same_v< std::remove_cv_t< Type >, std::remove_cv_t< SourceType > > )
         this->file.read( reinterpret_cast< char* >( buffer ), sizeof( BaseType ) * buffer_size );
      else {
         // increase the cast buffer
         if( buffer_size > cast_buffer_size ) {
            cast_buffer = std::make_unique< BaseType[] >( buffer_size );
            cast_buffer_size = buffer_size;
         }
         // read from file to cast buffer
         this->file.read( reinterpret_cast< char* >( cast_buffer.get() ), sizeof( BaseType ) * buffer_size );
         // cast the elements in the buffer
         for( std::size_t i = 0; i < buffer_size; i++ )
            buffer[ i ] = static_cast< BaseType >( cast_buffer[ i ] );
      }
   };
   Backend::bufferedTransferToDevice( destination, elements, fill );
}

template< typename Type, typename TargetType, typename Allocator >
void
File::save( const Type* source, std::streamsize elements )
{
   static_assert( std::is_same_v< std::remove_cv_t< Type >, std::remove_cv_t< typename Allocator::value_type > >,
                  "Allocator::value_type must be the same as Type." );
   if( elements < 0 )
      throw std::invalid_argument( "File::save: number of elements to save must be non-negative." );
   else if( elements > 0 )
      save_impl< Type, TargetType, Allocator >( source, elements );
}

// Host allocators
template< typename Type, typename TargetType, typename Allocator, typename >
void
File::save_impl( const Type* source, std::streamsize elements )
{
   if constexpr( std::is_same_v< std::remove_cv_t< Type >, std::remove_cv_t< TargetType > > )
      file.write( reinterpret_cast< const char* >( source ), sizeof( Type ) * elements );
   else {
      using BaseType = std::remove_cv_t< TargetType >;
      auto fill = [ = ]( std::size_t offset, BaseType* buffer, std::size_t buffer_size )
      {
         TNL_ASSERT_LE(
            offset + buffer_size, (std::size_t) elements, "bufferedTransferToDevice supplied wrong offset or buffer size" );
         for( std::size_t i = 0; i < buffer_size; i++ )
            buffer[ i ] = static_cast< TargetType >( source[ offset + i ] );
      };
      auto push = [ this ]( std::size_t offset, const BaseType* buffer, std::size_t buffer_size, bool& next_iter )
      {
         this->file.write( reinterpret_cast< const char* >( buffer ), sizeof( TargetType ) * buffer_size );
      };
      Backend::bufferedTransfer< BaseType >( elements, fill, push );
   }
}

// Allocators::Cuda
template< typename Type, typename TargetType, typename Allocator, typename, typename >
void
File::save_impl( const Type* source, std::streamsize elements )
{
   using BaseType = std::remove_cv_t< TargetType >;
   std::unique_ptr< BaseType[] > cast_buffer = nullptr;
   std::size_t cast_buffer_size = 0;

   auto push = [ &, this ]( std::size_t offset, const Type* buffer, std::size_t buffer_size, bool& next_iter )
   {
      if constexpr( std::is_same_v< std::remove_cv_t< Type >, std::remove_cv_t< TargetType > > )
         this->file.write( reinterpret_cast< const char* >( buffer ), sizeof( BaseType ) * buffer_size );
      else {
         // increase the cast buffer
         if( buffer_size > cast_buffer_size ) {
            cast_buffer = std::make_unique< BaseType[] >( buffer_size );
            cast_buffer_size = buffer_size;
         }
         // cast the elements in the buffer
         for( std::size_t i = 0; i < buffer_size; i++ )
            cast_buffer[ i ] = static_cast< BaseType >( buffer[ i ] );
         // write from cast buffer to file
         this->file.write( reinterpret_cast< const char* >( cast_buffer.get() ), sizeof( BaseType ) * buffer_size );
      }
   };
   Backend::bufferedTransferToHost( source, elements, push );
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
