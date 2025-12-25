// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <utility>

#include <TNL/Meshes/TypeResolver/GridTypeResolver.h>

namespace TNL::Meshes {

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
void
GridTypeResolver< ConfigTag, Device >::run( Reader& reader, Functor&& functor )
{
   detail< Reader, Functor >::resolveGridDimension( reader, std::forward< Functor >( functor ) );
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
void
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveGridDimension( Reader& reader, Functor&& functor )
{
   if( reader.getMeshDimension() == 1 )
      resolveReal< 1 >( reader, std::forward< Functor >( functor ) );
   else if( reader.getMeshDimension() == 2 )
      resolveReal< 2 >( reader, std::forward< Functor >( functor ) );
   else if( reader.getMeshDimension() == 3 )
      resolveReal< 3 >( reader, std::forward< Functor >( functor ) );
   else
      throw std::invalid_argument( "Unsupported mesh dimension: " + std::to_string( reader.getMeshDimension() ) );
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< int MeshDimension >
void
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveReal( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::GridDimensionTag< ConfigTag, MeshDimension >::enabled ) {
      if( reader.getRealType() == "float" )
         resolveIndex< MeshDimension, float >( reader, std::forward< Functor >( functor ) );
      else if( reader.getRealType() == "double" )
         resolveIndex< MeshDimension, double >( reader, std::forward< Functor >( functor ) );
      else if( reader.getRealType() == "long double" )
         resolveIndex< MeshDimension, long double >( reader, std::forward< Functor >( functor ) );
      else
         throw std::invalid_argument( "Unsupported real type: " + reader.getRealType() );
   }
   else {
      throw std::invalid_argument( "The grid dimension " + std::to_string( MeshDimension )
                                   + " is disabled in the build configuration." );
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< int MeshDimension, typename Real >
void
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveIndex( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::GridRealTag< ConfigTag, Real >::enabled ) {
      if( reader.getGlobalIndexType() == "std::int16_t" || reader.getGlobalIndexType() == "std::uint16_t" )
         resolveGridType< MeshDimension, Real, std::int16_t >( reader, std::forward< Functor >( functor ) );
      else if( reader.getGlobalIndexType() == "std::int32_t" || reader.getGlobalIndexType() == "std::uint32_t" )
         resolveGridType< MeshDimension, Real, std::int32_t >( reader, std::forward< Functor >( functor ) );
      else if( reader.getGlobalIndexType() == "std::int64_t" || reader.getGlobalIndexType() == "std::uint64_t" )
         resolveGridType< MeshDimension, Real, std::int64_t >( reader, std::forward< Functor >( functor ) );
      else
         throw std::invalid_argument( "Unsupported index type: " + reader.getRealType() );
   }
   else {
      throw std::invalid_argument( "The grid real type " + getType< Real >() + " is disabled in the build configuration." );
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< int MeshDimension, typename Real, typename Index >
void
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveGridType( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::GridIndexTag< ConfigTag, Index >::enabled ) {
      using GridType = Meshes::Grid< MeshDimension, Real, Device, Index >;
      resolveTerminate< GridType >( reader, std::forward< Functor >( functor ) );
   }
   else {
      throw std::invalid_argument( "The grid index type " + getType< Index >() + " is disabled in the build configuration." );
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename GridType >
void
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveTerminate( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::GridTag< ConfigTag, GridType >::enabled ) {
      std::forward< Functor >( functor )( reader, GridType{} );
   }
   else {
      throw std::invalid_argument( "The mesh type " + TNL::getType< GridType >() + " is disabled in the build configuration." );
   }
}

}  // namespace TNL::Meshes
