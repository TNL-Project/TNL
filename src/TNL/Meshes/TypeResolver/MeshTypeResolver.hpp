// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <utility>

#include <TNL/Meshes/TypeResolver/MeshTypeResolver.h>
#include <TNL/Meshes/VTKTraits.h>

namespace TNL::Meshes {

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
void
MeshTypeResolver< ConfigTag, Device >::run( Reader& reader, Functor&& functor )
{
   detail< Reader, Functor >::resolveCellTopology( reader, std::forward< Functor >( functor ) );
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
void
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveCellTopology( Reader& reader, Functor&& functor )
{
   switch( reader.getCellShape() ) {
      case VTK::EntityShape::Line:
         resolveSpaceDimension< Topologies::Edge >( reader, std::forward< Functor >( functor ) );
         break;
      case VTK::EntityShape::Triangle:
         resolveSpaceDimension< Topologies::Triangle >( reader, std::forward< Functor >( functor ) );
         break;
      case VTK::EntityShape::Quad:
         resolveSpaceDimension< Topologies::Quadrangle >( reader, std::forward< Functor >( functor ) );
         break;
      case VTK::EntityShape::Tetra:
         resolveSpaceDimension< Topologies::Tetrahedron >( reader, std::forward< Functor >( functor ) );
         break;
      case VTK::EntityShape::Hexahedron:
         resolveSpaceDimension< Topologies::Hexahedron >( reader, std::forward< Functor >( functor ) );
         break;
      case VTK::EntityShape::Polygon:
         resolveSpaceDimension< Topologies::Polygon >( reader, std::forward< Functor >( functor ) );
         break;
      case VTK::EntityShape::Wedge:
         resolveSpaceDimension< Topologies::Wedge >( reader, std::forward< Functor >( functor ) );
         break;
      case VTK::EntityShape::Pyramid:
         resolveSpaceDimension< Topologies::Pyramid >( reader, std::forward< Functor >( functor ) );
         break;
      case VTK::EntityShape::Polyhedron:
         resolveSpaceDimension< Topologies::Polyhedron >( reader, std::forward< Functor >( functor ) );
         break;
      default:
         throw std::invalid_argument( "unsupported cell topology: " + VTK::getShapeName( reader.getCellShape() ) );
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename CellTopology >
void
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveSpaceDimension( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshCellTopologyTag< ConfigTag, CellTopology >::enabled ) {
      switch( reader.getSpaceDimension() ) {
         case 1:
            resolveReal< CellTopology, 1 >( reader, std::forward< Functor >( functor ) );
            break;
         case 2:
            resolveReal< CellTopology, 2 >( reader, std::forward< Functor >( functor ) );
            break;
         case 3:
            resolveReal< CellTopology, 3 >( reader, std::forward< Functor >( functor ) );
            break;
         default:
            throw std::invalid_argument( "unsupported space dimension: " + std::to_string( reader.getSpaceDimension() ) );
      }
   }
   else {
      throw std::invalid_argument( "The cell topology " + getType< CellTopology >()
                                   + " is disabled in the build configuration." );
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename CellTopology, int SpaceDimension >
void
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveReal( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshSpaceDimensionTag< ConfigTag, CellTopology, SpaceDimension >::enabled ) {
      if( reader.getRealType() == "float" )
         resolveGlobalIndex< CellTopology, SpaceDimension, float >( reader, std::forward< Functor >( functor ) );
      else if( reader.getRealType() == "double" )
         resolveGlobalIndex< CellTopology, SpaceDimension, double >( reader, std::forward< Functor >( functor ) );
      else if( reader.getRealType() == "long double" )
         resolveGlobalIndex< CellTopology, SpaceDimension, long double >( reader, std::forward< Functor >( functor ) );
      else
         throw std::invalid_argument( "Unsupported real type: " + reader.getRealType() );
   }
   else {
      throw std::invalid_argument( "The combination of space dimension (" + std::to_string( SpaceDimension )
                                   + ") and mesh dimension (" + std::to_string( CellTopology::dimension )
                                   + ") is either invalid or disabled in the build configuration." );
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename CellTopology, int SpaceDimension, typename Real >
void
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveGlobalIndex( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshRealTag< ConfigTag, Real >::enabled ) {
      if( reader.getGlobalIndexType() == "std::int16_t" || reader.getGlobalIndexType() == "std::uint16_t" )
         resolveLocalIndex< CellTopology, SpaceDimension, Real, std::int16_t >( reader, std::forward< Functor >( functor ) );
      else if( reader.getGlobalIndexType() == "std::int32_t" || reader.getGlobalIndexType() == "std::uint32_t" )
         resolveLocalIndex< CellTopology, SpaceDimension, Real, std::int32_t >( reader, std::forward< Functor >( functor ) );
      else if( reader.getGlobalIndexType() == "std::int64_t" || reader.getGlobalIndexType() == "std::uint64_t" )
         resolveLocalIndex< CellTopology, SpaceDimension, Real, std::int64_t >( reader, std::forward< Functor >( functor ) );
      else
         throw std::invalid_argument( "Unsupported global index type: " + reader.getGlobalIndexType() );
   }
   else {
      throw std::invalid_argument( "The mesh real type " + getType< Real >() + " is disabled in the build configuration." );
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex >
void
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveLocalIndex( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshGlobalIndexTag< ConfigTag, GlobalIndex >::enabled ) {
      if( reader.getLocalIndexType() == "std::int16_t" || reader.getLocalIndexType() == "std::uint16_t" )
         resolveMeshType< CellTopology, SpaceDimension, Real, GlobalIndex, std::int16_t >( reader,
                                                                                           std::forward< Functor >( functor ) );
      else if( reader.getLocalIndexType() == "std::int32_t" || reader.getLocalIndexType() == "std::uint32_t" )
         resolveMeshType< CellTopology, SpaceDimension, Real, GlobalIndex, std::int32_t >( reader,
                                                                                           std::forward< Functor >( functor ) );
      else if( reader.getLocalIndexType() == "std::int64_t" || reader.getLocalIndexType() == "std::uint64_t" )
         resolveMeshType< CellTopology, SpaceDimension, Real, GlobalIndex, std::int64_t >( reader,
                                                                                           std::forward< Functor >( functor ) );
      else
         throw std::invalid_argument( "Unsupported local index type: " + reader.getLocalIndexType() );
   }
   else {
      throw std::invalid_argument( "The mesh global index type " + getType< GlobalIndex >()
                                   + " is disabled in the build configuration." );
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex, typename LocalIndex >
void
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveMeshType( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshLocalIndexTag< ConfigTag, LocalIndex >::enabled ) {
      using MeshConfig = typename BuildConfigTags::MeshConfigTemplateTag<
         ConfigTag >::template MeshConfig< CellTopology, SpaceDimension, Real, GlobalIndex, LocalIndex >;
      resolveTerminate< MeshConfig >( reader, std::forward< Functor >( functor ) );
   }
   else {
      throw std::invalid_argument( "The mesh local index type " + getType< LocalIndex >()
                                   + " is disabled in the build configuration." );
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename MeshConfig >
void
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveTerminate( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshDeviceTag< ConfigTag, Device >::enabled
                 && BuildConfigTags::MeshTag< ConfigTag,
                                              Device,
                                              typename MeshConfig::CellTopology,
                                              MeshConfig::spaceDimension,
                                              typename MeshConfig::RealType,
                                              typename MeshConfig::GlobalIndexType,
                                              typename MeshConfig::LocalIndexType >::enabled )
   {
      using MeshType = Meshes::Mesh< MeshConfig, Device >;
      std::forward< Functor >( functor )( reader, MeshType{} );
   }
   else {
      throw std::invalid_argument( "The mesh config type " + getType< MeshConfig >()
                                   + " is disabled in the build configuration for device " + getType< Device >() + "." );
   }
}

}  // namespace TNL::Meshes
