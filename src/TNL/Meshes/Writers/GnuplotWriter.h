// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Meshes {
namespace Writers {

template< typename Mesh >
class GnuplotWriter
{
public:

   GnuplotWriter() = delete;

   GnuplotWriter( std::ostream& str );

   template< typename Array >
   void
   writePointData( const Mesh& mesh, const Array& array, const std::string& name, int numberOfComponents = 1 );

   template< typename Array >
   void
   writeCellData( const Mesh& mesh, const Array& array, const std::string& name, int numberOfComponents = 1 );


   void writeHeader( Mesh& mesh );

   template< typename Element >
   static void
   write( std::ostream& str, const Element& d );

   template< typename Real >
   static void
   write( std::ostream& str, const Containers::StaticVector< 1, Real >& d );

   template< typename Real >
   static void
   write( std::ostream& str, const Containers::StaticVector< 2, Real >& d );

   template< typename Real >
   static void
   write( std::ostream& str, const Containers::StaticVector< 3, Real >& d );

protected:

   std::ostream& str;
};

}  // namespace Writers
}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/Writers/GnuplotWriter.hpp>
