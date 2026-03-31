// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <iomanip>

#include "Logging.h"

#include <TNL/Assert.h>

namespace TNL::Benchmarks {

class JsonLogging : public Logging
{
public:
   JsonLogging( std::ostream& log, int verbose = 1 )
   : Logging( log, verbose )
   {}

   void
   writeRow( const HeaderElements& headerElements, const RowElements& rowElements, const std::string& errorMessage )
   {
      if( headerElements.size() != rowElements.size() ) {
         std::stringstream ss;
         ss << "writeRow: Header elements and row elements must have equal sizes. Header: " << headerElements.size()
            << ", Row: " << rowElements.size();
         throw std::invalid_argument( ss.str() );
      }

      log << "{";

      // write common logs
      int idx( 0 );
      for( const auto& lg : this->metadataColumns ) {
         if( idx++ > 0 )
            log << ", ";
         log << "\"" << lg.first << "\": \"" << lg.second << "\"";
      }

      std::size_t i = 0;
      for( const auto& el : rowElements ) {
         if( idx++ > 0 )
            log << ", ";
         log << "\"" << headerElements[ i ] << "\": \"" << el << "\"";
         i++;
      }
      if( ! errorMessage.empty() ) {
         if( idx++ > 0 )
            log << ", ";
         log << "\"error\": \"" << escape_json( errorMessage ) << "\"";
      }
      log << "}\n";
      log << std::flush;
   }

   void
   logResult(
      const std::string& performer,
      const HeaderElements& headerElements,
      const RowElements& rowElements,
      const std::string& errorMessage = "" ) override
   {
      setMetadataElement( { "performer", performer } );
      writeRow( headerElements, rowElements, errorMessage );
   }

   void
   writeErrorMessage( const std::string& message ) override
   {
      log << "{";

      // write common logs
      int idx( 0 );
      for( const auto& lg : this->metadataColumns ) {
         if( idx++ > 0 )
            log << ", ";
         log << "\"" << lg.first << "\": \"" << lg.second << "\"";
      }

      if( idx++ > 0 )
         log << ", ";
      log << "\"error\": \"" << escape_json( message ) << "\"";

      log << "}\n";
      log << std::flush;
   }

protected:
   // https://stackoverflow.com/a/33799784
   static std::string
   escape_json( const std::string& s )
   {
      std::ostringstream o;
      for( auto c : s ) {
         switch( c ) {
            case '"':
               o << "\\\"";
               break;
            case '\\':
               o << "\\\\";
               break;
            case '\b':
               o << "\\b";
               break;
            case '\f':
               o << "\\f";
               break;
            case '\n':
               o << "\\n";
               break;
            case '\r':
               o << "\\r";
               break;
            case '\t':
               o << "\\t";
               break;
            default:
               if( '\x00' <= c && c <= '\x1f' )
                  o << "\\u" << std::hex << std::setw( 4 ) << std::setfill( '0' ) << static_cast< int >( c );
               else
                  o << c;
         }
      }
      return o.str();
   }
};

}  // namespace TNL::Benchmarks
