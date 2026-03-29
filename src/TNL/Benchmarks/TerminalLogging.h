// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <iomanip>

#include "Logging.h"

namespace TNL::Benchmarks {

class TerminalLogger : public Logging
{
public:
   TerminalLogger( std::ostream& terminal, int verbose = 1 )
   : Logging( terminal, verbose )
   {}

   void
   logResult(
      const std::string& performer,
      const HeaderElements& headerElements,
      const RowElements& rowElements,
      const std::string& errorMessage = "" ) override
   {
      if( verbose <= 0 )
         return;

      // Build a map of header elements to row values
      std::map< std::string, std::string > rowMap;
      auto rowIt = rowElements.begin();
      for( const auto& header : headerElements ) {
         if( rowIt != rowElements.end() )
            rowMap[ header ] = *rowIt++;
      }

      // Check if metadata has changed from previous row
      bool metadataChanged = ( metadataColumns != lastMetadata );

      // Print metadata block header only when metadata changed
      if( metadataChanged ) {
         log << "\n=== Configuration ===\n";
         for( const auto& [ key, value ] : metadataColumns ) {
            log << std::setw( 14 ) << std::left << key << ": " << value << "\n";
         }
         log << "=== Results ===\n";
         lastMetadata = metadataColumns;
      }

      if( errorMessage.empty() ) {
         // Print result row: performer + key metrics
         log << std::setw( 14 ) << std::left << performer << ": ";
         if( verbose > 1 ) {
            // Print all key=value pairs separated by "  "
            for( size_t i = 0; i < headerElements.size(); i++ ) {
               if( i > 0 )
                  log << "  ";
               log << headerElements[ i ] << "=" << rowMap[ headerElements[ i ] ];
            }
         }
         else {
            // Print only essential results
            log << "time=" << rowMap[ "time" ];
            log << "  speedup=" << rowMap[ "speedup" ];
            log << "  bandwidth=" << rowMap[ "bandwidth" ];
            log << "  loops=" << rowMap[ "loops" ];
         }
         log << "\n";
      }
      else {
         log << std::setw( 14 ) << std::left << performer << ": ERROR - " << errorMessage << "\n";
      }
   }

   void
   writeErrorMessage( const std::string& message ) override
   {
      log << "ERROR: " << message << "\n";
   }

protected:
   MetadataColumns lastMetadata;
};

}  // namespace TNL::Benchmarks
