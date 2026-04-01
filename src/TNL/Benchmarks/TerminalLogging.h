// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <iomanip>

#include "Logging.h"

namespace TNL::Benchmarks {

/**
 * \brief Logger that outputs benchmark results to terminal/console.
 *
 * Provides human-readable output with configurable verbosity:
 *
 * - Level 0: No output (silent)
 * - Level 1: Summary with key metrics (time, speedup, bandwidth, loops)
 * - Level 2+: Full details including all metrics and configuration
 *
 * Metadata is printed once when it changes, followed by result rows.
 *
 * Example output (verbose=1):
 *
 * \code
 * === Configuration ===
 * operation     : multiply
 * precision     : double
 * === Results ===
 * CPU           : time=4.44e-04  speedup=N/A  bandwidth=1.11GB/s  loops=10
 * GPU           : time=1.11e-04  speedup=4.0  bandwidth=4.44GB/s  loops=10
 * \endcode
 */
class TerminalLogger : public Logging
{
public:
   /**
    * \brief Constructs terminal logger.
    *
    * \param terminal Output stream (typically \ref std::cout)
    * \param verbose Verbosity level (0=silent, 1=normal, 2=verbose)
    */
   TerminalLogger( std::ostream& terminal, int verbose = 1 )
   : Logging( terminal, verbose )
   {}

   /**
    * \brief Logs a benchmark result to terminal.
    *
    * Prints metadata block header when metadata changes, then the result row.
    * Format varies based on verbosity level.
    *
    * \param performer Name of implementation being tested
    * \param headerElements Column names
    * \param rowElements Formatted row values
    * \param errorMessage Optional error description
    */
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
      log << std::flush;
   }

   /**
    * \brief Logs an error message to terminal.
    *
    * \param message Error description
    */
   void
   writeErrorMessage( const std::string& message ) override
   {
      log << "ERROR: " << message << "\n";
      log << std::flush;
   }

protected:
   MetadataColumns lastMetadata;
};

}  // namespace TNL::Benchmarks
