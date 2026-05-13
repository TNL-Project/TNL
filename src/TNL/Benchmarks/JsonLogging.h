// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <nlohmann/json.hpp>

#include "Logging.h"

#include <TNL/Assert.h>

namespace TNL::Benchmarks {

/**
 * \brief Logger that outputs benchmark results as JSON lines (JSONL).
 *
 * Each result is written as a separate JSON object on its own line,
 * making it easy to parse and process with standard tools.
 * See https://jsonltools.com/what-is-jsonl for details.
 *
 * Output format example:
 * \code
 * {"operation": "multiply", "performer": "CPU", "time": "1.23e-04", ...}
 * {"operation": "multiply", "performer": "GPU", "time": "4.56e-05", ...}
 * \endcode
 *
 * Error messages are also logged as JSON objects with an "error" field.
 */
class JsonLogging : public Logging
{
public:
   /**
    * \brief Constructs JSON logger.
    *
    * \param log Output stream for JSONL data
    * \param verbose Verbosity level (passed to base class, not used by this class)
    */
   JsonLogging( std::ostream& log, int verbose = 1 )
   : Logging( log, verbose )
   {}

   /**
    * \brief Writes a single result row as JSON.
    *
    * Outputs all metadata columns followed by result values, each as
    * key-value pairs in the JSON object.
    *
    * \param headerElements Column names
    * \param rowElements Formatted row values
    * \param errorMessage Optional error description
    */
   void
   writeRow( const HeaderElements& headerElements, const RowElements& rowElements, const std::string& errorMessage )
   {
      if( headerElements.size() != rowElements.size() ) {
         std::stringstream ss;
         ss << "writeRow: Header elements and row elements must have equal sizes. Header: " << headerElements.size()
            << ", Row: " << rowElements.size();
         throw std::invalid_argument( ss.str() );
      }

      nlohmann::json record;

      for( const auto& lg : this->metadataColumns )
         record[ lg.first ] = lg.second;

      std::size_t i = 0;
      for( const auto& el : rowElements )
         record[ headerElements[ i++ ] ] = el;

      if( ! errorMessage.empty() )
         record[ "error" ] = errorMessage;

      log << record.dump() << "\n" << std::flush;
   }

   /**
    * \brief Logs a benchmark result as JSON.
    *
    * Adds performer metadata and writes the result row.
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
      setMetadataElement( { "performer", performer } );
      writeRow( headerElements, rowElements, errorMessage );
   }

   /**
    * \brief Logs an error message as JSON.
    *
    * Outputs current metadata plus an "error" field containing the message.
    *
    * \param message Error description
    */
   void
   writeErrorMessage( const std::string& message ) override
   {
      nlohmann::json record;

      for( const auto& lg : this->metadataColumns )
         record[ lg.first ] = lg.second;

      record[ "error" ] = message;

      log << record.dump() << "\n" << std::flush;
   }
};

}  // namespace TNL::Benchmarks
