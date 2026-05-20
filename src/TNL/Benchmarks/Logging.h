// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <list>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>

namespace TNL::Benchmarks {

/**
 * \brief Container for formatted row data in benchmark output.
 *
 * Provides stream-like interface for building rows of string values.
 * Uses fixed-point notation with 6 decimal places by default.
 */
class LoggingRowElements
{
public:
   LoggingRowElements()
   {
      stream << std::setprecision( 6 ) << std::fixed;
   }

   /**
    * \brief Appends a value to the row.
    *
    * Converts the value to string and stores it for later iteration.
    *
    * \tparam T Type of value to append
    * \param b Value to add
    * \return Reference to this object for chaining
    */
   template< typename T >
   LoggingRowElements&
   operator<<( const T& b )
   {
      stream << b;
      elements.push_back( stream.str() );
      stream.str( std::string() );
      return *this;
   }

   /**
    * \brief Sets output precision.
    *
    * Allows changing precision mid-stream (e.g., `std::setprecision(2)`).
    *
    * \param setprec Precision manipulator
    * \return Reference to this object
    */
   LoggingRowElements&
   operator<<( decltype( std::setprecision( 2 ) )& setprec )
   {
      stream << setprec;
      return *this;
   }

   /**
    * \brief Sets number format (fixed or scientific).
    *
    * \param setfixed Format manipulator (\ref std::fixed or \ref std::scientific)
    * \return Reference to this object
    */
   LoggingRowElements&
   operator<<( decltype( std::fixed )& setfixed )  // the same works also for std::scientific
   {
      stream << setfixed;
      return *this;
   }

   //! \brief Returns the number of elements in the row.
   [[nodiscard]] std::size_t
   size() const noexcept
   {
      return elements.size();
   }

   // iterators
   [[nodiscard]] auto
   begin() noexcept
   {
      return elements.begin();
   }

   [[nodiscard]] auto
   begin() const noexcept
   {
      return elements.begin();
   }

   [[nodiscard]] auto
   cbegin() const noexcept
   {
      return elements.cbegin();
   }

   [[nodiscard]] auto
   end() noexcept
   {
      return elements.end();
   }

   [[nodiscard]] auto
   end() const noexcept
   {
      return elements.end();
   }

   [[nodiscard]] auto
   cend() const noexcept
   {
      return elements.cend();
   }

protected:
   std::list< std::string > elements;

   std::stringstream stream;
};

/**
 * \brief Abstract base class for benchmark result loggers.
 *
 * Provides common functionality for logging benchmark results to various
 * outputs (files, terminals, etc.). Supports metadata tracking and
 * configurable verbosity levels.
 */
class Logging
{
public:
   using MetadataElement = std::pair< std::string, std::string >;
   using MetadataColumns = std::vector< MetadataElement >;

   using HeaderElements = std::vector< std::string >;
   using RowElements = LoggingRowElements;

   virtual ~Logging() = default;

   /**
    * \brief Constructs logger with output stream.
    *
    * Enables exceptions on the output stream for error detection.
    *
    * \param log Output stream (file, cout, etc.)
    * \param verbose Verbosity level (0=silent, 1=normal, 2=verbose)
    */
   Logging( std::ostream& log, int verbose = 1 )
   : log( log ),
     verbose( verbose )
   {
      try {
         // check if we got an open file
         auto& file = dynamic_cast< std::ofstream& >( log );
         if( file.is_open() )
            // enable exceptions, but only if we got an open file
            // (under MPI, only the master rank typically opens the log file and thus
            // logs from other ranks are ignored here)
            file.exceptions( std::ostream::failbit | std::ostream::badbit | std::ostream::eofbit );
      }
      catch( std::bad_cast& ) {
         // also enable exceptions if we did not get a file
         log.exceptions( std::ostream::failbit | std::ostream::badbit | std::ostream::eofbit );
      }
   }

   /**
    * \brief Sets verbosity level.
    *
    * \param verbose Verbosity: 0=silent, 1=normal, 2=verbose
    */
   void
   setVerbose( int verbose )
   {
      this->verbose = verbose;
   }

   /**
    * \brief Gets current verbosity level.
    *
    * \return Verbosity setting
    */
   [[nodiscard]] int
   getVerbose() const
   {
      return verbose;
   }

   /**
    * \brief Sets all metadata columns at once.
    *
    * Tracks changes to detect when headers need updating.
    *
    * \param elements Vector of key-value metadata pairs
    */
   virtual void
   setMetadataColumns( const MetadataColumns& elements )
   {
      // check if a header element changed (i.e. a first item of the pairs)
      if( metadataColumns.size() != elements.size() )
         header_changed = true;
      else
         for( std::size_t i = 0; i < metadataColumns.size(); i++ )
            if( metadataColumns[ i ].first != elements[ i ].first ) {
               header_changed = true;
               break;
            }
      metadataColumns = elements;
   }

   /**
    * \brief Updates or adds a single metadata element.
    *
    * If the key already exists, updates its value. Otherwise adds new entry.
    * Negative insertPosition values count from the end.
    *
    * \param element Key-value pair to set
    * \param insertPosition Insert position (-1=end, -2=second-to-last, etc.)
    */
   virtual void
   setMetadataElement(
      const typename MetadataColumns::value_type& element,
      int insertPosition = -1 /* negative values insert from the end */ )
   {
      bool found = false;
      for( auto& it : metadataColumns )
         if( it.first == element.first ) {
            if( it.second != element.second )
               it.second = element.second;
            found = true;
            break;
         }
      if( ! found ) {
         if( insertPosition < 0 )
            metadataColumns.insert( metadataColumns.end() + insertPosition + 1, element );
         else
            metadataColumns.insert( metadataColumns.begin() + insertPosition, element );
         header_changed = true;
      }
   }

   /**
    * \brief Copies metadata from another logger instance.
    *
    * Used to synchronize state between multiple loggers.
    *
    * \param other Source logger to copy from
    */
   void
   syncMetadata( Logging& other )
   {
      other.metadataColumns = this->metadataColumns;
   }

   /**
    * \brief Gets current metadata columns.
    *
    * \return Reference to metadata vector
    */
   [[nodiscard]] const MetadataColumns&
   getMetadataColumns() const
   {
      return metadataColumns;
   }

   /**
    * \brief Logs a benchmark result row.
    *
    * Must be implemented by derived classes. Called after each measurement.
    *
    * \param performer Name of implementation being tested
    * \param headerElements Column names
    * \param rowElements Formatted row values
    * \param errorMessage Optional error description (empty if success)
    */
   virtual void
   logResult(
      const std::string& performer,
      const HeaderElements& headerElements,
      const RowElements& rowElements,
      const std::string& errorMessage = "" ) = 0;

   /**
    * \brief Logs an error message.
    *
    * Must be implemented by derived classes. Called when benchmark fails.
    *
    * \param message Error description
    */
   virtual void
   writeErrorMessage( const std::string& message ) = 0;

protected:
   std::ostream& log;
   int verbose = 0;

   MetadataColumns metadataColumns;
   bool header_changed = true;
};

}  // namespace TNL::Benchmarks
