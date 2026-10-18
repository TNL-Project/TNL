// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>

namespace TNL::Matrices {

/**
 * \brief Helper class for exporting of matrices to different output formats.
 *
 * Currently it supports:
 *
 * 1. [Coordinate MTX Format](https://math.nist.gov/MatrixMarket/formats.html#coord) is supported.
 * 2. Gnuplot format for matrix visualization in [Gnuplot](http://www.gnuplot.info).
 * 3. EPS format for matrix pattern visualization in [Encapsulated
 * PostScript](https://en.wikipedia.org/wiki/Encapsulated_PostScript)
 *
 * Matrices allocated on \ref TNL::Devices::Host and \ref TNL::Devices::Sequential are
 * written directly, matrices allocated on other devices are first copied to a host matrix.
 *
 * \tparam Matrix is a type of matrix we want to export.
 *
 * \par Example
 * \include Matrices/MatrixWriterReaderExample.cpp
 * \par Output
 * \include MatrixWriterReaderExample.out
 *
 */
template< typename Matrix >
class MatrixWriter
{
public:
   /**
    * \brief Type of matrix elements values.
    */
   using RealType = typename Matrix::RealType;

   /**
    * \brief Device where the matrix is allocated.
    */
   using DeviceType = typename Matrix::DeviceType;

   /**
    * \brief Type used for indexing of matrix elements.
    */
   using IndexType = typename Matrix::IndexType;

   /**
    * \brief Method for exporting matrix to file with given filename using Gnuplot format.
    *
    * \param fileName is the name of the target file.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeGnuplot( const std::string& fileName, const Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for exporting matrix to STL output stream using Gnuplot format.
    *
    * \param str is the output stream.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeGnuplot( std::ostream& str, const Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for exporting matrix to file with given filename using EPS format.
    *
    * \param fileName is the name of the target file.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeEps( const std::string& fileName, const Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for exporting matrix to STL output stream using EPS format.
    *
    * \param str is the output stream.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeEps( std::ostream& str, const Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for exporting matrix to file with given filename using MTX format.
    *
    * \param fileName is the name of the target file.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeMtx( const std::string& fileName, const Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for exporting matrix to STL output stream using MTX format.
    *
    * \param str is the output stream.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeMtx( std::ostream& str, const Matrix& matrix, bool verbose = false );

protected:
   /**
    * \brief Host copy of the matrix used for exporting matrices from other devices.
    */
   using HostMatrix = typename Matrix::template Self< RealType, TNL::Devices::Host >;

   /**
    * \brief Method for writing the EPS header.
    *
    * \param str is the output stream.
    * \param matrix is the source matrix.
    * \param elementSize is the size of a single element in PostScript units.
    */
   static void
   writeEpsHeader( std::ostream& str, const Matrix& matrix, int elementSize );

   /**
    * \brief Method for writing the EPS body with the matrix pattern.
    *
    * \param str is the output stream.
    * \param matrix is the source matrix.
    * \param elementSize is the size of a single element in PostScript units.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeEpsBody( std::ostream& str, const Matrix& matrix, int elementSize, bool verbose );
};

}  // namespace TNL::Matrices

#include <TNL/Matrices/MatrixWriter.hpp>
