// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <istream>
#include <string>

#include <TNL/String.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Matrices {

/**
 * \brief Handling of matrix elements which appear more than once in an MTX file.
 *
 * In a symmetric or skew-symmetric MTX file, the elements (i,j) and (j,i) are the same element.
 */
enum class MtxDuplicateElements : std::uint8_t
{
   //! Throw an exception.
   Error,
   //! Use the value which appears last in the file.
   KeepLast,
   //! Use the sum of all values of the element.
   Sum
};

/**
 * \brief Options for importing of matrices from MTX files by \ref MatrixReader.
 */
struct MtxReaderOptions
{
   //! Controls verbosity of the matrix import.
   bool verbose = false;

   //! Throw an exception if the number of matrix elements in the file differs from the number given in its size line,
   //! e.g. because the file was truncated.
   bool checkElementsCount = true;

   //! Handling of matrix elements which appear more than once in the file.
   MtxDuplicateElements duplicates = MtxDuplicateElements::Error;
};

/**
 * \brief Helper class for importing of matrices from different input formats.
 *
 * Currently it supports:
 *
 * 1. [Coordinate MTX Format](https://math.nist.gov/MatrixMarket/formats.html#coord) is supported.
 *
 * \tparam Matrix is a type of matrix into which we want to import the MTX file.
 * \tparam Device is used only for the purpose of template specialization.
 *
 * \par Example
 * \include Matrices/MatrixWriterReaderExample.cpp
 * \par Output
 * \include MatrixWriterReaderExample.out
 */
template< typename Matrix, typename Device = typename Matrix::DeviceType >
class MatrixReader
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
    * \brief Method for importing matrix from file with given filename.
    *
    * \param fileName is the name of the source file.
    * \param matrix is the target matrix.
    * \param verbose controls verbosity of the matrix import.
    */
   static void
   readMtx( const std::string& fileName, Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for importing matrix from STL input stream.
    *
    * \param str is the input stream.
    * \param matrix is the target matrix.
    * \param verbose controls verbosity of the matrix import.
    */
   static void
   readMtx( std::istream& str, Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for importing matrix from file with given filename.
    *
    * \param fileName is the name of the source file.
    * \param matrix is the target matrix.
    * \param options are the options of the import, see \ref MtxReaderOptions.
    */
   static void
   readMtx( const std::string& fileName, Matrix& matrix, const MtxReaderOptions& options );

   /**
    * \brief Method for importing matrix from STL input stream.
    *
    * \param file is the input stream.
    * \param matrix is the target matrix.
    * \param options are the options of the import, see \ref MtxReaderOptions.
    */
   static void
   readMtx( std::istream& file, Matrix& matrix, const MtxReaderOptions& options );

   /**
    * \brief Check if the MTX file with given filename declares a symmetric matrix.
    *
    * Only the header of the file is read, i.e. the symmetry is the one declared by the
    * header line `%%MatrixMarket matrix coordinate <field> symmetric`, the matrix elements
    * are not checked.
    *
    * \param fileName is the name of the source file.
    * \return `true` if the header declares a symmetric matrix and `false` if it declares
    * a general or skew-symmetric one.
    */
   [[nodiscard]] static bool
   isSymmetric( const std::string& fileName );

   /**
    * \brief Check if the MTX file in STL input stream declares a symmetric matrix.
    *
    * The header is read from the beginning of the stream.
    *
    * \param file is the input stream.
    * \return `true` if the header declares a symmetric matrix and `false` if it declares
    * a general or skew-symmetric one.
    */
   [[nodiscard]] static bool
   isSymmetric( std::istream& file );

protected:
   using HostMatrix = typename Matrix::template Self< RealType, TNL::Devices::Host >;
};

// This is to prevent from appearing in Doxygen documentation.
/// \cond
template< typename Matrix >
class MatrixReader< Matrix, TNL::Devices::Host >
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
    * \brief Method for importing matrix from file with given filename.
    *
    * \param fileName is the name of the source file.
    * \param matrix is the target matrix.
    * \param verbose controls verbosity of the matrix import.
    *
    * \par Example
    * \include Matrices/MatrixWriterReaderExample.cpp
    * \par Output
    * \include Matrices/MatrixWriterReaderExample.out
    *
    */
   static void
   readMtx( const std::string& fileName, Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for importing matrix from STL input stream.
    *
    * \param file is the input stream.
    * \param matrix is the target matrix.
    * \param verbose controls verbosity of the matrix import.
    */
   static void
   readMtx( std::istream& file, Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for importing matrix from file with given filename.
    *
    * \param fileName is the name of the source file.
    * \param matrix is the target matrix.
    * \param options are the options of the import, see \ref MtxReaderOptions.
    */
   static void
   readMtx( const std::string& fileName, Matrix& matrix, const MtxReaderOptions& options );

   /**
    * \brief Method for importing matrix from STL input stream.
    *
    * \param file is the input stream.
    * \param matrix is the target matrix.
    * \param options are the options of the import, see \ref MtxReaderOptions.
    */
   static void
   readMtx( std::istream& file, Matrix& matrix, const MtxReaderOptions& options );

   /**
    * \brief Check if the MTX file with given filename declares a symmetric matrix.
    *
    * Only the header of the file is read, i.e. the symmetry is the one declared by the
    * header line `%%MatrixMarket matrix coordinate <field> symmetric`, the matrix elements
    * are not checked.
    *
    * \param fileName is the name of the source file.
    * \return `true` if the header declares a symmetric matrix and `false` if it declares
    * a general or skew-symmetric one.
    */
   [[nodiscard]] static bool
   isSymmetric( const std::string& fileName );

   /**
    * \brief Check if the MTX file in STL input stream declares a symmetric matrix.
    *
    * The header is read from the beginning of the stream.
    *
    * \param file is the input stream.
    * \return `true` if the header declares a symmetric matrix and `false` if it declares
    * a general or skew-symmetric one.
    */
   [[nodiscard]] static bool
   isSymmetric( std::istream& file );

protected:
   struct MtxHeader
   {
      IndexType rows = 0;
      IndexType columns = 0;
      IndexType elements = 0;
      bool symmetric = false;
      bool skewSymmetric = false;
      bool pattern = false;
   };

   // Reads the banner and the size line from the beginning of the stream. The stream is left at the first line after the
   // size line.
   static MtxHeader
   readMtxHeader( std::istream& file, long& lineNumber );

   // Reads the next matrix element from the stream, skipping comments and blank lines. The elements of a symmetric or
   // skew-symmetric matrix are returned below the diagonal. Returns false at the end of the stream.
   static bool
   readMatrixElement(
      std::istream& file,
      const MtxHeader& header,
      std::string& line,
      long& lineNumber,
      IndexType& row,
      IndexType& column,
      RealType& value );

   [[noreturn]] static void
   throwDuplicateElementError( std::istream& file, IndexType row, IndexType column );

   static void
   readMatrixElements(
      std::istream& file,
      const MtxHeader& header,
      long& lineNumber,
      Matrix& matrix,
      const MtxReaderOptions& options );
};
/// \endcond

}  // namespace TNL::Matrices

#include <TNL/Matrices/MatrixReader.hpp>
