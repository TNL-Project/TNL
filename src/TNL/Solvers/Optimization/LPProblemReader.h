#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>

namespace TNL::Solvers::Optimization {
struct Constraint
{
   std::string name;
   char type;  // 'E' for equality, 'L' for less than, 'G' for greater than, 'N' for objective
};

struct ColumnEntry
{
   std::string colName;
   std::string rowName;
   double coefficient;
};

struct BoundEntry
{
   std::string boundType;
   std::string boundName;
   std::string variableName;
   double value;
};

struct RangeEntry
{
   std::string rangeName;
   std::string rowName;
   double value;
};

/**
 * \brief Reader of MPS files for linear programming problems.
 *
 * https://www.ibm.com/docs/en/cofz/12.9.0?topic=standard-records-in-mps-format
 *
 * \tparam LPProblem
 */
template< typename LPProblem >
struct LPProblemReader
{
   using LPProblemType = LPProblem;
   using MatrixType = typename LPProblemType::MatrixType;
   using VectorType = typename LPProblemType::VectorType;
   using RealType = typename LPProblemType::RealType;
   using IndexType = typename LPProblemType::IndexType;

   LPProblem
   read( const std::string& filename )
   {
      std::ifstream file( filename );
      return read( file );
   }

   LPProblem
   read( std::istream& in_stream )
   {
      std::string line;
      std::string section;
      std::vector< Constraint > constraints;
      std::vector< ColumnEntry > columns;
      std::vector< BoundEntry > bounds;
      std::vector< RangeEntry > ranges;
      std::map< std::string, RealType > rhs;
      std::map< std::string, IndexType > columnIndexes;
      std::map< std::string, IndexType > leInequalityRowIndexes;
      std::map< std::string, IndexType > geInequalityRowIndexes;
      std::map< std::string, IndexType > equalityRowIndexes;
      std::set< std::string > rangeRows;
      std::vector< std::pair< std::string, RealType > > objectiveFunction;
      std::string objectiveFunctionName;
      std::vector< std::string > variableNames_;

      IndexType nCols = 0;
      IndexType nInequalityRows = 0;
      IndexType nEqualityRows = 0;
      IndexType lineNumber = 0;
      while( std::getline( in_stream, line ) ) {
         lineNumber++;
         if( line.empty() || line[ 0 ] == '*' )
            continue;
         std::string word;
         std::istringstream iss( line );
         iss >> word;
         if( word == "NAME" || word == "ROWS" || word == "COLUMNS" || word == "RHS" || word == "BOUNDS" || word == "RANGES" ) {
            section = word;
            continue;
         }

         if( section == "ROWS" ) {
            std::istringstream iss( line );
            char type;
            std::string name;
            iss >> type >> name;
            if( type == 'L' )
               leInequalityRowIndexes[ name ] = nInequalityRows++;
            else if( type == 'G' )
               geInequalityRowIndexes[ name ] = nInequalityRows++;
            else if( type == 'E' )
               equalityRowIndexes[ name ] = nEqualityRows++;
            else if( type == 'N' )
               objectiveFunctionName = name;
            else {
               std::ostringstream oss;
               oss << "Error at line " << lineNumber << " unknown row type: '" << type << "'";
               std::cout << "Line: " << lineNumber << std::endl;
               throw std::runtime_error( oss.str() );
            }
         }
         else if( section == "COLUMNS" ) {
            std::istringstream iss( line );
            std::string colName, rowName;
            RealType coefficient;

            iss >> colName;
            while( iss >> rowName >> coefficient ) {
               if( columnIndexes.find( colName ) == columnIndexes.end() ) {
                  columnIndexes[ colName ] = nCols++;
                  variableNames_.push_back( colName );
               }
               if( rowName == objectiveFunctionName )
                  objectiveFunction.push_back( std::pair< std::string, RealType >( colName, coefficient ) );
               else
                  columns.push_back( { colName, rowName, coefficient } );
            }
         }
         else if( section == "RHS" ) {
            std::istringstream iss( line );
            std::string rhsName, consName;
            RealType value;
            iss >> rhsName;
            while( iss >> consName >> value ) {
               rhs[ consName ] = value;
            }
         }
         else if( section == "BOUNDS" ) {
            std::istringstream iss( line );
            BoundEntry bound;
            std::string boundType, boundName, variableName;
            RealType value;
            iss >> boundType >> boundName >> variableName >> value;
            bounds.push_back( { boundType, boundName, variableName, value } );
         }
         else if( section == "RANGES" ) {
            std::istringstream iss( line );
            std::string rangeName, rowName;
            RealType value;
            iss >> rangeName >> rowName >> value;
            ranges.push_back( { rangeName, rowName, value } );
            rangeRows.insert( rowName );
            if( leInequalityRowIndexes.find( rowName ) != leInequalityRowIndexes.end()
                || geInequalityRowIndexes.find( rowName ) != geInequalityRowIndexes.end() )
            {
               // To handle ranges, we need to convert inequality constraints to equality constraints
               // and add a new variable for the range ...
               std::string colName = "MPS_Rg" + rowName;
               if( leInequalityRowIndexes.find( rowName ) != leInequalityRowIndexes.end() ) {
                  leInequalityRowIndexes.erase( rowName );
                  columns.push_back( { colName, rowName, 1 } );
               }
               else {
                  geInequalityRowIndexes.erase( rowName );
                  columns.push_back( { colName, rowName, -1 } );
               }
               columnIndexes[ colName ] = nCols++;
               variableNames_.push_back( colName );
               equalityRowIndexes[ rowName ] = nEqualityRows;
               nInequalityRows--;
               nEqualityRows++;

               // ... and add the range constraint
               bounds.push_back( { "UP", "MPS_RgBOUND", colName, value } );
            }
         }
      }

      IndexType nRows = nInequalityRows + nEqualityRows;
      VectorType rhsVector( nRows, 0 );
      VectorType costFunction( nCols, 0 );
      VectorType lowerBounds( nCols, 0 );
      VectorType upperBounds( nCols, std::numeric_limits< RealType >::infinity() );

      IndexType inequalitiesOffset = this->inequalitiesFirst ? 0 : nEqualityRows;
      IndexType equalitiesOffset = this->inequalitiesFirst ? nInequalityRows : 0;

      std::map< std::pair< IndexType, IndexType >, RealType > matrixElements;
      for( const auto& col : columns ) {
         const IndexType colIndex = columnIndexes[ col.colName ];
         if( leInequalityRowIndexes.find( col.rowName ) != leInequalityRowIndexes.end() ) {
            const IndexType rowIndex = leInequalityRowIndexes[ col.rowName ] + inequalitiesOffset;
            matrixElements.insert( std::make_pair( std::make_pair( rowIndex, colIndex ), -col.coefficient ) );
         }
         else if( geInequalityRowIndexes.find( col.rowName ) != geInequalityRowIndexes.end() ) {
            const IndexType rowIndex = geInequalityRowIndexes[ col.rowName ] + inequalitiesOffset;
            matrixElements.insert(
               std::make_pair( std::make_pair( rowIndex, colIndex ), col.coefficient ) );  // TODO: check sign
         }
         else if( equalityRowIndexes.find( col.rowName ) != equalityRowIndexes.end() ) {
            const IndexType rowIndex = equalityRowIndexes[ col.rowName ] + equalitiesOffset;
            matrixElements.insert( std::make_pair( std::make_pair( rowIndex, colIndex ), col.coefficient ) );
         }
      }

      MatrixType constraintMatrix( nRows, nCols );
      constraintMatrix.template setElements< IndexType, RealType >( matrixElements );
      for( const auto& r : rhs ) {
         if( leInequalityRowIndexes.find( r.first ) != leInequalityRowIndexes.end() )
            rhsVector.setElement( leInequalityRowIndexes[ r.first ] + inequalitiesOffset, -r.second );
         else if( geInequalityRowIndexes.find( r.first ) != geInequalityRowIndexes.end() )
            rhsVector.setElement( geInequalityRowIndexes[ r.first ] + inequalitiesOffset, r.second );
         else if( equalityRowIndexes.find( r.first ) != equalityRowIndexes.end() )
            rhsVector.setElement( equalityRowIndexes[ r.first ] + equalitiesOffset, r.second );
         else {
            std::ostringstream oss;
            oss << "Unknown row name: " << r.first;
            throw std::runtime_error( oss.str() );
         }
      }

      Containers::Array< std::string > variableNames( variableNames_ );
      for( const auto& of : objectiveFunction )
         costFunction.setElement( columnIndexes[ of.first ], of.second );

      for( const auto& b : bounds ) {
         if( b.boundType == "UP" ) {
            upperBounds.setElement( columnIndexes[ b.variableName ], b.value );
            if( b.value <= 0 )
               lowerBounds.setElement( columnIndexes[ b.variableName ], -std::numeric_limits< RealType >::infinity() );
         }
         else if( b.boundType == "LO" ) {
            lowerBounds.setElement( columnIndexes[ b.variableName ], b.value );
         }
         else if( b.boundType == "FX" ) {
            upperBounds.setElement( columnIndexes[ b.variableName ], b.value );
            lowerBounds.setElement( columnIndexes[ b.variableName ], b.value );
         }
      }

      return LPProblem( constraintMatrix,
                        rhsVector,
                        max( equalitiesOffset, inequalitiesOffset ),
                        this->inequalitiesFirst,
                        costFunction,
                        lowerBounds,
                        upperBounds,
                        variableNames );
   }

   void
   setInequalitiesFirst( bool inequalitiesFirst )
   {
      this->inequalitiesFirst = inequalitiesFirst;
   }

protected:
   bool inequalitiesFirst = false;
};

}  // namespace TNL::Solvers::Optimization
