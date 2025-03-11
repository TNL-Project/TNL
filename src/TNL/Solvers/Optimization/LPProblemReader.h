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

   static LPProblem
   read( const std::string& filename )
   {
      std::ifstream file( filename );

      std::string line;
      std::string section;
      std::vector< Constraint > constraints;
      std::vector< ColumnEntry > columns;
      std::vector< BoundEntry > bounds;
      std::map< std::string, RealType > rhs;
      std::map< std::string, IndexType > columnIndexes;
      std::map< std::string, IndexType > leInequalityRowIndexes;
      std::map< std::string, IndexType > geInequalityRowIndexes;
      std::map< std::string, IndexType > equalityRowIndexes;
      std::vector< std::pair< std::string, RealType > > objectiveFunction;
      std::string objectiveFunctionName;

      IndexType nCols = 0;
      IndexType nInequalityRows = 0;
      IndexType nEqualityRows = 0;
      while( getline( file, line ) ) {
         std::string word;
         std::istringstream iss( line );
         iss >> word;
         //std::cout << line << std::endl;
         if( word == "NAME" || word == "ROWS" || word == "COLUMNS" || word == "RHS" || word == "BOUNDS" ) {
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
            //constraints.push_back( { name, type } );
         }
         else if( section == "COLUMNS" ) {
            std::istringstream iss( line );
            std::string colName, rowName;
            RealType coefficient;
            while( iss >> colName >> rowName >> coefficient ) {
               if( columnIndexes.find( colName ) == columnIndexes.end() ) {
                  columnIndexes[ colName ] = nCols++;
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
            while( iss >> rhsName >> consName >> value ) {
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
      }

      file.close();

      // Print parsed data
      /*std::cout << "Constraints:\n";
      for( const auto& c : constraints ) {
         std::cout << c.name << " (" << c.type << ") -> ";
         if( c.type == 'L' )
            std::cout << leInequalityRowIndexes[ c.name ] << "\n";
         else if( c.type == 'G' )
            std::cout << geInequalityRowIndexes[ c.name ] << "\n";
         else if( c.type == 'E' )
            std::cout << equalityRowIndexes[ c.name ] + nInequalityRows << "\n";
      }

      std::cout << "\nColumns:\n";
      for( const auto& col : columns ) {
         std::cout << col.colName << "( " << columnIndexes[ col.colName ] << " ) "
                   << " affects " << col.rowName << " with coefficient " << col.coefficient << "\n";
      }

      std::cout << "\nRHS values:\n";
      for( const auto& r : rhs ) {
         std::cout << r.first << " = " << r.second << "\n";
      }*/

      IndexType nRows = nInequalityRows + nEqualityRows;
      VectorType rhsVector( nRows, 0 );
      VectorType costFunction( nCols, 0 );
      VectorType lowerBounds( nCols, -std::numeric_limits< RealType >::infinity() );
      VectorType upperBounds( nCols, std::numeric_limits< RealType >::infinity() );

      //std::cout << "Number of columns: " << nCols << std::endl;
      //std::cout << "Number of rows: " << nRows << std::endl;
      //std::cout << "Number of inequality rows: " << nInequalityRows << std::endl;
      //std::cout << "Number of equality rows: " << nEqualityRows << std::endl;

      std::map< std::pair< IndexType, IndexType >, RealType > matrixElements;
      for( const auto& col : columns ) {
         const IndexType colIndex = columnIndexes[ col.colName ];
         if( leInequalityRowIndexes.find( col.rowName ) != leInequalityRowIndexes.end() ) {
            const IndexType rowIndex = leInequalityRowIndexes[ col.rowName ];
            matrixElements.insert( std::make_pair( std::make_pair( rowIndex, colIndex ), -col.coefficient ) );
         }
         else if( geInequalityRowIndexes.find( col.rowName ) != geInequalityRowIndexes.end() ) {
            const IndexType rowIndex = geInequalityRowIndexes[ col.rowName ];
            matrixElements.insert(
               std::make_pair( std::make_pair( rowIndex, colIndex ), col.coefficient ) );  // TODO: check sign
         }
         else if( equalityRowIndexes.find( col.rowName ) != equalityRowIndexes.end() ) {
            const IndexType rowIndex = geInequalityRowIndexes[ col.rowName ] + nInequalityRows;
            matrixElements.insert( std::make_pair( std::make_pair( rowIndex, colIndex ), col.coefficient ) );
         }
      }

      MatrixType constraintMatrix( nRows, nCols );
      constraintMatrix.template setElements< IndexType, RealType >( matrixElements );
      for( const auto& r : rhs ) {
         if( leInequalityRowIndexes.find( r.first ) != leInequalityRowIndexes.end() )
            rhsVector.setElement( leInequalityRowIndexes[ r.first ], -r.second );
         else if( geInequalityRowIndexes.find( r.first ) != geInequalityRowIndexes.end() )
            rhsVector.setElement( geInequalityRowIndexes[ r.first ], r.second );
         else if( equalityRowIndexes.find( r.first ) != equalityRowIndexes.end() )
            rhsVector.setElement( equalityRowIndexes[ r.first ] + nInequalityRows, r.second );
      }
      for( const auto& of : objectiveFunction ) {
         costFunction.setElement( columnIndexes[ of.first ], of.second );
      }

      for( const auto& b : bounds ) {
         if( b.boundType == "UP" ) {
            upperBounds.setElement( columnIndexes[ b.variableName ], b.value );
         }
         else if( b.boundType == "LO" ) {
            lowerBounds.setElement( columnIndexes[ b.variableName ], b.value );
         }
         else if( b.boundType == "FX" ) {
            upperBounds.setElement( columnIndexes[ b.variableName ], b.value );
            lowerBounds.setElement( columnIndexes[ b.variableName ], b.value );
         }
      }

      //std::cout << "Cost function: " << costFunction << std::endl;
      //std::cout << "Constraint matrix: " << constraintMatrix << std::endl;
      //std::cout << "RHS vector: " << rhsVector << std::endl;
      //std::cout << "Lower bounds: " << lowerBounds << std::endl;
      //std::cout << "Upper bounds: " << upperBounds << std::endl;
      return LPProblem( constraintMatrix, rhsVector, nInequalityRows, costFunction );
   }
};

}  // namespace TNL::Solvers::Optimization
