#include <iostream>
#include <string>

#ifdef HAVE_ORTOOLS
   #include <ortools/linear_solver/linear_solver.h>
#endif

bool
ReadMpsFile( const std::string& file_name, operations_research::MPModelProto* model_proto )
{
   std::ifstream input( file_name, std::ios::in | std::ios::binary );
   if( ! input ) {
      std::cerr << "Unable to open file: " << file_name << std::endl;
      return false;
   }
   if( ! model_proto->ParseFromIstream( &input ) ) {
      std::cerr << "Failed to parse MPS file: " << file_name << std::endl;
      return false;
   }
   return true;
}

void
PrintModel( const operations_research::MPSolver& solver )
{
   // Export the model in LP format as a string
   std::string lp_model_str;
   if( solver.ExportModelAsLpFormat( false, &lp_model_str ) )
      std::cout << "Model in LP format:\n" << lp_model_str << std::endl;
   else
      std::cerr << "Failed to export the model in LP format." << std::endl;

   // Optionally, write the LP model to a file
   //std::ofstream lp_file( "model.lp" );
   //lp_file << lp_model_str;
   //lp_file.close();

   // Export the model in MPS format as a string
   /*std::string mps_model_str;
   if( solver.ExportModelAsMpsFormat( false, false, &mps_model_str ) )
      std::cout << "Model in MPS format:\n" << mps_model_str << std::endl;
   else
      std::cerr << "Failed to export the model in MPS format." << std::endl;*/

   // Print the MPS model to the console
   //std::cout << "Model in MPS format:\n" << mps_model_str << std::endl;

   // Optionally, write the MPS model to a file
   //std::ofstream mps_file( "model.mps" );
   //mps_file << mps_model_str;
   //mps_file.close();
}

template< typename LPProblem >
void
orToolsLPBenchmark( TNL::Benchmarks::Benchmark<>& benchmark, const LPProblem& lpProblem )
{
   using RealType = typename LPProblem::RealType;
   using IndexType = typename LPProblem::IndexType;
   // Create the linear solver with the PDLP backend.
   operations_research::MPSolver solver( "LinearProgramSolver", operations_research::MPSolver::GLOP_LINEAR_PROGRAMMING );
   //operations_research::MPSolver solver( "LinearProgramSolver", operations_research::MPSolver::PDLP_LINEAR_PROGRAMMING );

   // Load the MPS file.
   //const operations_research::MPSolver::ResultStatus load_status = solver.LoadModelFromFile( fileName.getData() );
   //if( load_status != operations_research::MPSolver::OK ) {
   //   std::cerr << "Error loading MPS file: " << filename << std::endl;
   //   return 1;
   //}

   // Initialize the protocol buffer
   //std::cout << "Reading the MPS file... OR Tools" << std::endl;
   //operations_research::MPModelProto model_proto;
   //if( ! ReadMpsFile( fileName.getData(), &model_proto ) ) {
   //   return;
   // }
   //std::string error_message;
   //if( ! solver.LoadModelFromProto( model_proto, &error_message ) ) {
   //   std::cerr << "Error loading model: " << error_message << std::endl;
   //   return;
   //}

   const auto infinity = solver.infinity();
   TNL::Containers::Vector< operations_research::MPVariable* > variables( lpProblem.getVariableCount() );
   for( int i = 0; i < lpProblem.getVariableCount(); ++i ) {
      variables[ i ] = solver.MakeNumVar(
         lpProblem.getLowerBounds()[ i ] == -std::numeric_limits< RealType >::infinity() ? -infinity
                                                                                         : lpProblem.getLowerBounds()[ i ],
         lpProblem.getUpperBounds()[ i ] == std::numeric_limits< RealType >::infinity() ? infinity
                                                                                        : lpProblem.getUpperBounds()[ i ],
         lpProblem.getVariableNames()[ i ].data() );
   }
   using RowView = typename LPProblem::MatrixType::ConstRowView;
   lpProblem.getConstraintMatrix().forAllRows(
      [ & ]( const RowView& row ) mutable
      {
         const IndexType i = row.getRowIndex();
         operations_research::MPConstraint* constraint;
         if( i < lpProblem.getInequalityCount() )
            constraint = solver.MakeRowConstraint( lpProblem.getConstraintVector()[ i ], infinity );
         else
            constraint = solver.MakeRowConstraint( lpProblem.getConstraintVector()[ i ], lpProblem.getConstraintVector()[ i ] );
         for( IndexType j = 0; j < row.getSize(); ++j ) {
            constraint->SetCoefficient( variables[ row.getColumnIndex( j ) ], row.getValue( j ) );
         }
      } );

   operations_research::MPObjective* const objective = solver.MutableObjective();
   for( IndexType i = 0; i < lpProblem.getObjectiveFunction().getSize(); ++i ) {
      objective->SetCoefficient( variables[ i ], lpProblem.getObjectiveFunction()[ i ] );
   }
   objective->SetMinimization();

   PrintModel( solver );

   // Solve the problem.
   std::cout << "Solving the problem using OR Tools..." << std::endl;
   const operations_research::MPSolver::ResultStatus result_status = solver.Solve();

   // Check the result.
   if( result_status == operations_research::MPSolver::OPTIMAL ) {
      std::cout << "Optimal objective value: " << solver.Objective().Value() << std::endl;
      const auto& variables = solver.variables();
      for( const auto* const var : variables ) {
         std::cout << var->name() << " = " << var->solution_value() << std::endl;
      }
   }
   else {
      std::cerr << "The problem does not have an optimal solution." << std::endl;
   }
}
