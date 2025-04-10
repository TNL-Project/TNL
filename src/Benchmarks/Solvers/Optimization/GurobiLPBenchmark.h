#include <iostream>
#include <vector>

#ifdef HAVE_GUROBI
   #include <gurobi_c++.h>
#endif

void
gurobiBenchmark( TNL::Benchmarks::Benchmark<>& benchmark, const TNL::String& fileName )
{
#ifdef HAVE_GUROBI
   // Initialize Gurobi environment
   GRBEnv env = GRBEnv();

   // Read the MPS file into a model
   GRBModel model = GRBModel( env, fileName.getData() );
   //model.write( "gurobi-model.lp" );
   //model.write( "gurobi-model.mps" );

   // Optimize the model
   auto f = [ &model ]()
   {
      model.optimize();
   };
   benchmark.time< TNL::Devices::Host >( "host", f );

   // Check the optimization result
   int optimstatus = model.get( GRB_IntAttr_Status );
   if( optimstatus == GRB_OPTIMAL ) {
      // Retrieve the number of variables
      int numvars = model.get( GRB_IntAttr_NumVars );

      // Get the variables
      auto vars = model.getVars();

      // Display the optimal values of the variables
      /*std::cout << "Optimal solution:" << std::endl;
      for( int i = 0; i < numvars; ++i ) {
         std::cout << vars[ i ].get( GRB_StringAttr_VarName ) << " = " << vars[ i ].get( GRB_DoubleAttr_X ) << std::endl;
      }*/
   }
   else if( optimstatus == GRB_INF_OR_UNBD ) {
      std::cout << "Model is infeasible or unbounded." << std::endl;
   }
   else if( optimstatus == GRB_INFEASIBLE ) {
      std::cout << "Model is infeasible." << std::endl;
   }
   else if( optimstatus == GRB_UNBOUNDED ) {
      std::cout << "Model is unbounded." << std::endl;
   }
   else {
      std::cout << "Optimization was stopped with status = " << optimstatus << std::endl;
   }
#endif
}
