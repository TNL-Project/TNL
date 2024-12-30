// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <limits>

#include <TNL/3rdparty/spy.hpp>

#ifndef SPY_OS_IS_WINDOWS
   #include <sys/ioctl.h>
   #include <unistd.h>
#endif

#include <TNL/Solvers/DirectSolver.h>

namespace TNL::Solvers {

template< typename Real, typename Index >
void
DirectSolverMonitor< Real, Index >::setStage( const std::string& stage )
{
   saved_stage = this->stage;

   this->stage = stage;
   saved = true;
   attributes_changed = true;
}

template< typename Real, typename Index >
void
DirectSolverMonitor< Real, Index >::setVerbose( const Index& verbose )
{
   this->verbose = verbose;
   attributes_changed = true;
}

template< typename Real, typename Index >
void
DirectSolverMonitor< Real, Index >::refresh()
{
   // NOTE: We can't check if stdout is attached to a terminal or not, because
   // isatty(STDOUT_FILENO) *always* reports 1 under mpirun, regardless if it
   // runs from terminal or under SLURM or PBS. Hence, we use the verbose flag
   // to determine interactivity: verbose=2 expects stdout to be attached to a
   // terminal, verbose=1 does not use terminal features (can be used with
   // stdout redirected to a file, or with a terminal) and verbose=0 disables
   // the output completely.

   if( this->verbose > 0 ) {
      // Check if we should display the current values or the values saved after
      // the previous stage. If the iterations cycle much faster than the solver
      // monitor refreshes, we display only the values saved after the whole
      // cycle to hide the irrelevant partial progress.
      const bool saved = this->saved;
      this->saved = false;

      const int line_width = getLineWidth();
      int free = line_width > 0 ? line_width : std::numeric_limits< int >::max();

      auto real_to_string = []( Real value, int precision = 6 )
      {
         std::stringstream stream;
         stream << std::setprecision( precision ) << value;
         return stream.str();
      };

      auto print_item = [ &free ]( const std::string& item, int width = 0 )
      {
         width = std::min( free, width > 0 ? width : (int) item.length() );
         std::cout << std::setw( width ) << item.substr( 0, width );
         free -= width;
      };

      if( this->verbose >= 2 )
         // \33[2K erases the current line, see https://stackoverflow.com/a/35190285
         std::cout << "\33[2K\r";
      else if( ! attributes_changed )
         // verbose == 1, attributes were not updated since the last refresh
         return;

      if( timer != nullptr ) {
         print_item( " ELA:" );
         print_item( real_to_string( getElapsedTime(), 5 ), 8 );
      }

      const std::string displayed_stage = ( saved ) ? saved_stage : stage;
      if( ! displayed_stage.empty() && free > 5 ) {
         if( (int) displayed_stage.length() <= free - 2 ) {
            std::cout << "  " << displayed_stage;
            free -= ( 2 + displayed_stage.length() );
         }
         else {
            std::cout << "  " << displayed_stage.substr( 0, free - 5 ) << "...";
            free = 0;
         }
      }

      elapsed_time_before_refresh = getElapsedTime();

      if( this->verbose >= 2 )
         // return to the beginning of the line
         std::cout << "\r" << std::flush;
      else
         // linebreak and flush
         std::cout << "\n" << std::flush;

      // reset the changed flag
      attributes_changed = false;
   }
}

template< typename Real, typename Index >
int
DirectSolverMonitor< Real, Index >::getLineWidth()
{
#ifndef SPY_OS_IS_WINDOWS
   struct winsize w;
   ioctl( STDOUT_FILENO, TIOCGWINSZ, &w );
   return w.ws_col;
#else
   return 0;
#endif
}

}  // namespace TNL::Solvers
