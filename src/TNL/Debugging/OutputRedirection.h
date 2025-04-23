// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>

#include <TNL/3rdparty/spy.hpp>

#ifdef SPY_SUPPORTS_POSIX
   #include <unistd.h>
#endif

namespace TNL::Debugging {

class OutputRedirection
{
   int backupFd = -1;
   int targetFd = -1;
   FILE* file = nullptr;

public:
   OutputRedirection() = delete;

   OutputRedirection( int targetFd )
   : targetFd( targetFd )
   {}

   bool
   redirect( const std::string& fname )
   {
#ifdef SPY_SUPPORTS_POSIX
      // restore the original stream if there is any backup
      if( backupFd >= 0 || file != nullptr )
         if( ! restore() )
            return false;

      // first open the file
      file = ::fopen( fname.c_str(), "w" );
      if( file == nullptr ) {
         std::cerr << "error: fopen() failed, output is not redirected.\n";
         return false;
      }

      // then backup the original file descriptors
      backupFd = ::dup( targetFd );
      if( backupFd < 0 ) {
         std::cerr << "error: dup() failed, output is not redirected.\n";
         return false;
      }

      // finally redirect stdout and stderr
      if( ::dup2( ::fileno( file ), targetFd ) < 0 ) {
         std::cerr << "error: dup2() failed, output is not redirected.\n";
         return false;
      }

      return true;
#else
      std::cerr << "Output redirection is supported only on POSIX systems.\n";
      return false;
#endif
   }

   bool
   restore()
   {
#ifdef SPY_SUPPORTS_POSIX
      // first restore the original file descriptor
      if( backupFd >= 0 ) {
         if( ::dup2( backupFd, targetFd ) < 0 ) {
            std::cerr << "error: dup2() failed, output is not restored.\n";
            return false;
         }
         backupFd = -1;
      }

      // then close the file
      if( file != nullptr ) {
         ::fclose( file );
         file = nullptr;
      }
      return true;
#else
      std::cerr << "Output redirection is supported only on POSIX systems.\n";
      return false;
#endif
   }

   ~OutputRedirection()
   {
      restore();
   }
};

inline bool
redirect_stdout_stderr( const std::string& stdout_fname, const std::string& stderr_fname, bool restore = false )
{
#ifdef SPY_SUPPORTS_POSIX
   static OutputRedirection stdoutRedir( STDOUT_FILENO );
   static OutputRedirection stderrRedir( STDERR_FILENO );

   if( ! restore ) {
      if( ! stdoutRedir.redirect( stdout_fname ) )
         return false;
      if( ! stderrRedir.redirect( stderr_fname ) )
         return false;
   }
   else {
      stdoutRedir.restore();
      stderrRedir.restore();
   }

   return true;
#else
   std::cerr << "Output redirection is supported only on POSIX systems.\n";
   return false;
#endif
}

}  // namespace TNL::Debugging
