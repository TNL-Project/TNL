#include <iostream>
#include <TNL/Logger.h>

using namespace TNL;

int
main()
{
   Logger logger( 50, std::cout );

   logger.writeSystemInformation( false );

   logger.writeHeader( "MyTitle" );
   logger.writeSeparator();
   logger.writeSystemInformation( true );
   logger.writeSeparator();
}
