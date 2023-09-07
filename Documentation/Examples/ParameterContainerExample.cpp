#include <iostream>
#include <string>

#include <TNL/Config/ParameterContainer.h>

using namespace TNL;

int main()
{
    Config::ParameterContainer parameters;
    auto param = parameters.getParameter< std::string >( "distributed-grid-io-type" );
//    parameters.checkParameter< std::string >( "distributed-grid-io-type" );
}
