#include <iostream>
#include <string>

#include <TNL/Config/ConfigDescription.h>

using namespace TNL;
using namespace std;

int main()
{
    Config::ConfigDescription confd;
    confd.template addEntry< std::string >("--new-entry","Specific description.");
    confd.template addEntryEnum< std::string >("option1");
    confd.template addEntryEnum< std::string >("option2");
    confd.addDelimiter("-----------------------------");
}
