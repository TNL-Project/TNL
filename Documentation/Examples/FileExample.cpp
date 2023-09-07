#include <iostream>
#include <string>

#include <TNL/File.h>

using namespace TNL;

int main()
{
    File file;

    file.open( std::string("new-file.tnl"), std::ios_base::out );
    std::string title("'string to file'");
    file << title;
    file.close();

    file.open( std::string("new-file.tnl"), std::ios_base::in );
    std::string restoredString;
    file >> restoredString;
    file.close();

    std::cout << "restored string = " << restoredString << std::endl;
}
