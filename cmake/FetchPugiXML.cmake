include(FetchContent)
FetchContent_Declare(
    pugixml
    GIT_REPOSITORY https://github.com/zeux/pugixml.git
    GIT_TAG v1.14
    FIND_PACKAGE_ARGS NAMES pugixml
    SYSTEM
)
FetchContent_MakeAvailable(pugixml)
