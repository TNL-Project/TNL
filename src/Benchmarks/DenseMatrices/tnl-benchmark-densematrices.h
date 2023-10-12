#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include "DenseMatricesBenchmark.h"
#include "CublasBenchmark.h"
#include "BlasBenchmark.h"

void configSetup(TNL::Config::ConfigDescription& config) {
    config.addDelimiter("Precision settings:");
    config.addEntry<TNL::String>("precision", "Precision of the arithmetics.", "double");
    config.addEntryEnum("float");
    config.addEntryEnum("double");
    config.addEntryEnum("all");
}

template <typename Real>
bool 
runDenseMatricesBenchmark(TNL::Config::ParameterContainer& parameters) {
    TNL::Benchmarks::DenseMatrices::DenseMatricesBenchmark<Real> benchmark(parameters);
    benchmark.runBenchmark();
    return true;
}

int main(int argc, char* argv[]) {
    TNL::Config::ConfigDescription config;
    configSetup(config);
    TNL::Devices::Host::configSetup(config);
    TNL::Devices::Cuda::configSetup(config);
    TNL::Benchmarks::DenseMatrices::DenseMatricesBenchmark<>::configSetup(config);

    TNL::Config::ParameterContainer parameters;

    if (!TNL::Config::parseCommandLine(argc, argv, config, parameters))
        return EXIT_FAILURE;

    if (!TNL::Devices::Host::setup(parameters) || !TNL::Devices::Cuda::setup(parameters))
        return EXIT_FAILURE;

    bool success = false;
    auto precision = parameters.getParameter<TNL::String>("precision");

    if (precision == "float" || precision == "all") {
        success = runDenseMatricesBenchmark<float>(parameters);
    } else if (precision == "double" || precision == "all") {
        success = runDenseMatricesBenchmark<double>(parameters);
    } else {
        std::cerr << "Unknown precision " << precision << "." << std::endl;
        return EXIT_FAILURE;
    }

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
