#pragma once

#include <cstddef>
#include <cstdlib>

#include <TNL/Backend/DeviceInfo.h>

namespace TNL::UnitTests {

/**
 * \brief Returns true if the test runs in the CI environment.
 */
inline bool
isRunningInCi()
{
   return std::getenv( "GITLAB_CI" ) != nullptr;
}

/**
 * \brief Returns true if the requested device memory is currently available.
 *
 * Tests that allocate arrays of several GiB are skipped in CI on devices that
 * cannot provide the memory, instead of failing with an out-of-memory error which
 * would not indicate a bug in the tested code. Outside CI the tests always run, so
 * that a shortage of memory on a developer machine is not silently turned into a
 * skipped test.
 *
 * \param requiredBytes size of the array allocated by the test
 * \param auxiliaryBuffers number of additional arrays of the same size that the
 *        algorithm under test allocates internally
 */
inline bool
hasEnoughDeviceMemory( std::size_t requiredBytes, int auxiliaryBuffers = 0 )
{
   // the device context and the runtime take a few hundred MiB as well
   constexpr std::size_t margin = 512 * 1024 * 1024;
   const std::size_t required = requiredBytes * static_cast< std::size_t >( auxiliaryBuffers + 1 );
   return Backend::getFreeGlobalMemory() >= required + margin;
}

}  // namespace TNL::UnitTests
