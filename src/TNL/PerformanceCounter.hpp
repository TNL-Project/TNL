// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/PerformanceCounter.hpp>

namespace TNL {

#ifdef __APPLE__
#include <dlfcn.h>

inline void* PerformanceCounter::kperf = nullptr;
inline int ( *PerformanceCounter::kpc_set_config )( uint32_t, void* ) = nullptr;
inline int ( *PerformanceCounter::kpc_force_all_ctrs_set )( int ) = nullptr;
inline int ( *PerformanceCounter::kpc_set_counting )( int ) = nullptr;
inline int ( *PerformanceCounter::kpc_set_thread_counting )( int ) = nullptr;
inline int ( *PerformanceCounter::kpc_get_counter_count )( int ) = nullptr;
inline int ( *PerformanceCounter::kpc_get_config_count )( int ) = nullptr;
inline int ( *PerformanceCounter::kpc_get_thread_counters )( int, unsigned int, void * ) = nullptr;

inline TNL::Containers::Array< uint64_t > PerformanceCounter::appleKperfCounters;
inline TNL::Containers::Array< uint64_t > PerformanceCounter::appleKperfConfig;
#endif

inline PerformanceCounter::PerformanceCounter()
{
#ifdef __APPLE__
   bool error = false;
   if( ! kperf ) {
      const char* kperf_file = "/System/Library/PrivateFrameworks/kperf.framework/kperf";
      void *kperf = dlopen( kperf_file, RTLD_LAZY); // TODO: check the path in cmake
      if( ! kperf ) {
         std::cerr << "Unable to read kperf DLL " << kperf_file << std::endl;
         error = true;
      }
      else
      {
         kpc_set_config          = ( int(*)( uint32_t, void* ) )           dlsym( kperf, "kpc_set_config" );
         kpc_force_all_ctrs_set  = ( int(*) ( int ) )                      dlsym( kperf, "kpc_force_all_ctrs_set" );
         kpc_set_counting        = ( int(*) ( int ) )                      dlsym( kperf, "kpc_set_counting" );
         kpc_set_thread_counting = ( int(*) ( int ) )                      dlsym( kperf, "kpc_set_thread_counting" );
         kpc_get_counter_count   = ( int(*) ( int ) )                      dlsym( kperf, "kpc_get_counter_count" );
         kpc_get_config_count    = ( int(*) ( int ) )                      dlsym( kperf, "kpc_get_config_count" );
         kpc_get_thread_counters = ( int(*) ( int, unsigned int, void* ) ) dlsym( kperf, "kpc_get_thread_counters" );
         if( !kpc_set_config || !kpc_force_all_ctrs_set || !kpc_set_counting || !kpc_set_thread_counting ||
               !kpc_get_counter_count || !kpc_get_config_count || !kpc_get_thread_counters ) {
            std::cerr << "Unable to get all kperf functions." << std::endl;
            error = true;
         }
         int appleKperfCountersCount = kpc_get_counter_count(KPC_MASK);
         if( appleKperfCountersCount < 2 ) {
            std::cerr << "Wrong number of counters count " << appleKperfCountersCount << " for macOS kperf." <<std::endl;
            error = true;
         }
         int appleKperfConfigCount = kpc_get_config_count(KPC_MASK);
         if ( appleKperfConfigCount < 6 ) {
            std::cerr << "Wrong number of config count " << appleKperfConfigCount << " for macOS kperf." << std::endl;
            error = true;
         }
         else
         {
            appleKperfCounters.setSize( appleKperfCountersCount );
            appleKperfConfig.setSize( appleKperfConfigCount );
            appleKperfConfig[0] = CPMU_CORE_CYCLE | CFGWORD_EL0A64EN_MASK;
            appleKperfConfig[3] = CPMU_INST_BRANCH | CFGWORD_EL0A64EN_MASK;
            appleKperfConfig[4] = CPMU_SYNC_BR_ANY_MISP | CFGWORD_EL0A64EN_MASK;
            appleKperfConfig[5] = CPMU_INST_A64 | CFGWORD_EL0A64EN_MASK;

            if (kpc_set_config(KPC_MASK, appleKperfConfig.getData() )) {
               std::cerr << "kpc_set_config failed" << std::endl;
               error = true;
            }

            if (kpc_force_all_ctrs_set(1)) {
               std::cerr << "kpc_force_all_ctrs_set failed" << std::endl;
               error = true;
            }

            if (kpc_set_counting(KPC_MASK)) {
               std::cerr << "kpc_set_counting failed" << std::endl;
               error = true;
            }

            if (kpc_set_thread_counting(KPC_MASK)) {
               std::cerr << "kpc_set_thread_counting failed" << std::endl;
               error = true;
            }
         }
      }
   }
   if( error )
   {
      std::cerr << "Error occured during inicialization of kperf, administrative access might be required - use sudo command." << std::endl;
      std::cerr << "Measuring CPU cycles will not be possible." << std::endl;
   }
#endif
}

inline unsigned long long int
PerformanceCounter::getCPUCycles()
{
#ifdef _MSC_VER
   return 0;
#elif defined ( __APPLE__ )

   if( kpc_get_thread_counters != nullptr )
   {
      kpc_get_thread_counters(0, ( unsigned int ) appleKperfCounters.getSize(), ( void* ) appleKperfCounters.getData() );
      //   std::cerr << "Cannot read kperf counters." << std::endl;
      return appleKperfCounters[ 2 ];
   }
#else
   return rdtsc();
#endif
   return 0;
}

inline unsigned long long int
PerformanceCounter::rdtsc()
{
#if ! defined( __APPLE__ ) && ! defined( _MSC_VER )
   unsigned hi;
   unsigned lo;
   __asm__ __volatile__( "rdtsc" : "=a"( lo ), "=d"( hi ) );
   return ( (unsigned long long) lo ) | ( ( (unsigned long long) hi ) << 32 );
#else
   return 0;
#endif
}

} // namespace TNL
