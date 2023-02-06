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

inline int ( *PerformanceCounter::kpc_set_config )( uint32_t, void* ) = nullptr;
inline int ( *PerformanceCounter::kpc_force_all_ctrs_set )( int ) = nullptr;
inline int ( *PerformanceCounter::kpc_set_counting )( int ) = nullptr;
inline int ( *PerformanceCounter::kpc_set_thread_counting )( int ) = nullptr;
inline int ( *PerformanceCounter::kpc_get_counter_count )( int ) = nullptr;
inline int ( *PerformanceCounter::kpc_get_config_count )( int ) = nullptr;
inline int ( *PerformanceCounter::kpc_get_thread_counters )( int, unsigned int, void * ) = nullptr;
#endif

inline PerformanceCounter::PerformanceCounter()
{
#ifdef __APPLE__
   bool error = false;
   if( ! kperf ) {
      const char* kperf_file_name = "/System/Library/PrivateFrameworks/kperf.framework/kperf";
      void *kperf = dlopen( kperf_file_name, RTLD_LAZY); // TODO: check the path in cmake
      if( !kperf )
         std::cerr << "Unable to read kperf DLL " << kperf_file_name << std::endl;
      else {
         kpc_set_config          = ( int(*)( uint32_t, void* ) )           dlsym( kperf, "kpc_set_config" );
         kpc_force_all_ctrs_set  = ( int(*) ( int ) )                      dlsym( kperf, "kpc_force_all_ctrs_set" );
         kpc_set_counting        = ( int(*) ( int ) )                      dlsym( kperf, "kpc_set_counting" );
         kpc_set_thread_counting = ( int(*) ( int ) )                      dlsym( kperf, "kpc_set_thread_counting" );
         kpc_get_counter_count   = ( int(*) ( int ) )                      dlsym( kperf, "kpc_get_counter_count" );
         kpc_get_config_count    = ( int(*) ( int ) )                      dlsym( kperf, "kpc_get_config_count" );
         kpc_get_thread_counters = ( int(*) ( int, unsigned int, void* ) ) dlsym( kperf, "kpc_get_thread_counters" );
         if( !kpc_set_config || !kpc_force_all_ctrs_set || !kpc_set_counting || !kpc_set_thread_counting ||
             !kpc_get_counter_count || !kpc_get_config_count || !kpc_get_thread_counters ) {
            std::cerr << "Unable to get all kperf functions for macOS." << std::endl;
            error = true;
         }
         else
         {
            this->appleKpcCountersCount = kpc_get_counter_count(KPC_MASK);
            TNL_ASSERT_LE( appleKpcCountersCount, appleKpcCountersMaxCount, "Wrong number of macOS kperf counters." );
            TNL_ASSERT_LE( kpc_get_config_count(KPC_MASK), appleKpcConfigMaxCount, "Wrong config size of macOS kperf." );
            g_config[0] = CPMU_CORE_CYCLE | CFGWORD_EL0A64EN_MASK;
            g_config[3] = CPMU_INST_BRANCH | CFGWORD_EL0A64EN_MASK;
            g_config[4] = CPMU_SYNC_BR_ANY_MISP | CFGWORD_EL0A64EN_MASK;
            g_config[5] = CPMU_INST_A64 | CFGWORD_EL0A64EN_MASK;

            if( kpc_set_config(KPC_MASK, g_config) ||
                kpc_force_all_ctrs_set(1) ||
                kpc_set_counting(KPC_MASK) ||
                kpc_set_thread_counting(KPC_MASK) )
               error = true;
         }
      }
   }
   if( error )
      std::cerr << "Initiation of kperf failed, measuring CPU cycles will not be possible." << std::endl
                << "Note: Administrative access might be required. Try to run the program using sudo." << std::endl;
#endif
}

inline unsigned long long int
PerformanceCounter::getCPUCycles() const
{
#ifdef _MSC_VER
   return 0;
#elif defined ( __APPLE__ )
   if( kpc_get_thread_counters != nullptr )
   {
      kpc_get_thread_counters(0, appleKpcCountersCount, g_counters );
      return g_counters[ 2 ];
   }
   else return 0;
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
