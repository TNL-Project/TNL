# tests built only for one backend (GPU if available)
set(COMMON_TESTS
    ODESolverTest_Euler
    ODESolverTest_Merson
    # This is to test the static ODE solver for CUDA and HIP
    StaticODESolverTest_Merson
    StaticODESolverTest_BogackiShampin
    StaticODESolverTest_CashKarp
    StaticODESolverTest_DormandPrince
    StaticODESolverTest_Euler
    StaticODESolverTest_Fehlberg2
    StaticODESolverTest_Fehlberg5
    StaticODESolverTest_Heun2
    StaticODESolverTest_Heun3
    StaticODESolverTest_Kutta
    StaticODESolverTest_Midpoint
    StaticODESolverTest_OriginalRungeKutta
    StaticODESolverTest_Ralston2
    StaticODESolverTest_Ralston3
    StaticODESolverTest_Ralston4
    StaticODESolverTest_Rule38
    StaticODESolverTest_SSPRK3
    StaticODESolverTest_VanDerHouwenWray
)

foreach(test IN ITEMS ${COMMON_TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h BACKENDS CUDA HIP HOST SINGLE)
endforeach()
