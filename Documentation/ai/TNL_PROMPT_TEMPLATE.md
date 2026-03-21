# TNL Prompt Template

Copy this template when asking for changes. Fill in placeholders so the assistant can produce correct, portable, and well-tested patches.

---
**Goal**: <concise feature/fix description>

**Target files/areas**: <list paths and relevant symbols>

**Behavior**: <expected behavior, inputs/outputs, edge cases>

**Backends**: <CPU/CUDA/HIP/OpenMP/MPI support expectations and any constraints>

**Performance/constraints**: <latency/throughput goals, memory limits, determinism requirements>

**Testing**: <tests to add/update; which backends to run; data sizes>

**Docs**: <Doxygen/Users’ Guide/pages to update; examples to add>

**Formatting/lint**: use project clang-format + clang-tidy (mention if to run scripts locally).

**Build config**: <CMake options in use, compilers, flags>

**Notes**: <existing patterns to mirror, related modules, compatibility concerns>

**Deliverables**:
- Code changes with brief rationale
- Tests (if applicable)
- Docs/examples (if applicable)
- Next-step suggestions (if anything remains)
