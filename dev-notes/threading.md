# Threading the solver with KernelAbstractions

Oct 9, 2026, after Phase 4. The solver's per-iteration loops now run as KernelAbstractions kernels
on the CPU backend, over all of Julia's threads (`julia -t`). Results are unchanged: they agree
bitwise with the serial code they replace, on any number of threads.

## 1. What changed

1. **A two-pass gather replaces the face scatter.**
   - The face kernel evaluates F̂ once at each face Gauss point, into a face buffer: the surface
     flux on interior faces, and the boundary's condition on boundary faces.
   - The element kernel sums each element's volume integral, source terms and face integrals,
     then divides by |K|.
   - Each element takes its faces in increasing order: the order in which the old face loops
     reached it, interior faces first and then each boundary in turn. That is why the sums are
     bitwise the old ones. The signed, sorted face lists are built with the semidiscretization
     (`sides`).
   - All boundaries share one face kernel. Each face's condition is picked from the tuple by
     unrolled recursion, so the call stays type-stable whatever the mix of conditions.
2. **Other kernels.**
   - The time step: face wave speeds, then a per-element gather into the speed and the step.
   - The positivity check: a per-element flag. The serial scan afterwards raises the same error
     as before, at the first bad element and its first bad point.
   - The SSP-RK3 stage updates, for global and local steps alike. The global step's updates used
     to be broadcasts; the stage kernel is bitwise the same.
   - `residual_norms` and `max_dt`'s minimum stay serial: one cheap pass over the elements.
3. **`DGSemidiscretization` is a `mutable struct` with `const` fields.**
   - The CPU backend boxes a launch's arguments: an inference barrier inside
     KernelAbstractions.
   - Boxing the 464-byte immutable struct copied it on every launch, about 1 KB.
   - Passed by reference, it costs nothing.
4. **Workgroup sizes are set per launch, through `grainsize`.**
   - The CPU backend gives each thread a contiguous run of whole workgroups, 1024 items each by
     default, so a 2,272-element mesh would use only three threads.
   - The workgroups now split the items evenly over the threads, with at least `grainsize` (64)
     items each and at most `n`. Past `n`, the backend steps through every empty slot, and
     `typemax(Int)` hung.
   - A launch over at most `grainsize` items runs on the calling thread, and
     `grainsize=typemax(Int)` runs everything serially.

## 2. Allocations

- Every kernel body is allocation-free. `test_kernels.jl` runs each body serially over all its
  items, on cut, half-plane (with source terms) and periodic meshes.
- A launch still allocates KernelAbstractions' own bookkeeping: the boxed arguments, the
  kernel object and the index range. On one thread that is about 150 bytes. It is independent of
  the mesh size from 512 items up, which the tests also check; below that, Julia's cached boxes
  of small integers save 16 bytes. With threads, each launch also allocates its tasks, about
  7.5 KB at 16 threads.
- A `solve_steady!` iteration makes 12 launches, so about 90 KB per iteration at 16 threads. The
  three `allocs(...) == 0` tests on `rhs!`, `max_dt` and `local_dt!` became the body and
  mesh-size tests.

## 3. Speed

The circle with the degree-4 rules, on an Intel Core Ultra 7 366H laptop (16 cores of three
kinds), in ms per `solve_steady!` iteration:

| Cells per diameter | Elements | 1 thread | 4 | 8 | 12 | 16 | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 2,272 | 1.6–1.9 | 0.87 | 0.71 | | 0.55 | about 3 |
| 16 | 9,052 | 5.5 | | 2.1 | 1.6 | 1.45 | 3.8 |

- `rhs!` alone scales a little better: 1.34 to 0.34 ms at 16 cells per diameter, 4.0×.
- A bare launch over 16 tasks costs about 14 µs.
- Single-thread timings vary by about 20% from run to run, depending on which kind of core the
  thread lands on.
- **The limit is the hardware, not the launches.**
  - The backend splits the work statically, so the slowest cores set the pace.
  - Dynamically scheduled chunks of the same bodies were 30% faster at 16 cells per diameter and
    slower at 8, which is within the noise.
- **Possible next steps,** if the explicit solver stays the workhorse:
  - Fuse the stage update into the element kernel.
  - Fuse the time step's face speeds into the first face pass of each iteration.
  - Fuse the positivity check into the volume loop.

  Together these halve the launches.
