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

## 4. Speed work (Oct 9)

**The profile.** The converged nested circle at 32 cells per diameter: 23,374 elements, 47,700
faces, 16 threads. Each kernel was timed on its own, per iteration:

| Kernel | Calls per iteration | Share |
| --- | --- | --- |
| Face fluxes | 3 | 44% |
| Element right-hand side | 3 | 29% |
| Face wave speeds | 1 | 15% |
| Stage updates | 3 | 6% |
| Positivity | 1 | 4% |
| Element steps, residual norms | 1 each | 2% |

- A whole iteration took 3.9 ms, 1 ms more than the parts timed alone. Inside an iteration the
  four state arrays and the 213,000 volume points no longer fit in cache.
- Garbage collection was 0%.
- The face pass moves about 400 bytes per face (basis values, normals, the two states, the
  result), about 19 MB per pass. That takes about its measured 0.4 ms at laptop bandwidth, so it is
  bound by memory, not arithmetic.

**Changes:**

1. **`box_degree`.** `volume_degree=4` was meant for cut fragments, where it removes the
   ear-clipping asymmetry, but it also gave every rectangle the 3 x 3 rule. Rectangles now take
   `box_degree`, 3 by default: 2 x 2 points, 4 instead of 9.
   - At 8 cells per diameter on the nested mesh the iterations are identical (50,987). cd, the
     stagnation Cp and the entropy agree to 5 digits.
   - The time per iteration falls 30%, from 2.02 ms to 1.43.
2. **One fused pass.** The steady solver's first right-hand side of each iteration
   (`_rhs_steady!`) also writes the face wave speeds, from the same traces, and the positivity
   flags, from the same volume points. That replaces the separate face-speed and positivity
   passes.
   - The results are bitwise those of `rhs!`, `local_dt!` and `assert_positive`. The saved
     reference runs, solve histories included, agree bit for bit.
   - At 32 cells per diameter an iteration fell from 3.9 to 3.3 ms.
   - The check now runs before the next iteration's fluxes instead of after the step. A state gone
     negative can therefore fail first in a flux's square root; `solve_steady!` catches that and
     throws the positivity error naming the element, for any stage's state.
3. **Errors out of kernels.** `_launch!` unwraps a failed task and throws its own error, such as a
   `DomainError`, rather than a nested `TaskFailedException`.

**The CFL number.** On the 8-cell nested circle, CFL 3 converges in 34,022 iterations against
50,987 at 2, with the same results. CFL 3.5 and 4 fail in the first few hundred iterations, at the
same element on the circle's shoulder, with a 2,000-iteration ramp too. CFL 3 then held on three
harder cases, each with the same forces as at 2:

| Case | Iterations at CFL 2 | Iterations at CFL 3 |
| --- | --- | --- |
| Circle, nested, 16 cells per diameter | 105,979 | 70,630 |
| NACA 0012, nested, 16 cells per chord | 113,602 | 75,765 |
| Circle with a 1e-12 sliver, 6 radii, 8 cells per diameter | 25,632 | 17,121 |

So `solve_steady!`'s default is now CFL 3; `integrate!`, time-accurate, keeps 2. The gate study's
saved runs were made at CFL 2, so their iteration counts are not comparable with new ones; their
steady states are.

**Together.** NACA 0012 at 16 cells per chord on the nested mesh went from 208 s (16-cell rings,
3 x 3 rule on rectangles, CFL 2) to 104 s. Wall times on this laptop vary by 10-20% from run to
run with thermal and core scheduling, so the iteration counts are the firmer measure.

**Not done, in rough order of payoff:**

- **The iteration count**, which grows about 3x per refinement on the nested mesh. The implicit
  solver is the real fix. Explicit options are implicit residual smoothing, multigrid over the
  nested levels, and low-Mach preconditioning of the pseudo-time derivative.
- **The face pass's memory traffic.** Store less per face, for instance basis values computed on
  the fly for faces between whole cells, and keep each face's data together.
- **Fusing the stage update into the element kernel**: about 5%.
