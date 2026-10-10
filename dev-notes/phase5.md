# Phase 5: the subsonic airfoil

Companion to the plan's Phase 5 checklist. Closed on Oct 10, 2026, by decision, on a three-grid
study at 32, 64 and 128 cells per chord. Part 1 records the setup, part 2 the results, part 3 the
pressure comparison, part 4 the trailing-edge reports, and part 5 what carries forward.

New code:

- `naca4_points(...; trailing_edge=:extended)`: Vassberg and Jameson's sharp trailing edge.
- `wall_forces`' `cm` is now nose-up positive.
- `examples/naca0012_study.jl`, the resumable study, and `examples/naca0012_flow.jl` on the
  reference geometry.

Tests: the geometry against the paper's numbers, and the sign of the pitching moment.

## 1. The setup

- **The reference.** Vassberg and Jameson, "In Pursuit of Grid Convergence for Two-Dimensional
  Euler Solutions", J. Aircraft 47(4), 2010: NACA 0012 at M 0.5 and 1.25 degrees.
  - Their Table 2, FLO82 extrapolated to the continuum: cl 0.180346, cd 5e-8, cm -0.002269.
  - On their finest grid (4096^2, about 16.8M cells): cl 0.180348, cm -0.002269.
  - The minus signs of the moment are lost in the PDF's text layer; the paper's figure gives the
    continuum moment as -0.002268812.
- **The geometry.** Theirs, not the usual sharpened section: the standard thickness formula with
  its chord extended to where it closes, x = 1.0089304115. The reference chord stays 1 and the
  moment is about x = 0.25.
  - `trailing_edge=:extended` reproduces their edge position (to 1.4e-10, the precision they
    print), their edge slope and their peak thickness.
  - At 32 cells per chord it lifts 5% more than our earlier section sharpened with the -0.1036
    coefficient, so matching the geometry mattered.
- **The mesh.** The airfoil is turned nose-up 1.25 degrees about its leading edge, with the
  freestream along x.
  - The finest box extends half a chord round the airfoil.
  - The trailing edge sits 0.3 h into its cell, on the middle line of its row, on every grid, so
    the cut topology there is the same on every grid.
  - The nested far field reaches 100 chords or more, with 8-cell rings. Their O-meshes reach
    about 150 chords.
  - Moving the box and the far field mattered little: 0.2% in lift at 32 cells per chord, against
    the box of `naca0012_flow.jl`.
- **The solver.** HLLC, the mirror-state slip wall, the characteristic farfield without a vortex
  correction, the degree-4 rules on cut cells and 2 x 2 on rectangles, local steps at CFL 3, and
  10 orders of convergence.

## 2. Results

| Cells per chord | Elements | Iterations | Time | cl (error) | cd | cm (error) | Entropy L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 9,360 | 206,191 | 5 min | 0.16768 (-7.0%) | +1.2e-3 | -0.00136 (40%) | 7.2e-3 |
| 64 | 18,065 | 414,871 | 18 min | 0.17058 (-5.4%) | -1.3e-3 | -0.00149 (34%) | 3.5e-3 |
| 128 | 46,105 | 502,442 | 50 min | 0.17730 (-1.7%) | -1.4e-5 | -0.00204 (10%) | 9.9e-4 |

Observed orders:

| Grids | cl error | cm error | Entropy L2 |
| --- | --- | --- | --- |
| 32 to 64 | 0.37 | 0.22 | 1.05 |
| 64 to 128 | 1.68 | 1.77 | 1.82 |

- **Asymptotic range.** From 64 to 128 the errors fall at about second order, where from 32 to 64
  they barely moved. The leading-edge radius, 1.6% of the chord, spans half a cell at 32 cells
  per chord and two at 128. Vassberg and Jameson report the same low early orders for lift and
  moment on their own meshes.
- **Extrapolation.** Assuming second order, the 64-128 pair gives cl about 0.1795 (-0.45%) and cm
  about -0.00222 (-2%), without using the reference. The three grids together are not in the
  asymptotic range: the change from 32 to 64 is smaller than from 64 to 128.
- **Spurious drag** falls to -1.4e-5 at 128 cells per chord.
- **Iterations** grow slowly here: 2.0x from 32 to 64 and 1.2x from 64 to 128. On the circle they
  grew about 3x per refinement.
- **No limit cycle**, unlike the 64-cell circle.

## 3. The pressure comparison

The reference pressures are Martin Hegedus's solution on Vassberg's 4096^2 grid, from Hegedus
Aerodynamics (www.hegedusaero.com/examples/Vassberg/, file `m050a0125_char_mom/surface_04097.dat`;
the author asks for credit in any public use). It holds 4,097 surface points with Cp.

Ours and theirs were averaged in bins of x/c, and the load and its quarter-chord moment summed bin
by bin. Our Cp minus theirs on the upper surface:

| x/c | 32 | 64 | 128 |
| --- | --- | --- | --- |
| 0.035-0.05 | +0.071 | +0.024 | +0.007 |
| 0.10-0.15 | +0.019 | +0.010 | +0.003 |
| 0.30-0.40 | +0.007 | +0.005 | +0.001 |
| 0.70-0.80 | +0.005 | +0.002 | -0.001 |
| 1.00-1.009, the trailing-edge spike (upper/lower) | -0.21/-0.20 | -0.15/-0.13 | -0.10/-0.08 |

- **The lift deficit was circulation.** It showed as an even offset along the upper surface, with
  the lower surface close throughout (within 0.002 at 128). The offset shrinks about 4x from 64 to
  128.
- **The nose is resolved at 128.** The swings of 0.1-0.2 at 32 and 64 below x/c = 0.02 are down to
  0.01-0.05, which is within the bin averaging of a steep stagnation gradient.
- **The trailing-edge stagnation spike is the last gap.**
  - The reference's Cp falls from 0.98 at the edge to 0.71 within a few millionths of a chord.
  - Ours closes only about 1.5x per refinement, as expected for a singular point a cell or so
    wide.
  - It carries little load: under 0.0002 of the lift at 128.
- **The binned moment agrees.** At 128 the bin sums give -0.00161 against the reference's -0.00167;
  at 64, -0.00116. These bin sums leave out the chordwise forces, for both sets alike, which is
  why they differ from the exact values.

## 4. The trailing-edge reports

The reports are in `out/naca0012_study/c*_te.txt`. On all three grids:

- the edge is a mesh vertex at the polygon's own point;
- the wall length equals the polygon's perimeter to every printed digit;
- the cell just ahead of the edge is split into an upper and a lower fragment, with volume
  fractions 0.37 and 0.41, and no face between them;
- the edge's own cell is one 9-vertex fragment with volume fraction 0.987 and 4 wall faces.

Other numbers by grid:

| Cells per chord | Wedge angle | Smallest volume fraction (elsewhere on the surface) |
| --- | --- | --- |
| 32 | 16.07° | 5.2e-3 |
| 64 | 16.11° | 4.2e-4 |
| 128 | 16.13° | 1.8e-5 |

The wedge angle approaches the 16.15° of the paper's edge slope as the polygon's last segment
shortens. No tip is truncated: the exact cut keeps the polygon's own wall pieces.

## 5. Carried forward

- **The pitching moment's sign.** `wall_forces`' `cm` was counter-clockwise positive, which with
  the flow along x is nose-down. It read +0.00149 at 64 cells per chord where the aerodynamic
  convention gives -0.00149.
  - It is now nose-up positive, the convention references use, and the raw `moment` stays
    counter-clockwise.
  - The study's summary flips the sign of the runs saved before the change.
- **Not done:**
  - The point-vortex farfield correction, which was optional. At 100 chords it is worth roughly
    0.5% in lift, so it matters for comparisons tighter than the present 1.7%.
  - The run-configuration format.
  - A 256-cell grid: about 150k elements and an estimated 3.5 hours explicitly.
- **Cost points to Phase 6.** The 128-cell grid took 50 minutes. Finer grids, and the 64-cell
  circle's limit cycle, are the case for the implicit solver.
