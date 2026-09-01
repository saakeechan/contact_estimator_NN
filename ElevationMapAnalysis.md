# Elevation Map Vertical Reference Analysis

## Data

The robot has a 17 x 17 elevation grid (289 elevation values) at each
timestep. The table below focuses on the robot ground-truth vertical
position (`Pos_z`) and elevation values 145-148.

  --------------------------------------------------------------------------
           Pos_z  Elevation 145  Elevation 146  Elevation 147  Elevation 148
  -------------- -------------- -------------- -------------- --------------
         0.70644        0.20644        0.20644        0.20644        0.20644

         0.70397        0.20397        0.20397        0.20397        0.20397

         0.70280        0.20280        0.20280        0.20280        0.20280

         0.70416        0.20416        0.20416        0.20416        0.20416

         0.70724        0.20724        0.20724        0.20724        0.20724

         0.71047        0.21047        0.21047        0.21047        0.21047

         0.71364        0.21364        0.21364        0.21364        0.21364

         0.71659        0.21659        0.21659        0.21659        0.13626

         0.71937        0.21937        0.21937        0.21937        0.13903

         0.72232        0.22232        0.22232        0.14198        0.14198

         0.72602        0.22602        0.22602        0.14569        0.14569

         0.73028        0.23028        0.23028        0.14994        0.14994

         0.73312        0.23312        0.15278        0.15278        0.15278

         0.73189        0.23189        0.15155        0.15155        0.15155

         0.72912        0.22912        0.14879        0.14879        0.14879

         0.72753        0.14720        0.14720        0.14720        0.14720

         0.73087        0.15053        0.15053        0.15053        0.15053

         0.73751        0.15717        0.15717        0.15717        0.15717

         0.74647        0.16613        0.16613        0.16613        0.16613
  --------------------------------------------------------------------------

## Initial Observation: Unknown Vertical Datum

For the first several timesteps, the terrain appears flat because the
elevation values are identical across neighboring grid cells. However,
the common elevation value moves with the robot's `Pos_z`.

Initially,

`Elevation = Pos_z - 0.50000 m`

For example:

-   0.70644 - 0.20644 = 0.50000 m
-   0.71364 - 0.21364 = 0.50000 m
-   0.73312 - 0.23312 = 0.50000 m

This suggests that the elevation values contain an unknown vertical
datum/reference. The absolute elevation values cannot yet be interpreted
directly as world terrain height.

## Later Change in the Offset

A second relationship appears later:

`Elevation ~= Pos_z - 0.58034 m`

For example:

-   0.71659 - 0.13626 = 0.58033 m
-   0.73028 - 0.14994 = 0.58034 m
-   0.74647 - 0.16613 = 0.58034 m

Therefore, an approximately 8.03 cm difference appears between the two
regimes:

`0.58034 - 0.50000 ~= 0.08034 m`

Importantly, this transition propagates through adjacent elevation
columns:

`148 -> 147/148 -> 146/147/148 -> 145/146/147/148`

This suggests that the change may be spatial/index-related (for example,
grid serialization or circular-buffer behavior) rather than simply a
global vertical datum suddenly changing.

## Proposed Relative Terrain Reconstruction

Assume Elevation 145 is always the physical center cell of the
robot-centered grid.

Let:

-   `z(t)` = robot `Pos_z`
-   `e145(t)` = center-cell elevation
-   `H(t)` = terrain height at the robot center relative to the terrain
    height at the initial timestep

If the unknown vertical offset is constant, the absolute datum can be
eliminated by taking temporal differences:

`Delta H = Delta e145 - Delta z`

Equivalently, relative to the initial timestep:

`H(t) = [e145(t) - e145(t0)] - [z(t) - z(t0)]`

Example using the first two rows:

`Delta z = 0.70397 - 0.70644 = -0.00247 m`

`Delta e145 = 0.20397 - 0.20644 = -0.00247 m`

Therefore:

`Delta H = -0.00247 - (-0.00247) = 0`

This correctly indicates no terrain-height change if the robot is
walking over flat ground.

## Reconstructing the Rest of the Grid

Once the center terrain height `H(t)` is estimated, each other grid cell
can be represented relative to the center:

`h_i(t) = H(t) + [e_i(t) - e145(t)]`

The term

`e_i(t) - e145(t)`

describes the local terrain shape relative to the center cell, while
`H(t)` tracks how the terrain beneath the robot changes relative to the
initial terrain level.

## Critical Assumptions

This reconstruction is only valid if both assumptions below hold.

### 1. Elevation 145 is always the center cell

For a normally flattened 17 x 17 row-major grid, the center is the 145th
value (1-indexed):

`(9 - 1) * 17 + 9 = 145`

However, this does not prove that the exported 289 values use ordinary
logical row-major ordering. If the underlying elevation map uses a
circular buffer or another internal ordering, raw column 145 may not
always correspond to the physical center.

This assumption must be verified.

### 2. The unknown vertical offset is constant

The differencing approach assumes a model of the form:

`e145(t) = z(t) + H(t) + C`

where `C` is an unknown but constant vertical offset.

Then:

`Delta e145 = Delta z + Delta H`

and `C` cancels.

If instead the datum varies with time,

`e145(t) = z(t) + H(t) + C(t)`

then:

`Delta e145 - Delta z = Delta H + Delta C`

In that case, terrain-height changes and datum changes are
indistinguishable from these measurements alone.

The observed transition from an approximately 0.500 m offset to an
approximately 0.58034 m offset is therefore important and needs to be
explained before trusting the reconstruction.

## Current Interpretation

The early data is consistent with:

-   locally flat terrain,
-   elevation values containing robot vertical motion plus an unknown
    offset,
-   temporal differencing potentially recovering relative terrain-height
    changes.

However, the later approximately 8 cm transition means the
reconstruction should not yet be assumed valid over the full dataset.

The two main questions to resolve are:

1.  Does Elevation 145 always correspond to the physical center of the
    robot-centered grid?
2.  Is the vertical datum truly constant, or does the apparent offset
    change because of grid indexing/buffer behavior or another frame
    transformation?