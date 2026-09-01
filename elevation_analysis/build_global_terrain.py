#!/usr/bin/env python3
"""Fuse robot-centred 17x17 elevation patches into a world-XY terrain grid.

Heights are relative to the centre terrain in the first CSV row, so the
unknown constant vertical datum cancels.  Zero-valued cells in the output are
the requested flat default; use the accompanying count grid to identify cells
that were never observed.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


GRID_SIZE = 17
CENTER = GRID_SIZE // 2


def elevation_columns(fieldnames):
    columns = []
    for index in range(GRID_SIZE * GRID_SIZE):
        name = "elevation_map" if index == 0 else f"elevation_map.{index}"
        if name not in fieldnames:
            raise ValueError(f"Missing expected CSV column: {name}")
        columns.append(name)
    return columns


def local_cells(cell_resolution, flip_row, flip_col):
    """Return (row, col, local_x, local_y), with columns = +x and rows = +y."""
    cells = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = (col - CENTER) * cell_resolution
            y = (row - CENTER) * cell_resolution
            cells.append((row, col, -x if flip_col else x, -y if flip_row else y))
    return cells


def read_observations(csv_path, cell_resolution, flip_row, flip_col):
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"pos_x", "pos_y", "pos_z", "yaw"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing pose columns: {', '.join(sorted(missing))}")
        elevation_names = elevation_columns(reader.fieldnames)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV has no data rows")

    initial_reference = float(rows[0][elevation_names[CENTER * GRID_SIZE + CENTER]]) - float(rows[0]["pos_z"])
    observations = []
    for sample in rows:
        base_x, base_y = float(sample["pos_x"]), float(sample["pos_y"])
        base_z, yaw = float(sample["pos_z"]), float(sample["yaw"])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        for row, col, local_x, local_y in local_cells(cell_resolution, flip_row, flip_col):
            elevation = float(sample[elevation_names[row * GRID_SIZE + col]])
            world_x = base_x + cos_yaw * local_x - sin_yaw * local_y
            world_y = base_y + sin_yaw * local_x + cos_yaw * local_y
            # Subtracting this reference removes the unknown constant datum C.
            height = (elevation - base_z) - initial_reference
            observations.append((world_x, world_y, height))
    return observations, initial_reference


def fuse(observations, resolution):
    xs = np.array([point[0] for point in observations])
    ys = np.array([point[1] for point in observations])
    origin_x = math.floor(xs.min() / resolution) * resolution
    origin_y = math.floor(ys.min() / resolution) * resolution
    cols = int(round((xs.max() - origin_x) / resolution)) + 1
    rows = int(round((ys.max() - origin_y) / resolution)) + 1
    total = np.zeros((rows, cols), dtype=float)
    count = np.zeros((rows, cols), dtype=np.int32)

    for world_x, world_y, height in observations:
        col = int(round((world_x - origin_x) / resolution))
        row = int(round((world_y - origin_y) / resolution))
        total[row, col] += height
        count[row, col] += 1

    terrain = np.zeros_like(total)  # Requested default: unobserved terrain is flat.
    observed = count > 0
    terrain[observed] = total[observed] / count[observed]
    return terrain, count, origin_x, origin_y


def save_plot(terrain, count, origin_x, origin_y, resolution, input_csv, output_path):
    rows, cols = terrain.shape
    extent = [origin_x, origin_x + (cols - 1) * resolution,
              origin_y, origin_y + (rows - 1) * resolution]
    image = np.ma.masked_where(count == 0, terrain)
    cmap = plt.colormaps["terrain"].copy()
    cmap.set_bad("lightgray")
    with input_csv.open(newline="") as file:
        poses = list(csv.DictReader(file))

    plt.figure(figsize=(10, 8))
    plt.imshow(image, origin="lower", extent=extent, cmap=cmap, aspect="equal")
    plt.plot([float(pose["pos_x"]) for pose in poses],
             [float(pose["pos_y"]) for pose in poses], "k-", linewidth=1.2, label="Robot path")
    plt.plot(float(poses[0]["pos_x"]), float(poses[0]["pos_y"]), "ko", markersize=4, label="Start")
    plt.colorbar(label="Height relative to initial centre terrain [m]")
    plt.xlabel("World x [m]")
    plt.ylabel("World y [m]")
    plt.title("Fused global terrain (gray = unobserved)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_surface_plot(terrain, count, origin_x, origin_y, resolution, input_csv, output_path):
    rows, cols = terrain.shape
    x = origin_x + np.arange(cols) * resolution
    y = origin_y + np.arange(rows) * resolution
    world_x, world_y = np.meshgrid(x, y)
    height = np.ma.masked_where(count == 0, terrain)
    terrain_cmap = LinearSegmentedColormap.from_list(
        "brown_terrain", ["#6b3e1e", "#a66b3d", "#c99b62", "#d8c58f"]
    )
    with input_csv.open(newline="") as file:
        poses = list(csv.DictReader(file))
    trajectory_x = np.array([float(pose["pos_x"]) for pose in poses])
    trajectory_y = np.array([float(pose["pos_y"]) for pose in poses])
    trajectory_z = np.array([float(pose["pos_z"]) for pose in poses])
    trajectory_col = np.rint((trajectory_x - origin_x) / resolution).astype(int).clip(0, cols - 1)
    trajectory_row = np.rint((trajectory_y - origin_y) / resolution).astype(int).clip(0, rows - 1)
    projected_z = terrain[trajectory_row, trajectory_col].copy()
    projected_z[count[trajectory_row, trajectory_col] == 0] = np.nan

    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_subplot(projection="3d")
    surface = axis.plot_surface(world_x, world_y, height, cmap=terrain_cmap,
                                rcount=rows, ccount=cols, linewidth=0)
    axis.plot(trajectory_x, trajectory_y, trajectory_z,
              color="black", linewidth=1.5, label="Recorded robot height")
    axis.plot(trajectory_x, trajectory_y, projected_z + resolution,
              color="tab:blue", linewidth=1.5, label="Terrain-projected height")
    axis.scatter(trajectory_x[0], trajectory_y[0], trajectory_z[0],
                 color="black", s=24, label="Start")
    figure.colorbar(surface, ax=axis, label="Relative height [m]")
    axis.set(xlabel="World x [m]", ylabel="World y [m]", zlabel="Height [m]")
    z_values = np.concatenate((height.compressed(), trajectory_z, projected_z[np.isfinite(projected_z)]))
    axis.set_box_aspect((np.ptp(x), np.ptp(y), max(np.ptp(z_values), resolution)))
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input robot-state CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("global_terrain"))
    parser.add_argument("--global-resolution", type=float, default=0.04)
    parser.add_argument("--local-resolution", type=float, default=0.04)
    parser.add_argument("--flip-row", action="store_true", help="Use if increasing CSV rows mean -local-y")
    parser.add_argument("--flip-col", action="store_true", help="Use if increasing CSV columns mean -local-x")
    args = parser.parse_args()
    if args.global_resolution <= 0 or args.local_resolution <= 0:
        parser.error("resolutions must be positive")

    observations, reference = read_observations(args.csv, args.local_resolution, args.flip_row, args.flip_col)
    terrain, count, origin_x, origin_y = fuse(observations, args.global_resolution)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "terrain_height.csv", terrain, delimiter=",", fmt="%.6f")
    np.savetxt(args.output_dir / "observation_count.csv", count, delimiter=",", fmt="%d")
    metadata = {
        "input_csv": str(args.csv.resolve()),
        "origin_world_xy": [origin_x, origin_y],
        "global_resolution_m": args.global_resolution,
        "local_resolution_m": args.local_resolution,
        "shape_rows_cols": list(terrain.shape),
        "height_reference": "first snapshot centre terrain is 0 m",
        "initial_elevation_minus_pos_z_m": reference,
        "row_axis": "local +y" if not args.flip_row else "local -y",
        "column_axis": "local +x" if not args.flip_col else "local -x",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_plot(terrain, count, origin_x, origin_y, args.global_resolution, args.csv,
              args.output_dir / "terrain.png")
    save_surface_plot(terrain, count, origin_x, origin_y, args.global_resolution, args.csv,
                      args.output_dir / "terrain_3d.png")
    print(f"Wrote {terrain.shape[0]}x{terrain.shape[1]} terrain grid to {args.output_dir}")


if __name__ == "__main__":
    main()
