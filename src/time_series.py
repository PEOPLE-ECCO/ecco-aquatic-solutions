"""
Time-series statistics on yearly habitat extent results.

Takes several result folders produced by the main SAV workflow (one per year,
over the same area), collects the binary habitat maps
(``<date>_aggregated_<habitat>.tif``), aligns them on a common grid if needed,
and writes per-pixel temporal statistics:

- ``frequency_<habitat>.tif``: percentage of years in which habitat was detected.
- ``lastyear_<habitat>.tif``: last year in which habitat was detected.
- ``tendency_<habitat>.tif``: change between the first and the last year
  (1 = appeared, -1 = disappeared, 0 = present in both).

Example
-------
From the repository root::

    python -m src.time_series \\
        --input-folders results/2021 results/2022 results/2023 \\
        --output-dir results/time_series
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.coords import BoundingBox
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

HABITAT_TYPES = ["sav", "coral"]


def find_hab_files(
    in_folders: list[Path], habitat_type: str
) -> tuple[list[Path], bool]:
    """
    Finds the binary habitat files in a list of result folders, and checks
    whether they all share the same grid.

    Parameters
    ----------
    in_folders : list[Path]
        Result folders, typically one per year.
    habitat_type : str
        Habitat type, "sav" or "coral".

    Returns
    -------
    tuple[list[Path], bool]
        Sorted list of the binary files found, and whether they are aligned
        (same crs, transform and shape).
    """
    files = []
    for afolder in in_folders:
        files.extend(Path(afolder).rglob(f"*aggregated_{habitat_type}.tif"))
    files = sorted(files, key=lambda f: f.name)

    if not files:
        raise FileNotFoundError(
            f"No '*aggregated_{habitat_type}.tif' file found in {in_folders}"
        )

    data_list = []
    for afile in files:
        with rasterio.open(afile) as src:
            data_list.append(
                {
                    "fname": afile.name,
                    "year": int(afile.name[0:4]),
                    "crs": src.crs.to_string(),
                    "transform": tuple(src.transform),
                    "shape": src.shape,
                }
            )
    df = pd.DataFrame(data_list).set_index("fname")

    duplicated = df["year"][df["year"].duplicated()].unique()
    if len(duplicated) > 0:
        raise ValueError(
            f"Several {habitat_type} files found for years {list(duplicated)}"
        )

    aligned = all(df[col].nunique() == 1 for col in ["crs", "transform", "shape"])
    if aligned:
        print(f"All {len(files)} {habitat_type} tifs aligned")
    else:
        print(f"{habitat_type} tifs are not on the same grid, they need to be aligned")

    return files, aligned


def align_files(
    file_list: list[Path],
    out_dir: Path,
    extent: str = "intersection",
) -> list[Path]:
    """
    Reprojects a list of binary rasters on a common grid and writes them as new
    files, keeping their original names.

    The grid uses the crs and resolution of the first file. Its extent is
    either the intersection (default) or the union of all file extents. Pixels
    outside a file's footprint are set to 0 (no habitat), so with a union
    extent, areas not covered every year will show as habitat losses or gains.

    Parameters
    ----------
    file_list : list[Path]
        Binary habitat files to align.
    out_dir : Path
        Folder where the aligned files are written.
    extent : str, optional
        "intersection" or "union", by default "intersection".

    Returns
    -------
    list[Path]
        Paths to the aligned files.
    """
    if extent not in ("intersection", "union"):
        raise ValueError("extent must be 'intersection' or 'union'")
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(file_list[0]) as ref:
        dst_crs = ref.crs
        res_x, res_y = ref.res

    # Bounds of all files in the reference crs
    all_bounds = []
    for afile in file_list:
        with rasterio.open(afile) as src:
            all_bounds.append(
                BoundingBox(*transform_bounds(src.crs, dst_crs, *src.bounds))
            )

    combine = (max, min) if extent == "intersection" else (min, max)
    left = combine[0](b.left for b in all_bounds)
    bottom = combine[0](b.bottom for b in all_bounds)
    right = combine[1](b.right for b in all_bounds)
    top = combine[1](b.top for b in all_bounds)
    if left >= right or bottom >= top:
        raise ValueError("The input files do not overlap")

    # Snap to the reference pixel size
    left = np.floor(left / res_x) * res_x
    top = np.ceil(top / res_y) * res_y
    width = int(np.ceil((right - left) / res_x))
    height = int(np.ceil((top - bottom) / res_y))
    dst_transform = from_origin(left, top, res_x, res_y)

    aligned_files = []
    for afile in file_list:
        out_file = out_dir / afile.name
        with rasterio.open(afile) as src:
            destination = np.zeros((height, width), dtype="int8")
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
        profile = {
            "driver": "COG",
            "dtype": "int8",
            "count": 1,
            "width": width,
            "height": height,
            "crs": dst_crs,
            "transform": dst_transform,
            "compress": "ZSTD",
        }
        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(destination, 1)
        aligned_files.append(out_file)

    print(
        f"Aligned {len(aligned_files)} files ({extent}, {width}x{height} px) in {out_dir}"
    )
    return aligned_files


def ts_metrics(file_list: list[Path], save_path: Path, habitat_type: str) -> None:
    """
    Creates and writes time series products.

    Parameters
    ----------
    file_list : list[Path]
        List of the binary files containing the habitat presence, on the same grid.
    save_path : Path
        Path to the folder where results are saved.
    habitat_type : str
        Name of the habitat type for output names.
    """
    if len(file_list) < 2:
        raise ValueError("At least two years are needed to compute time series metrics")
    save_path.mkdir(parents=True, exist_ok=True)

    arrays = []
    for afile in file_list:
        with rxr.open_rasterio(afile) as da:
            arrays.append(
                da.load()
                .squeeze("band", drop=True)
                .astype("int16")
                .expand_dims(year=[int(afile.name[0:4])])
            )
    ds = xr.concat(arrays, dim="year", join="exact").sortby("year")
    ds = ds.where(ds == 1, 0)  # make sure the maps are strictly binary
    years = ds.year.values.tolist()
    print(f"{habitat_type}: computing metrics over years {years}")

    def write(da, name, dtype, nodata):
        da.astype(dtype).rio.write_nodata(nodata).rio.to_raster(
            save_path / f"{name}_{habitat_type}.tif", compress="ZSTD", driver="COG"
        )

    # Frequency of habitat presence, in % of years (0 = never detected = nodata)
    freq = 100 * ds.sum(dim="year") / len(years)
    write(freq, "frequency", "float32", 0)

    # Tendency between first and last year:
    # 1 = appeared, -1 = disappeared, 0 = present in both, -99 = absent in both
    first, last = ds.isel(year=0), ds.isel(year=-1)
    tend = (last - first).where((first + last) > 0, -99)
    write(tend, "tendency", "int8", -99)

    # Last year with habitat detected (0 = never detected = nodata)
    last_year = (ds * ds.year).max(dim="year")
    write(last_year, "lastyear", "int16", 0)


def run(
    in_folders: list[Path],
    output_dir: Path,
    habitat_types: list[str] = HABITAT_TYPES,
    extent: str = "intersection",
) -> None:
    """Runs the whole time series flow for each habitat type."""
    for habitat_type in habitat_types:
        files, aligned = find_hab_files(in_folders, habitat_type)
        if not aligned:
            files = align_files(
                files, output_dir / "aligned" / habitat_type, extent=extent
            )
        ts_metrics(files, output_dir, habitat_type)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Computes time series statistics from yearly habitat extent results."
    )
    parser.add_argument(
        "--input-folders",
        type=Path,
        nargs="+",
        required=True,
        help="Result folders to use, typically one per year.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where the time series results are written.",
    )
    parser.add_argument(
        "--habitat-types",
        nargs="+",
        default=HABITAT_TYPES,
        choices=HABITAT_TYPES,
        help="Habitat types to process, by default sav and coral.",
    )
    parser.add_argument(
        "--extent",
        default="intersection",
        choices=["intersection", "union"],
        help="Common extent used when the inputs need aligning, by default intersection.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input_folders, args.output_dir, args.habitat_types, args.extent)
