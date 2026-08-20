import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pystac
import rasterio
from openeo.rest.connection import Connection
from openeo.rest.job import JobFailedException
from pystac import Catalog

from .utils import Parameters


def main(
    conn: Connection, in_path: Path, parameters: Parameters, catalog: Catalog | None
) -> None:
    """
    Connects to openeo and downloads a yearly mosaic over a test area.
    """

    print("Start setup for downloading data")
    save_path = in_path / "openeo_tifs"
    save_path.mkdir(exist_ok=True)

    # Openeo processing, currently hard set values for a test area
    dc = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=parameters.SPATIAL_EXTENT,
        temporal_extent=parameters.TEMP_EXTENT,
        bands=parameters.BANDS + ["CLD", "SCL", "B11"],
        max_cloud_cover=parameters.MAX_CLOUD_COVER,
    )

    # load_collection does not actually filter spatially, contrary to its documentation
    # so we need to filter spatially manually if more specific (than bbox) geometry is given
    if not parameters.SPATIAL_EXTENT.get("west"):
        # We assume we have a valid geojson extent, and not just a bbox
        dc = dc.filter_spatial(parameters.SPATIAL_EXTENT)

    try:
        dc = apply_cm(dc)
        dc = dc.filter_bands(parameters.BANDS)
        dc.save_result(format="GTiff")
        job = dc.create_job()

        print("Start execute_batch data-download")
        job.start_and_wait()
        results = job.get_results()
        results.download_files(save_path)

        if catalog:
            file_list = list(save_path.glob("*.tif"))
            for file in file_list:
                add_to_catalog(catalog=catalog, save_path=file)

    except JobFailedException as e:
        print("Openeo job failed. There might be no scene available.")
        print(f"Error message: {e}")
        raise


def add_to_catalog(catalog: Catalog, save_path: Path):
    # we neet to transform to COG
    output_path = save_path.parent.parent / f"{save_path.stem}.cog.tif"
    with rasterio.open(save_path) as src:
        cog_profile = {
            "driver": "COG",
            "dtype": src.dtypes[0],
            "count": src.count,
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
            "COMPRESS": "LZW",
            "NUM_THREADS": "ALL_CPUS",
            "OVERVIEWS": "IGNORE_EXISTING",
            "RESAMPLING": "NEAREST",
            "INTERLEAVE": "PIXEL",
            "BLOCKSIZE": "512",
        }
        with rasterio.open(output_path, "w", **cog_profile) as dst:
            for band_idx in range(1, src.count + 1):
                dst.write(src.read(band_idx), band_idx)

    # Catalog metadata
    if catalog is not None:
        date_str = re.search(r"\d{4}-\d{2}-\d{2}", output_path.name)
        if date_str is None:
            date = datetime.now()
        else:
            g = date_str.group()
            date = datetime.strptime(g, "%Y-%m-%d")
        item = pystac.Item(
            id=output_path.name,
            geometry=None,
            bbox=None,
            datetime=date,
            properties={
                "type": "OpenEO Raw Files",
                "epsg": 32650,
                "style": '{"color":["color",["interpolate",["linear"],["band",3],0,0,3000,255],["interpolate",["linear"],["band",2],0,0,3000,255],["interpolate",["linear"],["band",1],0,0,3000,255],["case", ["<=", ["band", 3], 0], 0, 1]]}',
            },
        )

        item.add_asset(
            key="image",
            asset=pystac.Asset(
                href=output_path.as_posix(), media_type=pystac.MediaType.GEOTIFF
            ),
        )
        catalog.add_item(item)


def apply_cm(dc, ks: int = 5):
    """
    Applies cloud masking to an openeo cube.

    Parameters
    ----------
    ks : int, optional
        Kernel size for cloud buffering, by default 5
    """
    kernel = np.ones((ks, ks), dtype=int)
    cm = (
        (dc.band("CLD") >= 80)
        | (dc.band("SCL") == 8)
        | (dc.band("SCL") == 9)
        | (dc.band("SCL") == 10)
    )
    # Dilation (expand mask)
    dilated = cm.apply_kernel(kernel)
    dilated = dilated > 0
    dc = dc.mask(dilated)

    dc = dc.mask((dc.band("SCL") == 2) | (dc.band("SCL") == 3) | (dc.band("SCL") == 4))

    swir_mask = dc.band("B11") > 250
    swir_mask = swir_mask.apply_kernel(kernel)
    swir_mask = swir_mask > 0
    swir_mask = swir_mask.apply_kernel(kernel)
    swir_mask = swir_mask == ks**2

    dc = dc.mask(swir_mask)

    return dc
