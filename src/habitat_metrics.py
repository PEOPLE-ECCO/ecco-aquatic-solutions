import json
import re
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pystac
import rasterio
from pystac import Catalog
from rasterio.features import shapes
from shapely.geometry import shape
from skimage.morphology import remove_small_objects, binary_opening, disk


def habitat_metric(prediction_dir: Path, catalog: Catalog, buffer_distance=10, neighbor_threshold=1000):
    """
    processes a single-band raster image to clean objects and convert them into
    a simplified vector file.

    args:
        raster_input_path: path to the input raster file, eg. sav or coral predictions.
        shapefile_output_path: path for the output file.
    """

    # Get file list and crs
    file_list = []
    for suffix in ["*aggregated_sav.tif", "*aggregated_coral.tif"]:
        # .glob() searches the directory and yields matching Path objects
        file_list.extend((prediction_dir / "aggr_results").glob(suffix))

    for file in file_list:
        try:
            print(f"Calculating metrics for: {file}")
            with rasterio.open(file) as src:
                sav_data = src.read(1)
                transform = src.transform
                crs = src.crs.to_string()

            # Print the basic statistics
            print(f"--- Stats for {file} ---")
            print(f"Min:  {sav_data.min()}")
            print(f"Max:  {sav_data.max()}")
            print(f"Mean: {sav_data.mean():.4f}")
            print("-------------------------")

            # Nothing was detected - skip calculation as it crashes when nothing is detected
            if sav_data.max() == 0:
                print(f"Input image {file} is entirely 0. Creating an empty GeoDataFrame.")
                # Initialize an empty GeoDataFrame with a geometry column and the correct CRS
                gdf = gpd.GeoDataFrame({'geometry': []}, crs=crs)
            else:
                sav_mask = sav_data == 1
                print(f"Pixels before removing small objects: {sav_data.sum()}")

                cleaned_mask = remove_small_objects(sav_mask, min_size=1000)
                print(f"Pixels after removing small objects: {cleaned_mask.sum()}")

                smoothed_mask = binary_opening(cleaned_mask, footprint=disk(2)).astype(np.uint8)
                print(f"Pixels after binary opening: {smoothed_mask.sum()}")

                extracted_features = (
                    shape(shape_obj)
                    for shape_obj, value in shapes(smoothed_mask, transform=transform)
                    if value == 1
                )

                geometries_list = list(extracted_features)
                print(f"Count of extracted features:: {len(geometries_list)}")

                gdf = gpd.GeoDataFrame(geometry=geometries_list, crs=crs)
                gdf['geometry'] = gdf['geometry'].simplify(tolerance=5, preserve_topology=True)

                gdf['geometry'] = gdf['geometry'].buffer(-buffer_distance, join_style=2, mitre_limit=1).buffer(buffer_distance, join_style=2, mitre_limit=1)
                gdf['patch_id'] = range(1, len(gdf) + 1)
                gdf = gdf[~gdf.geometry.is_empty]

                gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
                gdf["area_ha"] = (gdf_proj.geometry.area / 10000).round(2)

                gdf_proj["buffer"] = gdf_proj.geometry.buffer(neighbor_threshold)

                neighbors_list = []
                neighbor_total_area = []

                for i, row in gdf_proj.iterrows():
                    hits = gdf_proj[gdf_proj.geometry.intersects(row['buffer'])].index.tolist()
                    hits = [h for h in hits if h != i]  # exclude self
                    neighbors_list.append(hits)

                    total_neigh_area = gdf.loc[hits, "area_ha"].sum()
                    neighbor_total_area.append(round(total_neigh_area, 2))

                gdf["neighbors"] = neighbors_list
                gdf["neigh_area_ha"] = neighbor_total_area

            write_geojson(
                json.loads(gdf.to_json()),
                f"{file}.geojson",
                prediction_dir,
                gdf.crs.to_epsg() if gdf.crs else 4326,
                catalog
            )
        except Exception as e:
            print(f"Error while calculating habitat_metrics: {e}")


def write_geojson(
        geojson: dict,
        name: str,
        write_path: Path,
        crs: int,
        catalog: Catalog = None
) -> None:
    """
    Writes a geojson file from given dict,

    Parameters
    ----------
    geojson : dict
        geojson source
    name : str
        Name of the output file.
    write_path : Path
        Path for writing
    crs : int
        Output crs EPSG code
    catalog : Catalog, optional
        Pystac ctalog in which the output is saved, by default None.
    meta : dict, optional
        Additional metadata
    """

    with open(
            name,
            "w",
    ) as dst:
        json.dump(geojson, dst)
        # Catalog metadata

        date_str = re.search(r'\d{4}-\d{2}-\d{2}', name)
        if date_str is None:
            date = datetime.now()
        else:
            g = date_str.group()
            date = datetime.strptime(g, "%Y-%m-%d")
        item = pystac.Item(
            id=name,
            geometry=None,
            bbox=None,
            datetime=date,
            properties={
                "type": "geojson-" + ("sav" if "sav" in name else "coral"),
                "epsg": crs
            },
        )

        item.add_asset(
            key="image",
            asset=pystac.Asset(
                href=(write_path / name).as_posix(), media_type=pystac.MediaType.GEOJSON
            ),
        )
        if catalog is not None:
            catalog.add_item(item)

    return
