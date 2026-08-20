import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import geopandas as gpd
from shapely.geometry import Polygon, box, shape

from openeo.rest.connection import Connection
from pystac import Catalog

from .dl_openeo import main as download
from .habitat_metrics import habitat_metric
from .sav_predict import main as predict
from .utils import Parameters


class Algorithm:
    @staticmethod
    def model_regions() -> Dict[str, Polygon]:
        regions = {
            "Malaysia": box(118.67, 4.55, 118.85, 4.70),
            "PEI": box(-63.75, 46.10, -62.75, 46.50)
        }
        return regions

    @staticmethod
    def run(conn: Connection, catalog: Catalog, parameters: Dict) -> None:
        """
        Entrypoint for all runnable Algorithms.

        :param conn: openEO-Connection, already pre-authenticated
        :param catalog: STAC-Catalog storing all outputs this algorithm produces
        :param parameters: User-Supplied parameters.
        :return: None
        """
        os.chdir(Path(__file__).resolve().parent)

        try:
            # Set some sane default parameters
            BANDS = parameters.get("bands") or ["B02", "B03", "B04", "B08"]
            if parameters.get("rangestart") and parameters.get("rangeend"):
                TEMP_EXTENT = (parameters.get("rangestart"), parameters.get("rangeend"))
            else:
                TEMP_EXTENT = ("2023-01-01", "2023-02-10")

            SPATIAL_EXTENT = parameters.get("spatial_extent", {})
            if ("type" not in SPATIAL_EXTENT or SPATIAL_EXTENT["type"] != "Polygon") and ("west" not in SPATIAL_EXTENT):
                print("error parsing spatial extent!")
                print("expected geojson or bbox, - got this:")
                print(SPATIAL_EXTENT)
                return

            MAX_CLOUD_COVER = parameters.get("maxcloudcover", 40)
            STORE_INTERMEDIATES = parameters.get("store_intermediates", True)
            MODEL_NAME = parameters.get("model_name", None)
            CHUNK_SIZE = parameters.get("chunk_size", 512)
            AREA = parameters.get("area_name", "Malaysia")

            # resolve model from input spatial_extend
            if MODEL_NAME is None:
                print("Could not find MODEL_NAME, deriving from spatial extent ...")
                # nearest neighbor area
                nearest_name, distance = Algorithm.get_closest_feature_name(Algorithm.parse_spatial_extent(SPATIAL_EXTENT), Algorithm.model_regions())

                print(f"Nearest model region: {nearest_name}")
                print(f"Distance in degrees: {distance}")
                MODEL_NAME = nearest_name
                AREA = nearest_name

            params = Parameters(BANDS, TEMP_EXTENT, SPATIAL_EXTENT, MAX_CLOUD_COVER, STORE_INTERMEDIATES, MODEL_NAME, AREA, CHUNK_SIZE)
            print(f"Running with parameters {params}")

            with tempfile.TemporaryDirectory(delete=False) as workdir:
                print("Triggering download")
                download(conn, Path(workdir), params, catalog if STORE_INTERMEDIATES is not None else None)

                print("Triggering prediction")
                predict(
                    Path(workdir),
                    Path(Path(__file__).resolve().parent) / "models",
                    catalog,
                    params
                )

                print("Triggering habitat metric calculation")
                habitat_metric(Path(workdir) / "predictions", catalog)

        except Exception as e:
            # TODO: improve error logging
            print(e)

    @staticmethod
    def parse_spatial_extent(target_input: dict):
        if "west" in target_input:
            # Handle bounding box dictionary
            target_poly = box(
                target_input["west"],
                target_input["south"],
                target_input["east"],
                target_input["north"]
            )
        elif "type" in target_input and target_input["type"] == "Polygon":
            # Handle standard GeoJSON geometry object string/dictionary
            target_poly = shape(target_input)
        else:
            raise ValueError("spatial_extent format not recognized. Must be a BBox or GeoJSON geometry.")

        return target_poly

    @staticmethod
    def get_closest_feature_name(target_poly: Polygon, candidate_map: Dict[str, Polygon]) -> Tuple[str, float]:
        """
        Finds the name of the closest polygon feature from a map relative to a target polygon
        using a spatial index in EPSG:4326.

        Parameters:
            target_poly (Polygon): The feature you are measuring from.
            candidate_map (Dict[str, Polygon]): A dictionary mapping unique names (keys)
                                                to Polygon geometry objects (values).

        Returns:
            Tuple[str, float]: A tuple containing:
                - The string key (name) of the closest polygon feature
                - The spatial distance (in decimal degrees) to that feature
        """
        # separate keys and geometries while maintaining order
        names = list(candidate_map.keys())
        geometries = list(candidate_map.values())

        # build GeoDataFrames
        gdf_candidates = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
        gdf_target = gpd.GeoDataFrame(geometry=[target_poly], crs="EPSG:4326")

        # query the R-Tree spatial index for the nearest neighbor
        nearest_indices, distances = gdf_candidates.sindex.nearest(
            gdf_target.geometry,
            return_distance=True
        )

        # map index back to its corresponding string key
        closest_array_index = nearest_indices[1][0]
        closest_feature_name = names[closest_array_index]
        decimal_degree_distance = distances[0]

        return closest_feature_name, decimal_degree_distance