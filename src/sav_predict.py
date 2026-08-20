import argparse
import os
import pickle
from pathlib import Path

import geopandas as gpd
from pystac import Catalog
from shapely.geometry.polygon import Polygon

from .utils import Parameters, ts_predict


def main(
        work_dir: Path, model_dir: Path, catalog: Catalog, parameters: Parameters
) -> None:
    """
    Loads the model and generates predictions.
    It should contain 2 subfolders:
        - "openeo_tifs"
        - "models"
    Saves results in a new subfolder called "predictions"

    Parameters
    ----------
    chunk_size : int, optional
        Chunk size for image processing, by default 512. Can be reduced if ram is an issue.
    """

    # Load model
    model_name = f"rf_model_{parameters.MODEL_NAME}"
    print(f"loading model {model_name}")
    rf_model = load_model(model_dir, model_name)

    geometry_clip = None
    if not parameters.SPATIAL_EXTENT.get("west"):
        poly_coords = parameters.SPATIAL_EXTENT["coordinates"][0]
        polygon = Polygon(poly_coords)
        geometry_clip = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

    ts_predict(work_dir, rf_model=rf_model, chunk_size=parameters.CHUNK_SIZE, area=parameters.AREA, catalog=catalog, geometry_clip=geometry_clip)


def load_model(model_dir: Path, model_name: str):
    with open(
            model_dir / f"{model_name}.pkl",
            "rb",
    ) as f:
        rf_model = pickle.load(f)

    return rf_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        work_dir = Path(os.environ["DHI_TEST_PATH"])
    except KeyError:
        raise RuntimeError("DHI_TEST_PATH environment variable must be set.")
    main(work_dir, work_dir, chunk_size=args.chunk_size)
