import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
import pickle
import tempfile
import shutil
import pystac
import openeo
from openeo import Connection

from src.main import Algorithm

def get_openeo_connection() -> Connection:
    """
    Connects to openEO and configures it for batch job tracking.
    Expects CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in env.
    """
    cdse_client_id = os.getenv("OPENEO_AUTH_CLIENT_ID")
    cdse_client_secret = os.getenv("OPENEO_AUTH_CLIENT_SECRET")

    if not cdse_client_id or not cdse_client_secret:
        raise ValueError("OPENEO_AUTH_CLIENT_ID and OPENEO_AUTH_CLIENT_SECRET must be set as environment variables.")

    conn = openeo.connect("https://openeofed.dataspace.copernicus.eu/").authenticate_oidc_client_credentials(
        client_id=cdse_client_id,
        client_secret=cdse_client_secret,
        provider_id="CDSE"
    )

    def create_job_logged(*args, **kwargs):
        job = conn.create_job_orig(*args, **kwargs)
        print(f"Tracking execution for job: {job.job_id}")
        return job

    conn.execute = None  # We prevent sync execution until we have it logged
    conn.create_job_orig = conn.create_job
    conn.create_job = create_job_logged  # log all batch jobs to enable tracking

    return conn

def run_main_algorithm():
    """Loads .env and runs the Algorithm.run method """
    load_dotenv()

    conn = get_openeo_connection()
    catalog = pystac.Catalog(id="test-catalogu", description=f"Test Catalog")

    # Define example parameters
    parameters = {
        "rangestart": "2026-01-09",
        "rangeend": "2026-04-23",
        "spatial_extent": {
            "west": 118.545388694,
            "south": 4.537462666,
            "east": 118.568189394,
            "north": 4.553378275,
            "crs": "EPSG:4326"
        },
        "maxcloudcover": 40,
        "area_name": "Malaysia"
    }

    try:
        print("Attempting to run Algorithm.run...")
        Algorithm.run(conn, catalog, parameters)
        print("Algorithm.run completed successfully")
        print(json.dumps(catalog.to_dict()))
    except Exception as e:
        print(f"Algorithm.run encountered an error: {e}")
   
    print("Execution finished.")

if __name__ == "__main__":
   run_main_algorithm()
