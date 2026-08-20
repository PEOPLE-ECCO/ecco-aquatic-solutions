import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import dask.array as darr
import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import rasterio
import rioxarray as rxr
import xarray as xr
from openeo.udf import XarrayDataCube
from osgeo import gdal
from pystac import Catalog
from rasterio import features
from rasterio.transform import from_origin
from skimage import feature
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
gdal.DontUseExceptions()

from geopandas import GeoDataFrame
from sklearn.ensemble import RandomForestClassifier


@dataclass
class Parameters:
    BANDS: list[str]
    TEMP_EXTENT: tuple[str, str]
    SPATIAL_EXTENT: dict[str, float]
    MAX_CLOUD_COVER: int
    STORE_INTERMEDIATES: bool
    MODEL_NAME: str
    AREA: str
    CHUNK_SIZE: int = 512


SIGMA = 2
DEPTH = int(np.ceil(4 * SIGMA))

# Update these
# The probability tif file has only 3 bands, corresponding to RGB channels. They should all be stretched from 5000 to 9000 to highlight
# probability features best. The probability values are scaled from 0 to 10.000 and represent the following probabilities:
# The red channel stands for corals, the green for SAV presence and the blue for open water, or sand beneath shallow water.

# Prediction style should show classification results on a single band, with values stretching from -1 to 2.
# The classification result show the classification with the highest probability for each pixel, but doesn't account for
# potential low probability predictions. To have a better overview of areas predicted with high certainty, it is better to
# look at the probability RGB image.
# -1 : codes for land and cloud cover
# 0: Predicted Coral cover
# 1: Predicted SAV
# 2: Predicted open water, turbid water, or sand below shallow water

# The binary style should also accomodate for a single band, with 0s and 1s where the mapped feature is present.
# Two results are created using this style, one mapping SAV presence, and the other coral presence. Both these results
# are likely temprorary and will be replaced in the future by the connectivity layers.

PROBABILITY_STYLE = '{"color":["color",["interpolate",["linear"],["band",1],5000,0,9000,255],["interpolate",["linear"],["band",2],5000,0,9000,255],["interpolate",["linear"],["band",3],5000,0,9000,255],["case", ["<=", ["band", 3], 0], 0, 1]]}'
PREDICTION_STYLE = '{"color":["interpolate",["linear"],["band",1],-1,[0,0,0,0],0,[0,0,0,0],1,[200,0,0,1],2,[0,200,0,1],3,[0,0,255,0]]}'
GRAY_STYLE = '{"color":["color",["interpolate",["linear"],["band",1],2500,10,9000,240],["interpolate",["linear"],["band",1],2500,10,9000,240],["interpolate",["linear"],["band",1],2500,0,9000,240],["case", ["<=", ["band", 1], 0], 0, 1]]}'
S2_STYLE = '{"color":["color",["interpolate",["linear"],["band",3],100,0,3000,255],["interpolate",["linear"],["band",2],100,0,3000,255],["interpolate",["linear"],["band",1],100,0,3000,255],["case", ["<=", ["band", 3], 0], 0, 1]]}'


def open_ds(
    img_path: Path | str, chunk_size: int = 512, geometry_clip: gpd.GeoDataFrame = None
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Opens a tif file as an xarray dataset with spatial chunking.
    Normalizes band values and returns a simplified dataset with a cloud mask.

    Parameters
    ----------
    img_path : Path | str
        Path to the file to open.
    chunk_size : int, optional
        Spatial chunk size, by default 512
    Returns
    -------
    tuple of datasets
        Dataset with bands, and dataset containing the cloud mask.
    """
    da = xr.open_dataset(
        img_path, chunks={"band": -1, "x": chunk_size, "y": chunk_size}
    )
    da = da.assign_coords(band=("band", ["blue", "green", "red", "nir"]))

    # Cloud mask
    mask = da.sel(band="red").isnull()
    cm_da = mask.expand_dims(band=["cm"])

    # Geometry clipping to save memory
    if geometry_clip is not None:
        geometry_clip = geometry_clip.to_crs(da.rio.crs).geometry.buffer(150)
        da = da.rio.clip(geometry_clip)
        cm_da = cm_da.rio.clip(geometry_clip)

    # Clip and set as ints
    da = (da.clip(0, 10000).fillna(0)).astype("int16")

    return da.persist(), cm_da["band_data"].persist()


def compute_stats(arr: xr.Dataset, window: int) -> xr.Dataset:
    rolled = arr.rolling(y=window, x=window, center=True)
    means = rolled.mean()
    eps = 1e-3

    return xr.concat(
        [
            means.fillna(0).astype("int16"),
            (arr / (means + eps)).fillna(0).clip(0, 10).astype("float16"),
        ],
        dim="stat",
    ).assign_coords(stat=["mean", "ratio"])


def add_indices(data: xr.DataArray) -> xr.Dataset:
    blue, green, red, nir = (
        data.sel(band="blue"),
        data.sel(band="green"),
        data.sel(band="red"),
        data.sel(band="nir"),
    )
    eps = 1e-3

    # Compute some indices
    rb_idx = (10000 * ((blue - red) / (blue + red + eps))).astype("int16")
    bg_idx = (10000 * ((blue - green) / (blue + green + eps))).astype("int16")
    nirb_idx = (10000 * ((blue - nir) / (nir + blue + eps))).astype("int16")
    rdvi = (100 * ((nir - red) / np.sqrt(nir + red + eps))).astype("int16")

    rb_ratio = (100 * (red / (blue + eps))).astype("int16")
    bg_ratio = (100 * (green / (blue + eps))).astype("int16")
    nirb_ratio = (100 * (nir / (blue + eps))).astype("int16")
    nirr_ratio = (100 * (nir / (red + eps))).astype("int16")

    orig_band_names = ["red", "green", "blue", "nir"]
    mask = (red == 0) | (green == 0) | (blue == 0) | (nir == 0)

    orig_data = [
        data.sel(band=bname).expand_dims(band=[name])
        for bname, name in zip(["blue", "green", "red", "nir"], orig_band_names)
    ]

    index_bands = {
        "rb_idx": rb_idx,
        "bg_idx": bg_idx,
        "nirb_idx": nirb_idx,
        "rdvi": rdvi,
        "rb_ratio": rb_ratio,
        "bg_ratio": bg_ratio,
        "nirb_ratio": nirb_ratio,
        "nirr_ratio": nirr_ratio,
    }

    index_data = [v.expand_dims(band=[name]) for name, v in index_bands.items()]

    all_data = xr.concat(orig_data + index_data, dim="band")
    all_data = all_data.where(~mask, 0)
    return all_data


def extract_features_block(block):
    """
    block shape: (band, y, x)
    returns: (feature, y, x)
    """
    feats = feature.multiscale_basic_features(
        block,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=SIGMA,
        sigma_max=SIGMA,
        channel_axis=0,
    ).astype(np.float32)

    return np.moveaxis(feats, -1, 0)


def create_label_array(
    img_labels: GeoDataFrame, ref: xr.DataArray
) -> tuple[xr.DataArray]:
    """
    Rasterizes shapefiles with label values and split between train and test sets.

    Parameters
    ----------
    img_labels : GeoDataFrame
        Geodataframe with label values.
    ref : xr.DataArray
        Reference dataarray with transform and crs.

    Returns
    -------
    tuple[xr.DataArray]
        datarrays containing label values and train/ test values
    """
    labels_arr = features.rasterize(
        (
            (geom, value)
            for geom, value in zip(img_labels.geometry, img_labels["class_id"])
        ),
        out_shape=ref.shape,
        transform=ref.rio.transform(),
        fill=0,
        dtype="int16",
    )
    labels_da = xr.DataArray(
        labels_arr, coords=ref.coords, dims=ref.dims, name="labels"
    ).assign_coords(band="label")

    val_arr = features.rasterize(
        (
            (geom, value)
            for geom, value in zip(
                img_labels.geometry, img_labels["dataset"].map({"train": 0, "test": 1})
            )
        ),
        out_shape=ref.shape,
        transform=ref.rio.transform(),
        fill=0,
        dtype="int16",
    )
    val_da = xr.DataArray(
        val_arr, coords=ref.coords, dims=ref.dims, name="validation"
    ).assign_coords(band="validation")

    return labels_da, val_da


def compute_attributes(
    da: xr.Dataset,
    add_da: list = [],
    win_size: int = 5,
) -> xr.Dataset:
    """
    Computes all additional attributes needed for classification (indices, and convolutions on rolled arrays).
    Parameters
    ----------
    da : xr.Dataset
        Input dataset
    add_da : list, optional
        List of additional dataarrays to concatenate to the result, by default []
    win_size : int, optional
        Window size for convolutions, by default 5

    Returns
    -------
    xr.Dataset
        Dataset containing all bands for training and inference.
    """
    # Indices
    s2_data = add_indices(da["band_data"].copy()).chunk({"band": -1})

    # Convolutions
    n_features = (
        4 * s2_data.band.size
    )  # As long as we use a single sigma, and return all

    feat_arr = darr.map_overlap(
        extract_features_block,
        s2_data.data,
        depth={1: DEPTH, 2: DEPTH},
        boundary="reflect",
        trim=True,
        dtype=np.float32,
        drop_axis=0,
        new_axis=0,
        chunks=(
            n_features,
            tuple(np.array(s2_data.chunksizes["y"]) + 2 * DEPTH),
            tuple(np.array(s2_data.chunksizes["x"]) + 2 * DEPTH),
        ),
        meta=np.empty((0, 0, 0), dtype=np.float32),
    )

    conv_ds = xr.DataArray(
        feat_arr,
        dims=("feature", "y", "x"),
        coords={
            "feature": np.arange(n_features),
            "y": s2_data["y"],
            "x": s2_data["x"],
        },
    )
    conv_ds = conv_ds.rename({"feature": "band"})
    band_names = [f"conv_{i}" for i in range(n_features)]
    conv_ds = conv_ds.assign_coords(band=("band", band_names))

    # Compute local statistics
    stats_ds = compute_stats(s2_data, win_size)
    new_band_names = [
        f"{b}_{s}" for b in stats_ds.band.values for s in stats_ds.stat.values
    ]
    da_merged = stats_ds.stack(band_stat=("band", "stat")).transpose(
        "band_stat", "y", "x"
    )
    stats_ds = da_merged.drop_vars(["band_stat", "band", "stat"]).assign_coords(
        band_stat=new_band_names
    )
    stats_ds = stats_ds.rename({"band_stat": "band"})

    # Add indices to base bands
    s2_data = s2_data.assign_coords(band=("band", list(s2_data.band.values)))

    # Concatenate all
    all_ds = xr.concat(
        [s2_data, stats_ds, conv_ds] + add_da, dim="band", coords="minimal"
    )

    all_ds = all_ds.stack(pixel=("y", "x")).transpose("pixel", "band")

    return all_ds.astype("float32")


def create_transform(arr: xr.Dataset) -> rasterio.Affine:
    """
    Creates the output transform from the spatial information in the dataset.
    Parameters
    ----------
    arr : xr.Dataset
        Dataset with spatial information in x and y coordinates.
    Returns
    -------
    rasterio.Affine
        Transform for writing results with rasterio.
    """
    step_x = (arr.x[1] - arr.x[0]).values
    step_y = (arr.y[0] - arr.y[1]).values

    transform = from_origin(
        arr.x.min().values - step_x / 2,
        arr.y.max().values + step_y / 2,
        step_x,
        step_y,
    )
    return transform


def dask_predict(
    da_path: Path,
    rf_model: RandomForestClassifier,
    base_path: Path,
    base_name: str,
    crs: str,
    area: str,
    catalog: Catalog = None,
    chunk_size: int = 512,
    **kwargs,
) -> None:
    """
    Loads a tif file downloaded from openeo, transforms it for prediction, and runs inference
    using the loaded model. Writes a tif file with output probabilities.

    Parameters
    ----------
    da_path : Path
        Path to the input tif file which on which inference is ran.
    rf_model : RandomForestClassifier
        The trained random forest classifier for inference.
    base_path : Path
        Path to the output directory.
    base_name : str
        Base name for the outputs.
    crs : str
        crs for the output.
    area: str
        Name of the area predicted, changing the handling of results. Can be Malaysia or PEI, and defaults to None.
    catalog : Catalog, optional
        Pystac catalog, defaults to None.
    chunk_size: int
        Chunck size to handle spatial data. Defaults to 512.
    kwargs: dict
        chunk_size: int for loading and predicting values on spatial chunks.
    """

    if area not in ["Malaysia", "PEI"]:
        raise ValueError(
            f"area needs to be either Malaysia or PEI, got {area} instead."
        )

    rf_columns = list(rf_model.feature_names_in_)

    # Clip values and normalize them
    da, cm_da = open_ds(da_path, chunk_size, **kwargs)

    # Compute all attributes
    da = compute_attributes(da, add_da=[cm_da])

    # Predict and extract probabilities
    def predict_block(block):
        probs = rf_model.predict_proba(block)
        return probs

    cm = da.sel(band="cm")
    da2 = da.sel(band=rf_columns).chunk({"pixel": 25000, "band": -1})
    da2 = da2.assign_coords(cm=cm)

    probs = xr.apply_ufunc(
        predict_block,
        da2,
        input_core_dims=[["band"]],
        output_core_dims=[["class_pred"]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=["float16"],
        dask_gufunc_kwargs={"output_sizes": {"class_pred": rf_model.n_classes_}},
    )

    # Probabilities
    probs = probs.assign_coords(class_pred=rf_model.classes_)

    # Compute the dask array and set it back to the original image shape
    probs_2d = probs.unstack("pixel").compute()

    # Extract relevant values and update values based on cloud mask
    if area == "Malaysia":
        pred_rgb = 10000 * xr.concat(
            [
                probs_2d.sel(class_pred=2, drop=True),  # Coral / rock
                probs_2d.sel(class_pred=3, drop=True),  # SAV
                probs_2d.sel(class_pred=1)
                + probs_2d.sel(
                    class_pred=4
                ),  # Open water and sand beaneath shallow water
            ],
            dim="class_pred",
        )
    elif area == "PEI":
        zero_band = xr.zeros_like(probs_2d.sel(class_pred=2, drop=True))
        pred_rgb = 10000 * xr.concat(
            [
                zero_band,  # Set rock to 0 prob
                probs_2d.sel(class_pred=3, drop=True),  # SAV
                probs_2d.sel(class_pred=1, drop=True)
                + probs_2d.sel(class_pred=4, drop=True)
                + probs_2d.sel(class_pred=2, drop=True),
            ],
            dim="class_pred",
        )

    pred_rgb.values[:, pred_rgb.cm == 1] = -1

    # Save results (intermediary results)
    transform = create_transform(pred_rgb)

    # Check crs + style
    probability_metadata = {
        "type": "Single scene probability",
        "epsg": crs.to_epsg(),
        "style": PROBABILITY_STYLE,
    }

    probs_tif = base_path / f"{base_name}_probrgb_openeo.tif"
    write_tif(
        pred_rgb, probs_tif, crs, transform, catalog=catalog, meta=probability_metadata
    )

    return


def write_tif(
    arr: xr.DataArray,
    write_path: Path,
    crs: str,
    transform: rasterio.Affine,
    dtype: str = "int16",
    out_count: int = None,
    catalog: Catalog = None,
    meta: dict = None,
    nodata: float = None,
) -> None:
    """
    Writes a tif file, using an input array and a save path.

    Parameters
    ----------
    arr : xr.DataArray
        Array which values should be written
    write_path : Path
        Path for writing
    crs : str
        Output crs.
    transform : rasterio.Affine
        Output transform
    dtype : str, optional
        Output dtype, by default "int16"
    out_count : int, optional
        Number of bands in the output, by default None.
    catalog : Catalog, optional
        Pystac ctalog in which the output is saved, by default None.
    meta : dict, optional
        Metadata dict for the catalog item, by default None.
    nodata: float, optional
        Nodata value for the file. Defaults to None.
    """
    if out_count is None:
        if "class_pred" in arr.dims:
            out_count = arr.class_pred.shape[0]
        else:
            out_count = 1

    profile = {
        "driver": "COG",
        "height": arr.y.shape[0],
        "width": arr.x.shape[0],
        "count": out_count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "LZW",
        "blocksize": 512,
        "overviews": "IGNORE_EXISTING",
        "resampling": "NEAREST",
        "num_threads": "ALL_CPUS",
    }

    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(write_path, "w", **profile) as dst:
        if out_count == 1:
            dst.write(arr.values, 1)
        else:
            dst.write(arr.values)

        # Catalog metadata
        if catalog is not None:
            bounds = dst.bounds
            footprint = {
                "type": "Polygon",
                "coordinates": [
                    [
                        (bounds.left, bounds.bottom),
                        (bounds.left, bounds.top),
                        (bounds.right, bounds.top),
                        (bounds.right, bounds.bottom),
                        (bounds.left, bounds.bottom),
                    ]
                ],
            }
            item = pystac.Item(
                id=write_path.name,
                geometry=footprint,
                bbox=bounds,
                datetime=resolve_phenomenon_time(write_path.name),
                properties=meta,
            )

            item.add_asset(
                key="image",
                asset=pystac.Asset(
                    href=write_path.as_posix(), media_type=pystac.MediaType.GEOTIFF
                ),
            )
            catalog.add_item(item)

    return


def resolve_phenomenon_time(item_name: str):
    # resolve the datetime in the semantics of STAC (-> time for data coverage ~= sampling time)
    dt = datetime.now(tz=timezone.utc)  # default fallback
    item_name_wo_ext = item_name.split(".")[0]
    if "_" in item_name_wo_ext:
        chunks = item_name_wo_ext.split("_")
        for c in chunks:
            try:
                dt = datetime.fromisoformat(c[:10])
                break
            except ValueError:
                pass

    return dt


def create_filtered_tif(
    arr: xr.DataArray,
    out_path: Path,
    crs: str,
    transform: rasterio.Affine,
    sieve_threshold: int = 16,
    connectedness: int = 4,
    catalog: Catalog = None,
    meta: dict = None,
) -> None:
    """
    Applies a sieve filter to classification results, and writes the filtered
    raster file.
    Parameters
    ----------
    arr : xr.DataArray
        Input dataarray.
    out_path : Path
        Path where the result is written.
    crs : str
        Output crs.
    transform : rasterio.Affine
        Output transform.
    sieve_threshold : int, optional
        Sieve threshold for the sieve method, by default 16
    connectedness : int, optional
        Other parameter for sieving, by default 4.
    catalog : Catalog, optional
        Pystac ctalog in which the output is saved, by default None.
    meta : dict, optional
        Metadata dict for the catalog item, by default None.
    """
    # Values
    values = arr.values
    # Create a src_band with the values to apply the Sieve filter
    src_ds = gdal.GetDriverByName("MEM").Create(
        "", values.shape[1], values.shape[0], 1, gdal.GDT_Int8
    )
    src_band = src_ds.GetRasterBand(1)
    src_band.WriteArray(values)

    # Create destination in memory to keep filtered array
    tmp_ds = gdal.GetDriverByName("MEM").Create(
        "", values.shape[1], values.shape[0], 1, gdal.GDT_Int8
    )
    tmp_ds.SetGeoTransform(transform.to_gdal())
    tmp_ds.SetProjection(crs.to_wkt())
    tmp_band = tmp_ds.GetRasterBand(1)

    # Filter
    gdal.SieveFilter(
        srcBand=src_band,
        maskBand=None,
        dstBand=tmp_band,
        threshold=sieve_threshold,
        connectedness=connectedness,
    )

    tmp_band.FlushCache()

    # And save it as a COG
    dst_ds = gdal.GetDriverByName("COG").CreateCopy(
        out_path,
        tmp_ds,
        options=[
            "COMPRESS=LZW",
            "BLOCKSIZE=512",
            "OVERVIEWS=IGNORE_EXISTING",
            "RESAMPLING=NEAREST",
            "NUM_THREADS=ALL_CPUS",
        ],
    )
    # Catalog metadata
    if catalog is not None:
        with rasterio.open(out_path, "r") as src:
            bounds = src.bounds
            footprint = {
                "type": "Polygon",
                "coordinates": [
                    [
                        (bounds.left, bounds.bottom),
                        (bounds.left, bounds.top),
                        (bounds.right, bounds.top),
                        (bounds.right, bounds.bottom),
                        (bounds.left, bounds.bottom),
                    ]
                ],
            }
            item = pystac.Item(
                id=out_path.name,
                geometry=footprint,
                bbox=bounds,
                datetime=resolve_phenomenon_time(out_path.name),
                properties=meta,
            )

            item.add_asset(
                key="image",
                asset=pystac.Asset(
                    href=out_path.as_posix(), media_type=pystac.MediaType.GEOTIFF
                ),
            )
            catalog.add_item(item)

    # Close dataset
    dst_ds = None
    src_ds = None

    return


def da2df(da: xr.DataArray) -> pd.DataFrame:
    """
    Transforms the datarray into a dataframe for training/ prediction
    """
    full_df = pd.DataFrame(
        da, columns=[band for band in da.band.values], index=da.pixel
    ).reset_index()
    full_df.insert(0, "y", full_df["index"].map(lambda x: x[0]))
    full_df.insert(0, "x", full_df["index"].map(lambda x: x[1]))
    full_df.drop(columns="index", inplace=True)
    return full_df


def da2df_split(da, rf_columns):
    full_df = da2df(da)
    return full_df.loc[:, rf_columns], full_df[["x", "y", "cm"]]


def apply_cm(dc: XarrayDataCube, ks: int = 5) -> XarrayDataCube:
    """
    Applies standard masking to an xarray datacube

    Parameters
    ----------
    dc : XarrayDataCube
        Openeo datacube object
    ks : int, optional
        Kernel size for cloud buffering, by default 5
    Returns
    -------
    XarrayDataCube
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


def aggregate_probs(
    files: list[Path],
    save_path: Path,
    chunk_size: int = 512,
    catalog: Catalog = None,
    mean_time: str = "",
) -> xr.Dataset:
    """
    Aggregates class probabilities on a list of predictions into average probability for each class

    Parameters
    ----------
    files : list[Path]
        list of probability predictions
    save_path : Path
        path to the folder where to write the output
    chunk_size : int, optional
        Chunk size to load the data by default 512
    catalog: Catalog, optional
        Pystac catalog to add the aggregated rgb to.
    Returns
    -------
    xr.DataSet
        Average probability for each class.
    """

    data = []
    for afile in files:
        data.append(
            xr.open_dataset(
                afile, chunks={"band": -1, "x": chunk_size, "y": chunk_size}
            )
        )

    data = xr.concat(data, dim="time")
    masked = data.where(data != -1)
    mean_ = (
        masked.mean("time", skipna=True)
        .fillna(-1)
        .astype("int16")
        .compute()["band_data"]
    )

    out_transform = create_transform(mean_)

    probability_metadata = {
        "type": "Aggregated probability",
        "epsg": mean_.rio.crs.to_epsg(),
        "style": PROBABILITY_STYLE,
    }

    write_tif(
        mean_,
        write_path=save_path
        / f"{mean_time}{'_' if mean_time else ''}aggregated_probrgb.tif",
        crs=mean_.rio.crs,
        transform=out_transform,
        out_count=mean_.band.shape[0],
        catalog=catalog,
        meta=probability_metadata,
    )

    # Save the band probabilities
    coral_metadata = {
        "type": "Coral probability",
        "epsg": mean_.rio.crs.to_epsg(),
        "style": GRAY_STYLE,
    }

    write_tif(
        mean_.sel(band=1),
        write_path=save_path
        / f"{mean_time}{'_' if mean_time else ''}aggregated_coral_probability.tif",
        crs=mean_.rio.crs,
        transform=out_transform,
        out_count=1,
        catalog=catalog,
        meta=coral_metadata,
    )

    sav_metadata = {
        "type": "SAV probability",
        "epsg": mean_.rio.crs.to_epsg(),
        "style": GRAY_STYLE,
    }

    write_tif(
        mean_.sel(band=2),
        write_path=save_path
        / f"{mean_time}{'_' if mean_time else ''}aggregated_SAV_probability.tif",
        crs=mean_.rio.crs,
        transform=out_transform,
        out_count=1,
        catalog=catalog,
        meta=sav_metadata,
    )

    return mean_, out_transform


def create_output_preds(
    ds_prob: xr.Dataset | Path | str,
    save_path: Path,
    transform: rasterio.Affine = None,
    sav_prob_thresh: int = 6500,
    coral_prob_thresh: int = 5000,
    base_name: str = "aggregated",
    catalog: Catalog = None,
    mean_time: str = "",
) -> None:
    """
    Loads a probability file, and automatically runs the sieve filtering, and saves three raster files
    (Sav binary mask, coral binary mask, and classification result)
    Parameters
    ----------
    ds_prob : xr.Dataset | Path | str
        Input DataArray, or path to a tif file.
    save_path : Path
        Path to folder where outputs are saved
    transform : rasterio.Affine, optional
        Output transform, by default None. If None, the transform is created from the datarray.
    sav_prob_thresh : int, optional
        Probability threshold for SAV mapping, by default 6500
    coral_prob_thresh : int, optional
        Probability mapping for coral mapping, by default 5000
    base_name : str, optional
        Base name for output files, by default "aggregated"
    catalog : Catalog, optional
        Pystac ctalog in which the output is saved, by default None.
    """

    if isinstance(ds_prob, Path) | isinstance(ds_prob, str):
        ds_prob = xr.open_dataset(ds_prob).load().astype("int16")["band_data"]

    # Predicted class
    predicted = ds_prob.argmax(dim="band").astype("int8")
    predicted.values[ds_prob.max(dim="band") == -1] = -1
    predicted.values = predicted.values + 1

    coral_probs = ds_prob.sel(band=1)
    sav_probs = ds_prob.sel(band=2)

    crs = ds_prob.rio.crs
    if transform is None:
        transform = create_transform(ds_prob)

    # Metadata
    prediction_metadata = {
        "type": "Prediction",
        "epsg": crs.to_epsg(),
        "style": PREDICTION_STYLE,
    }

    # Create filtered thresholded maps / filtered predictions
    # SAV
    create_filtered_tif(
        sav_probs >= sav_prob_thresh,
        save_path / f"{mean_time}{'_' if mean_time else ''}{base_name}_sav.tif",
        crs,
        transform,
    )
    # Coral
    create_filtered_tif(
        coral_probs >= coral_prob_thresh,
        save_path / f"{mean_time}{'_' if mean_time else ''}{base_name}_coral.tif",
        crs,
        transform,
    )
    # Class prediction
    create_filtered_tif(
        predicted,
        save_path
        / f"{mean_time}{'_' if mean_time else ''}{base_name}_classification.tif",
        crs,
        transform,
        catalog=catalog,
        meta=prediction_metadata,
    )

    return


def ts_predict(
    work_dir: Path,
    rf_model: RandomForestClassifier,
    area: str,
    chunk_size: int = 512,
    catalog: Catalog = None,
    **kwargs,
) -> None:
    """
    Wrapper which runs the classification on all tif files in a folder,
    automatically aggregates probability results, and creates filtered
    result files.
    It should contain 2 subfolders:
        - "openeo_tifs"
        - "models"
    Saves results in a new subfolder called "predictions"

    Parameters
    ----------
    work_dir : Path
        Path to the working folder.
    rf_model : RandomForestClassifier
        Saved model.
    area : str
        Name of the model used.
    chunk_size : int, optional
        Chunk size for prediction and aggregation, by default 512
    """

    rf_model.n_jobs = 8

    print("--- Random Forest Model Info ---")
    print(f"Internal Threads (n_jobs): {rf_model.n_jobs}")
    print(f"Number of Trees (n_estimators): {rf_model.n_estimators}")
    print(
        f"Number of Classes (n_classes_): {getattr(rf_model, 'n_classes_', 'Unknown')}"
    )
    print(
        f"Expected Input Bands (n_features_in_): {getattr(rf_model, 'n_features_in_', 'Unknown')}"
    )
    print("--------------------------------")

    tif_dir = work_dir / "openeo_tifs"
    pred_dir = work_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    # Get file list and crs
    file_list = list(tif_dir.glob("*.tif"))
    print(f"{len(file_list)} images for prediction.")

    if len(file_list) > 0:
        with rasterio.open(file_list[0]) as src:
            crs = src.crs

        # Run prediction on all images
        for afile in tqdm(file_list):
            print("Working on File: " + afile.name)
            base_name = afile.name.split(".")[0]
            dask_predict(
                da_path=afile,
                rf_model=rf_model,
                base_path=pred_dir,
                base_name=base_name,
                crs=crs,
                catalog=catalog,
                chunk_size=chunk_size,
                area=area,
                **kwargs,
            )

        # Aggregation of results and creation of output files
        print(" Aggregating classification results")
        aggr_results_dir = pred_dir / "aggr_results"
        aggr_results_dir.mkdir(exist_ok=True)
        prob_files = list(pred_dir.glob("*probrgb*.tif"))

        # resolve the mean time, since the STAC catalogue needs a datetime
        date_strings = list(
            map(
                lambda f: resolve_phenomenon_time(f.name).strftime("%Y-%m-%d"),
                prob_files,
            )
        )

        mean_time = calculate_mean_date_time(date_strings)

        mean_prob, transform = aggregate_probs(
            prob_files,
            save_path=aggr_results_dir,
            chunk_size=chunk_size,
            catalog=catalog,
            mean_time=mean_time,
        )
        print(" Aggregated probabilities saved")

        # Get the list of probability files
        print(" Filtering classification results")
        create_output_preds(
            mean_prob,
            transform=transform,
            save_path=aggr_results_dir,
            catalog=catalog,
            mean_time=mean_time,
        )

        # Create an s2 composite
        print(" Creating an s2 composite")
        create_mean_composite(
            file_list,
            output_path=work_dir,
            mean_time=mean_time,
            catalog=catalog,
            nodata=0,
        )

        print(" Finished")
        if catalog is not None:
            print("Created catalog:")
            catalog.describe()

    else:
        print("No tif file found.")

    return


def calculate_mean_date_time(date_strings):
    # Parse strings into datetime objects
    dt_objects = [datetime.strptime(d, "%Y-%m-%d") for d in date_strings]

    # Convert datetimes to Unix timestamps
    timestamps = [dt.timestamp() for dt in dt_objects]

    # Calculate the average timestamp
    mean_timestamp = sum(timestamps) / len(timestamps)

    # Convert back to a datetime object
    mean_date = datetime.fromtimestamp(mean_timestamp)

    return mean_date.strftime("%Y-%m-%d")


def create_mean_composite(
    tif_paths: list[Path],
    output_path: Path,
    mean_time: str,
    chunksize: int = 512,
    nodata: int = None,
    catalog: Catalog = None,
):
    """
    Create a mean composite from multiple GeoTIFFs using Dask.

    Parameters
    ----------
    tif_paths : list[str] or list[Path]
        List of paths to input GeoTIFFs.
    output_path : str or Path
        Path to the output GeoTIFF.
    chunks : dict
        Spatial chunking for Dask.
    nodata : float or int, optional
        Nodata value (if not provided, inferred from files).
    """

    arrays = []
    for p in tif_paths:
        da = rxr.open_rasterio(p, chunks={"x": chunksize, "y": chunksize})
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)
        arrays.append(da)

    stack = xr.concat(arrays, dim="time")

    in_nodata = stack.rio.nodata
    stack = stack.where(stack != in_nodata)

    composite = stack.mean(dim="time", skipna=True)
    composite = composite.fillna(nodata).clip(0, 9000).astype("int16")
    composite = composite.rio.write_nodata(nodata)
    composite = composite.compute()

    s2_metadata = {
        "type": "S2 composite",
        "epsg": composite.rio.crs.to_epsg(),
        "style": S2_STYLE,
    }
    write_tif(
        composite,
        write_path=output_path
        / f"{mean_time}{'_' if mean_time else ''}s2_mean_composite.tif",
        crs=composite.rio.crs,
        transform=composite.rio.transform(),
        out_count=composite.band.shape[0],
        catalog=catalog,
        meta=s2_metadata,
        nodata=nodata,
    )

    return
