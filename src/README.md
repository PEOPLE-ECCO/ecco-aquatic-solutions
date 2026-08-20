# DHI SAV Implementation

- **main.py** main entrypoint. Parses parameters and then subsequently runs data download and prediction.
- **dl_openeo.py** to download a tiff file from openeo with the required bands. As of now, the spatial location and temporal extent are fixed in the script to a test area, to test out the flow, but will be user defined variables in the future. If you want to test larger spatial extents, you can update the spatial coverage of the area.
- **sav_predict.py** runs all the necessary transformations and applies the trained model to the downloaded image, and currently saves 4 different ouput tif files. If during the prediction, there is a ram issue, you can change the chunk-size parameter to a smaller value.

- **utils.py** utilities


