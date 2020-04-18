# Multispectral Physiological Parameter Estimation
This project deals with Physiological parameter estimation such as heart beat through non contact means by use of Multispectral camera. Camera used in the project has an wavelength range of 600-975nm. This project was carried out in collaboration with [Prof.Tauhidhur Rahman](https://www.cics.umass.edu/people/rahman-tauhidur). 

## **Data**
- **Inputs_ippg**
This folder contains all the multispectral data and the ground truth data files.
- **Outputs_ippg**
This folder contains the ippg prediction files for the input data given. 

## **How to Run?**

- Analyse the input data captured by multispectral camera,
```
python InputAnalysis.py --folder "path-to-folder" --file_type 0
[Usage:]
--folder      :  Set folder path
--file_type   :  0- Tiff videos,1- Tiff image sequence 
```
This file has functions for plotting input data, fft data and ICA data which should be enabled from MultiSpectral.py. 

- Video to Csv file format,
```
python CreateCsvData.py  --folder "path-to-folder" --file_type 0
[Usage:]
--folder      :  Set folder path
--file_type   :  0- Tiff videos,1- Tiff image sequence 
```
It converts the raw video captured by the multispectral camera to csv format data.

- Downsample the input data captured,
```
python Downsample.py --downsample 6 --downsampled_file "path-to-folder" --original_file  "path-to-folder"
[Usage:]
--downsample       :   No of times the data should be downsampled
--downsampled_file :   Post downsampling file name
--original_file    :   Input file to be downsampled
```
This file implements downsampling of video file. Input to this is the video csv and Output is the same folder.The downscale factor can be set in code to set the downsampling rate.

-  Run the main file for heart rate prediction,
```
python MultiSpectral.py --slide_time 50 --frame_rate 60 --slide_inc 20
[Usage:]
--slide_time       :  Sliding window time in seconds
--frame_rate       :  Frame rate captured by the camera 
--slide_inc        :  Overlapping increment in time in seconds
--NO_ICA           :  Set to 1 if ICA is enabled, 0 if ICA is disabled
--no_of_components :  No of components in the ICA
--ica_type         :  Type of ICA algorithm
--pca_components   :  No of PCA components
--ica_iter         :  No of ICA iterations to be performed
--ica_tolerance    :  ICA tolerance level to be performed
```
This file is the main file which contains all the functions for implementing ICA, PCA, FFT, Filtering and Data loading. The inputs to this file are from Inputs_ippg and outputs are written to Outputs_ippg folder.
