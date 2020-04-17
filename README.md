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
python InputAnalysis.py
```
This file has functions for plotting input data, fft data and ICA data which should be enabled from MultiSpectral.py. 

- Video to csv data,
```
python CreateCsvData.py
```
It converts the raw video captured by the multispectral camera to csv format data.

- Downsample the input data captured,
```
python Downsample.py
```
This file implements downsampling of video file. Input to this is the video csv and Output is the same folder.The downscale factor can be set in code to set the downsampling rate.

-  Run the main file for heart rate prediction,
```
python MultiSpectral.py 
```
This file is the main file which contains all the functions for implementing ICA, PCA, FFT, Filtering and Data loading. The inputs to this file are from Inputs_ippg and outputs are written to Outputs_ippg folder.

    
- Convert text file to CSV
```
python TextToCSV.py
```
This file converts .txt file formats to .csv format. Path for the input and output files needs to be specified in the code.








