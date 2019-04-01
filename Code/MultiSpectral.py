import cv2
from skimage import io
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.io import RawArray
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import os
import InputAnalysis as ia
import csv
from scipy.signal import butter,lfilter

#Set the following manually before run

DEBUG_NOISE =1       #This variable is set to plot all data related to noise data in case noise da is given as input.
PROCESS_NOISE = 0    #This variable is to enable the noise data processing
DEBUG_INPUT_DATA = 1 #This variable is set to plot all data related to input data.
DEBUG_FFT = 0        #This variable is set to plot all plots of FFT data.
BRUTEFORCE_FFT = 0   #This variable is set to try all combination of FFT's.
MULTIPLE_VIDEO = 0   #This variable is set to 0 for single video and set to 1 in case multiple files in folder.

#Set folder paths
folder = "Data for multispectral Camera 19-03-19" #set folder path
video_folder =folder+"/Inc"                       #In case only sub folders need to be modified from the base folder

#Set this to Avg all the ICA components and then perform fft on that
mean_fft =1

#Multispectral video data processing settings
frame_discard_start = 100
frame_discard_end = 100
frame_rate =24

#Set ICA details
no_of_components = 25 #no of components in the ICA
start_comp=0          #Start component of ICA
end_comp =25          #End Component of ICA
ica_type =0           #0- Extended infomax, 1 - fast ICA


write_out = None
wavelength_noise_mean = np.zeros(shape=(25))


wavelengths =[975,960,945,930,915,900,890,875,850,835,820,805,790,775,760,745,730,715,700,675,660,645,630,615,600]
wavelength_map = {
    (0, 4): '975',
    (0, 3): '960',
    (0, 2): '945',
    (0, 1): '930',
    (0, 0): '915',
    (1, 4): '900',
    (1, 3): '890',
    (1, 2): '875',
    (1, 1): '850',
    (1, 0): '835',
    (2, 4): '820',
    (2, 3): '805',
    (2, 2): '790',
    (2, 1): '775',
    (2, 0): '760',
    (3, 4): '745',
    (3, 3): '730',
    (3, 2): '715',
    (3, 1): '700',
    (3, 0): '675',
    (4, 4): '660',
    (4, 3): '645',
    (4, 2): '630',
    (4, 1): '615',
    (4, 0): '600'
}


#Normalizes data with zero mean and unit variance
def normalizeData(data):
    data_norm = np.zeros(shape=(len(data)))
    data_norm[:] = (data[:] - np.mean(data[:]))/np.std(data[:])
    return data_norm

#Get list of all files in the given directory.
def getFilesList(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        allFiles.append(fullPath)
        print("Full path ",fullPath)
    return allFiles

#Intermediate function to call the get all files paths from the directory
def getVideoFileList():
    videos_paths = getFilesList(video_folder)
    return  videos_paths

# Write CSV data to the file
def writeCSVFile(data):
    myFile = open('Mean_wavelength_values_new.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)

#This function is used in case there exists a background noise video without subject and only the noise needs to be eliminated
def readNoiseFile(video_path):
    global wavelength_noise_mean
    raw = io.imread(video_path)
    write_out.write("SIZE " + str(raw.shape))
    frame_discard = 50
    print("Frames discarded ", frame_discard)
    raw_io = raw[50:(raw.shape[0] - frame_discard), :]
    frames = raw_io.shape[0]

    print("Frame length", frames)
    wavelength_frame_avg = np.zeros(shape=(25, frames))

    wavelength_normalized = np.zeros(shape=(25))
    wavelength_sums = np.zeros(shape=(25))
    print("Total video image dimention:", raw_io.shape)
    print("Total frames:", raw_io.shape[0], " ", raw_io[0, 0, 0])
    for i in range(0, frames):
        data = raw_io[i,]
        for j in wavelength_map.keys():
            index = 24 - (j[0] * 5) + (j[1] - 4)
            wavelength_frame_avg[index, i] = np.mean(data[j[0]::5, j[
                                                                       1]::5])  # map frequencies in increasing order of the array ie., index 0 will have 600...index 24 will have 975
    if DEBUG_NOISE:
        #print("Noise shape:",wavelength_noise_mean.shape)
        ia.debug_plot_waves_single(wavelength_noise_mean,"Average of Wavelength 0 across all frames in noise")

#readVideoFile() reads the video file and returns the mean of every available wavelength for each frame present and also the normalizes the frequencies across frame
#The frequencies are mapped in ascending order to the array index
def  readVideoFile(video_path):
    global frame_discard_end
    global frame_discard_start
    global wavelength_noise_mean

    raw = io.imread(video_path)
    raw_io = raw[frame_discard_start:(raw.shape[0]-frame_discard_end),:]
    frames = raw_io.shape[0]

    wavelength_frame_avg = np.zeros(shape=(25,frames))
    wavelength_normalized = np.zeros(shape=(25,frames))
    wavelength_sums = np.zeros(shape=(25))

    for i in range(0,frames):
        data = raw_io[i,]
        for j in wavelength_map.keys():
            index = 24-(j[0]*5)+(j[1]-4)
            wavelength_frame_avg[index,i] = np.mean(data[j[0]::5,j[1]::5])  #map frequencies in increasing order of the array ie., index 0 will have 600...index 24 will have 975

    if DEBUG_NOISE:
        ia.debug_plot_waves_single(wavelength_frame_avg[0],"Average of Wavelength 0 across all frames before noise elimation")

    if PROCESS_NOISE == 1:
        for i in range(frames):
            wavelength_frame_avg[:,i] =  wavelength_frame_avg[:,i] - wavelength_noise_mean[:]

        if DEBUG_NOISE:
            #since almost all wavelengths are of similar waveform plot the 0th wave alone
            ia.debug_plot_waves_single(wavelength_frame_avg[0],"Average of Wavelength 0 across all frames after noise elimation")

    writeCSVFile(wavelength_frame_avg)

    for j in range(0,len(wavelength_map)):
        wavelength_sums[j] = int(np.sum(wavelength_frame_avg[j,:]))
        wavelength_normalized[j,:] = normalizeData(wavelength_frame_avg[j,:])
    for j in range(0, len(wavelength_map)):
        wavelength_sums[j] = int(np.sum(np.abs(wavelength_normalized[j, :])))
    if DEBUG_INPUT_DATA:
        ia.debug_bar_plot_data(wavelength_sums,np.asarray(wavelengths, dtype=np.float32),"Sum of frames for each wavelength", "Wavelength","Sum of values across all frames")
        ia.debug_plot_waves(wavelength_normalized,"Normalized wavelengths")
        ia.debug_sub_plot_waves(wavelength_normalized)
    #Band pass filter the normalized data
    wavelength_normalized = bandPassFilter(wavelength_normalized)
    return wavelength_frame_avg, wavelength_normalized,frames

#This function performs Band pass filtering
def bandPassFilter(norm_waves):
    b, a = butter_pass()
    y = lfilter(b, a, norm_waves)
    return y

#This function implements butter pass filter
def butter_pass():
    order =3
    lowcut =40/60
    highcut = 180/60
    nyq = 0.5 * frame_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#This function performs the FFT and extracts the maximum frequency component on the ICA data.
def FFT_Extract_Max_Freq(Component,fps,Windowsize):
   power =[]
   bpm = []
   print("toatl components",len(Component))
   length = len(Component)
   if mean_fft==1:
       length=1
   for i in range(0,length):
       if mean_fft==0:
           New_Component = Component[i,:]
       else:
           New_Component = Component[:]
       componentFftPower = np.abs(np.square(np.fft.fft(New_Component)))
       componentFftFreq = np.fft.fftfreq(n=Windowsize, d=1 / fps)
       componentFftFreq = 60 * componentFftFreq
       FreqLocation=np.where((componentFftFreq > 45 ) & (componentFftFreq < 180))
       #Enable the following line if there are max peaks in between particular frequencies to eliminate them.
       #FreqLocation = np.where(((componentFftFreq > 45) & (componentFftFreq < 110)) | ((componentFftFreq > 120) & (componentFftFreq < 180)))
       componentFftFreq = componentFftFreq[FreqLocation]
       componentFft = componentFftPower[FreqLocation]
       if DEBUG_FFT:
           plt.figure(4)
           plt.title('FFT'+str(i))
           plt.xlabel('frequency')
           plt.ylabel('power')
           plt.plot( componentFftFreq,componentFft,)
           plt.show()
       power.append(np.max(componentFft))
       bpm.append(componentFftFreq[np.argmax(componentFft)])
   return bpm[power.index(max(power))],max(power),power.index(max(power))


#This function performs ICA with extended infomax algorithm
def run_ica_mne(data):
    ica = ICA(n_components=no_of_components, method='extended-infomax', random_state=0,max_iter=1000)
    ica.fit(data)
    return ica

#This function performs Fast ICA
def run_ica_fast(channels):
    icamatrix = np.c_[channels[0,:],channels[1,:],channels[2,:],channels[3,:],channels[4,:]]
    ica = FastICA(n_components=5,random_state=2,max_iter=2000)
    return ica.fit_transform(icamatrix)

#This function processes individual multispectral data
def ProcessMultiSpecVideo(video_path):
    # read video file
    mean_wavelengths, norm_wavelength, no_of_frames = readVideoFile(video_path)

    filtered_wave = norm_wavelength

    # Set all the data required to perform ICA, raw_array is used to perform ICA with extended infomax and fast_ica_inp is used to perform Fast ICA.
    ch_names = [str(i) for i in wavelengths]
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg']
    info = mne.create_info(ch_names=ch_names[start_comp:end_comp], sfreq=10, ch_types=ch_types[start_comp:end_comp])
    raw_array = RawArray(filtered_wave[start_comp:end_comp, :], info)
    fast_ica_inp = filtered_wave[start_comp:end_comp, :]

    # Perform ICA based on the ica type selected
    ica_data = 0
    if ica_type == 0:  # perform ICA with extended infomax algorithm
        print("ica")
        ica_comp = run_ica_mne(raw_array)
        ica_data = ica_comp.get_sources(inst=raw_array).get_data()
    #else:  # perform Fast ICA
        #ica_data = geticacomponents(fast_ica_inp).transpose()

    # When bruteforce FFT is enabled then 10 ICA component sequences will be sent together for FFT to find combination which will yield best result.
    if BRUTEFORCE_FFT == 1:
        for i in range(0,15):
            write_out.write("FFT on "+str(i)+" to"+str(i+10)+"\n")

            bpm, power, comp = FFT_Extract_Max_Freq(ica_data[i:i+10,:], frame_rate, no_of_frames)

            write_out.write("Max BPM for " + str(loop) + " is " + str(bpm) + " with power " + str(power) + " Component " + str(
                comp) + "\n\n\n")
    else:
        print("Shape : ",np.mean(ica_data[:, :],axis=0).shape)
        if mean_fft == 1:
            new_ica_data = np.mean(ica_data[:, :],axis=0)
        else:
            new_ica_data = ica_data
        bpm, power, comp = FFT_Extract_Max_Freq(new_ica_data, frame_rate, no_of_frames)
        write_out.write(
            "Max BPM for  is " + str(bpm) + " with power " + str(power) + " Component " + str(
                comp) + "\n\n\n")
        return bpm,power,comp

#When all the files within a directory needs analysis
def multipleVideoFiles():
    video_files = getVideoFileList()
    for video_type in range(2):
        video_list = video_files
        for video_id in range(len(video_list)):
            write_out.write("Data " + video_list[video_id] + "\n")
            ProcessMultiSpecVideo(video_list[video_id])

# When only one multispectral data needs to be analysed
def singleVideoFile():
    print(folder)
    if PROCESS_NOISE==1:
        readNoiseFile(folder + '/Inc/Karthick_Resting_Normal_Light_only.tif')
    bpm, power, comp = ProcessMultiSpecVideo(folder+"/Inc/Karthick_resting_no_crop.tif")
    print("Max BPM is " + str(bpm) + " with power " + str(power) + " Component " + str(
        comp) + "\n\n\n")

##############################################################################################
#main function
#############################################################################################

write_out = open("Datas.txt", "w")
if MULTIPLE_VIDEO == 0:
    singleVideoFile()
else:
    multipleVideoFiles()
write_out.close()



