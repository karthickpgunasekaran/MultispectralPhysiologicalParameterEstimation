
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
import gc
import psutil
from numpy import genfromtxt

#Set input file format
CSV = 1

#Set the following manually before run

DEBUG_NOISE =0       #This variable is set to plot all data related to noise data in case noise da is given as input.
PROCESS_NOISE = 0    #This variable is to enable the noise data processing
DEBUG_INPUT_DATA = 0 #This variable is set to plot all data related to input data.
DEBUG_FFT = 0        #This variable is set to plot all plots of FFT data.
BRUTEFORCE_FFT = 0   #This variable is set to try all combination of FFT's.
MULTIPLE_VIDEO = 1  #This variable is set to 0 for single video and set to 1 in case multiple files in folder.
ML_GEN_DATA =0

#Set folder paths
folder = "Inputs_ippg/fps/60_downsample_60fps_to_20fps"      #set folder path
video_folder =folder+""       #In case only sub folders need to be modified from the base folder
file_id ="/005.tif"   #Set this variable when using single video file only
mean_fft =0

#Multispectral video data processing settings
frame_discard_start = 0
frame_discard_end = 0
#
slide_frame_start_discard = 0
slide_frame_end_discard   = 0
#
slide_time = 90 #seconds
frame_rate =20
sliding_window_size = frame_rate*slide_time
slide_inc = frame_rate * 5

#Set ICA details
NO_ICA = 0
no_of_components = 25  #no of components in the ICA
ica_type =0            #0- Extended infomax, 1 - fast ICA
pca_components = 13
ica_iter  =40000
ica_tolerance = 0.27
wavelength_noise_mean = np.zeros(shape=(25))
ml_data=[]

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
def normalizeData(data,entire):
    data_norm = np.zeros(shape=(len(data)))
    data_norm = (data - np.mean(data))/np.std(data)
    return data_norm

#Get list of all files in the given directory.
def getFilesList(dirName):
    listOfFile = sorted(os.listdir(dirName))
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)

        if os.path.exists(fullPath+"/"+entry+".csv"):
            allFiles.append(fullPath+"/"+entry+".csv")
    return allFiles

#Intermediate function to call the get all files paths from the directory
def getVideoFileList():
    global  folder
    videos_paths = getFilesList(folder)
    return  videos_paths

# Write CSV data to the file
def writeCSVFile(data):
    myFile = open('Outputs_ippg/Mean_wavelength.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)

#This function is used in case there exists a background noise video without subject and only the noise needs to be eliminated
def readNoiseFile(video_path):
    global wavelength_noise_mean
    raw = io.imread(video_path)
    frame_discard = 0
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
        ia.debug_plot_waves_single(wavelength_noise_mean,"Average of Wavelength 0 across all frames in noise")

#readVideoFile() reads the video file and returns the mean of every available wavelength for each frame present and also the normalizes the frequencies across frame
#The frequencies are mapped in ascending order to the array index
def  readVideoFile(video_path):
    global frame_discard_end
    global frame_discard_start
    global wavelength_noise_mean

    raw = video_path#io.imread(video_path)
    print("TOTAL NO OF FRAMES:", raw.shape[0])
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

    if DEBUG_NOISE and 0:
        ia.debug_plot_waves_single(wavelength_frame_avg[0],"Average of Wavelength 0 across all frames before noise elimation")

    if PROCESS_NOISE == 1:
        for i in range(frames):
            wavelength_frame_avg[:,i] =  wavelength_frame_avg[:,i] - wavelength_noise_mean[:]

        if DEBUG_NOISE and 0:
            #since almost all wavelengths are of similar waveform plot the 0th wave alone
            ia.debug_plot_waves_single(wavelength_frame_avg[0],"Average of Wavelength 0 across all frames after noise elimation")

    writeCSVFile(wavelength_frame_avg)
    print("MAX values in time domain: ", np.max(wavelength_frame_avg[:, 0]), " wavelength:",
          np.argmax(wavelength_frame_avg[:, 0]))
    check_wave = np.argmax(wavelength_normalized[:, 0])
    for j in range(0,len(wavelength_map)):
        wavelength_sums[j] = int(np.sum(wavelength_frame_avg[j,:]))
        wavelength_normalized[j,:] = normalizeData(wavelength_frame_avg[j,:],wavelength_frame_avg)

    for j in range(0, len(wavelength_map)):
        wavelength_sums[j] = int(np.sum(np.abs(wavelength_normalized[j, :])))
    if DEBUG_INPUT_DATA:
        ia.debug_plot_waves(wavelength_normalized,"Normalized wavelengths","Frames","Mean Normalized Value")

    #Band pass filter the normalized data
    wavelength_normalized = bandPassFilter(wavelength_normalized)
    return wavelength_frame_avg, wavelength_normalized,frames

#reads the video csv file and returns the mean of every available wavelength for each frame present and also the normalizes the frequencies across frame
#The frequencies are mapped in ascending order to the array index
def readCSVFile(wavelength_frame_avg):
    frames = wavelength_frame_avg.shape[1]
    wavelength_normalized = np.zeros(shape=(25, frames))
    wavelength_sums = np.zeros(shape=(25))
    print("MAX values in time domain: ", np.max(wavelength_frame_avg[:, 0]), " wavelength:",
          np.argmax(wavelength_frame_avg[:, 0]))
    check_wave = np.argmax(wavelength_normalized[:, 0])
    for j in range(0,len(wavelength_map)):
        wavelength_sums[j] = int(np.sum(wavelength_frame_avg[j,:]))
        wavelength_normalized[j,:] = normalizeData(wavelength_frame_avg[j,:],wavelength_frame_avg)

    for j in range(0, len(wavelength_map)):
        wavelength_sums[j] = int(np.sum(np.abs(wavelength_normalized[j, :])))
    if DEBUG_INPUT_DATA:
        ia.debug_plot_waves(wavelength_normalized,"Normalized wavelengths","Frames","Mean Normalized Value")

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

StoreFreqPower = np.zeros(shape=(2000))
saveFreq = np.zeros(shape=(2000))
saveMaxFft = np.zeros(shape=(25))
#This function performs the FFT and extracts the maximum frequency component on the ICA data.
def FFT_Extract_Max_Freq(Component,fps,Windowsize):
   global StoreFreqPower,saveFreq,saveMaxFft,ml_data
   power =[]
   bpm = []
   index =[]

   length = len(Component)

   for i in range(0,length):
       New_Component = Component[i,:]
       componentFftPower = np.abs(np.square(np.fft.fft(New_Component)))
       componentFftFreq = np.fft.fftfreq(n=Windowsize, d=1 / fps)
       componentFftFreq = 60 * componentFftFreq
       FreqLocation=np.where((componentFftFreq > 45 ) & (componentFftFreq < 180))

       componentFftFreq = componentFftFreq[FreqLocation]
       componentFft = componentFftPower[FreqLocation]

       saveFreq[:len(componentFftFreq)] = componentFftFreq
       StoreFreqPower[:len(componentFftFreq)] = StoreFreqPower[:len(componentFftFreq)] + componentFft

       saveMaxFft[i] = componentFftFreq[np.argmax(componentFft)]


       power.append(np.max(componentFft))
       bpm.append(componentFftFreq[np.argmax(componentFft)])
       index.append(np.argmax(componentFft))

       if DEBUG_FFT:
           plt.figure(4)
           plt.title('FFT'+str(i))
           plt.xlabel('BPM')
           plt.ylabel('Power')
           plt.plot( componentFftFreq,componentFft,)
           plt.show()
       if i==no_of_components-1 and DEBUG_FFT:
           plt.figure(6)
           plt.title('FFT plots')
           plt.xlabel('FFT Components')
           plt.ylabel('Bpm')
           plt.plot(np.arange(0,25),saveMaxFft,'o' )
           plt.show()
           plt.figure(7)
           plt.title('Max FFT power')
           plt.xlabel('Wavelength')
           plt.ylabel('Power')
           plt.plot(np.arange(0, 25), np.asarray(power), 'or')
           plt.show()
           print("Save max freq:", bpm)
           print("Save max power:", power)
           print("Median BPM: ",np.median(np.sort(saveMaxFft)))

   if ML_GEN_DATA == 1:
           new_data = ml_data + index + (bpm) + (power)
           ml_writer.writerow(new_data)


   return bpm[power.index(max(power))],max(power),power.index(max(power))

#This function performs ICA with extended infomax algorithm
def run_ica_mne(data):
    print("no_of_components:",no_of_components," pca comp:",pca_components," Tol:",ica_tolerance," Ica iter:",ica_iter)
    #ica = ICA(n_components=pca_components, method='fastica',max_iter=ica_iter,random_state=0)
    ica = ICA(n_components=pca_components, method='fastica', max_iter=ica_iter, random_state=0,fit_params={'tol': ica_tolerance})
    ica.fit(data)
    return ica

'''
#This function performs Fast ICA
def run_ica_fast(channels):
    print("icamatrix:",channels.shape)
    icamatrix = np.c_[channels[0, :], channels[1, :], channels[2, :], channels[3, :], channels[4, :],channels[0, :], channels[1, :], channels[2, :], channels[3, :], channels[4, :],channels[0, :], channels[1, :], channels[2, :], channels[3, :], channels[4, :]]
    ica = FastICA(n_components=1100,random_state=2,max_iter=ica_iter,tol=ica_tolerance)
    save = ica.fit_transform(icamatrix)
    print("save:",save.shape)
    return save
'''
# This function was just created to check the PCA components and variance
def check_pca(x):
    covar_matrix = PCA(n_components=25)
    covar_matrix.fit(x)
    variance = covar_matrix.explained_variance_ratio_
    var = np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3) * 100)
    print(var)

#This function processes individual multispectral data
def ProcessMultiSpecVideo(video_path,leave_one_out,single_freq):

    # read video file
    if CSV==1:
        mean_wavelengths, filtered_wave, no_of_frames = readCSVFile(video_path)
    else:
        mean_wavelengths, filtered_wave, no_of_frames = readVideoFile(video_path)
    check_pca(filtered_wave)

    '''
    if single_freq>0:
        filtered_wave = np.zeros((1, filtered_wave.shape[1]))
        filtered_wave[0] = filtered_wave[single_freq-1]
    
    if leave_one_out == 25:
        filtered_wave = filtered_wave[:leave_one_out-1]
    elif leave_one_out ==1:
        filtered_wave = filtered_wave[leave_one_out:]
    elif leave_one_out > 0:
        filtered_wave = np.concatenate((filtered_wave[:leave_one_out-1], filtered_wave[leave_one_out:]),0)
    '''

    # Perform ICA based on the ica type selected
    ica_data = 0
    if NO_ICA==0:
        # Set all the data required to perform ICA, raw_array is used to perform ICA with fast_ica_inp is used to perform Fast ICA.
        ch_names = [str(i) for i in wavelengths]
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                    'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                    'eeg']
        info = mne.create_info(ch_names=ch_names[:no_of_components], sfreq=10, ch_types=ch_types[:no_of_components])
        raw_array = RawArray(filtered_wave, info)
        ica_comp = run_ica_mne(raw_array)
        ica_data = ica_comp.get_sources(inst=raw_array).get_data()

    if NO_ICA == 1:
        ica_data = filtered_wave[:, :]
    bpm, power, comp = FFT_Extract_Max_Freq(ica_data, frame_rate, no_of_frames)
    write_out.write(str(bpm) + "\n")
    return bpm,power,comp

#When all the files within a directory needs analysis
def multipleVideoFiles():
    video_files = getVideoFileList()

    print(video_files)
    bpm_list= []
    for video_id in range(0,len(video_files)):
        print("video id:",video_files[video_id])
        write_out.write("Data " + video_files[video_id] + "\n")

        if CSV == 1:
            raw = genfromtxt(video_files[video_id], delimiter=',').transpose()#csv.reader(video_files[video_id])
        else:
            raw = io.imread(video_files[video_id])
        bpm, power, comp = ProcessMultiSpecVideo(raw,0,0)
        print("Max BPM :" + str(bpm) + " with power " + str(power) + " Component " + str(
            comp) + "\n\n\n")
        bpm_list.append(bpm)
        print(psutil.virtual_memory())
    return bpm_list

# When only one multispectral data needs to be analysed
def singleVideoFile():
    global saveFreq,StoreFreqPower,file_id

    print("Folder set to ",folder)
    write_out.write(file_id[1:] + "\n")
    if PROCESS_NOISE==1:
        readNoiseFile(folder + '/forsad_system_video.tif')
    raw = io.imread(folder + file_id)

    print("Raw file:",raw.shape)
    bpm, power, comp = ProcessMultiSpecVideo(raw,0,0)
    print("Max BPM :" + str(bpm) + " with power " + str(power) + " Component " + str(
        comp) + "\n\n\n")
    print("Sum MAX power: ",np.max(StoreFreqPower)," freq:",(saveFreq[np.argmax(StoreFreqPower)]))
def extractFileFolderNames(video_file):
    splits = video_file.split("/")
    folder =""
    for i in range(0,len(splits)-1):
        folder= folder+splits[i] +"/"
    return folder, splits[len(splits)-1]



def SlidingWindow(video_file='storecsv_inc/003.csv'):
    global saveFreq, StoreFreqPower,sliding_window_size,slide_inc,ml_data, ica_tolerance,pca_components,frame_rate

    if PROCESS_NOISE == 1:
        readNoiseFile(folder + '/forsad_system_video.tif')
    if CSV == 1:
        raw =  genfromtxt(video_file, delimiter=',').transpose()  #
    else:
        raw = io.imread(folder + video_file)

    length_raw = raw.shape[1]
    if length_raw < sliding_window_size:
        print("length_raw < sliding_window_size is True")
        return
    folder_name, file_name = extractFileFolderNames(video_file)
    print("FOlder:", folder_name, " File name:", file_name)
    raw = raw[:,slide_frame_start_discard:length_raw-slide_frame_end_discard]
    for slide in range(sliding_window_size,length_raw,slide_inc):
        write_out.write("Slide minute:"+ str(slide/sliding_window_size) + "\n")
        if ML_GEN_DATA==1:
            ml_data = [video_file, str(slide - sliding_window_size), str(slide),str(sliding_window_size)]
            #print("ml_data2 ",ml_data)
        bpm, power, comp = ProcessMultiSpecVideo(raw[:,slide-sliding_window_size:slide],0,0)

        slide_writer.writerow({'FOLDER NAME':folder_name,'FILE NAME':file_name,'START TIME' : str((slide - sliding_window_size)/frame_rate),'END TIME' : str(slide/frame_rate),'SLIDING WINDOW TIME':str(sliding_window_size/frame_rate),'TOL':ica_tolerance,'PCA':pca_components,'BPM':str(bpm)})

        print("Max BPM :" + str(bpm) + " with power " + str(power) + " Component " + str(
            comp) + "\n\n\n")

#Get all the videos in the pointed directory and perform sliding window on each video
def SlideAllVideos(pca_comp=13,tol=0.27,iter=25000):
    global sliding_window_size, ica_tolerance,pca_components,ica_iter
    video_files = getVideoFileList()
    ica_iter=iter
    pca_components = pca_comp
    ica_tolerance = tol
    for video_id in range(0, len(video_files)):
        SlidingWindow(video_files[video_id])

# Generate the inputs necessary for performing Regression/KNN
def MachineLearningInputGeneration():
    global sliding_window_size,ica_tolerance, pca_components, ica_iter
    video_files = getVideoFileList()
    slide = [1000]
    ica_iter = 25000
    pca_components = 13
    ica_tolerance = 0.27
    print("Slide:", video_files)
    bpm_list = []
    for iter in slide:
        sliding_window_size = iter
        for video_id in range(0, len(video_files)):
            print(" ",video_id)
            SlidingWindow(video_files[video_id])

# For a set of files in  the folder mentioned in code, given the ground truth this function displays the MAE
def efficiencyCalculation():
    groundTruth = [77,75,75,75,104,104,104,77,77]
    global ica_tolerance,pca_components,ica_iter
    ica_iter=25000
    comp = 13
    while comp < 14:
        pca_components = comp
        ica_tolerance = 0.27

        while ica_tolerance < 0.28:
            predList = multipleVideoFiles()
            ica_tolerance = ica_tolerance+0.02
            save_mae = np.mean(np.abs(np.asarray(groundTruth)-np.asarray(predList)))
            efficiency_writer.writerow({'PCA_COMP': str(comp), 'TOL': str(ica_tolerance), 'MAE': str(save_mae)})
            print("Tol:",ica_tolerance,"Mae:",save_mae)

        comp=comp+1

# Helper function to initialize the ML generate input file
def gen_list_ml_inp():
    index = ["FILE NAME", "START FRAME", "END FRAME", "SLIDING WINDOW SIZE"]
    for i in range(0,13,1):
        index.append('Index'+str(i))
    for i in range(0, 13, 1):
        index.append('Bpm' + str(i))
    for i in range(0, 13, 1):
        index.append('Power' + str(i))
    print(index)
    return index

# Vary tolerance and PCA components
def SlideBruteVideos():
    comp = 3
    while comp < 26:
        ica_tol = 0.27#0.001
        while ica_tol < 0.28:
            SlideAllVideos(comp,ica_tol)
            ica_tol = ica_tol+0.4
        comp=comp+1

# Close all the files
def close_files():
    write_out.close()
    efficiency_csv.close()
    slide_csv.close()
    ml_csv.close()

# Create and Initialize all the data files
def open_init_files():
    efficiency_csv = open("Outputs_ippg/EfficiencyCalc.csv", "w", newline='')
    efficiency_writer = csv.DictWriter(efficiency_csv,
                            fieldnames=['PCA_COMP', 'TOL', 'MAE'])
    efficiency_writer.writerow({'PCA_COMP':'PCA_COMP', 'TOL':'TOL', 'MAE':'MAE'})

    write_out = open("Outputs_ippg/Logs.txt", "a")

    slide_csv = open('Outputs_ippg/Sliding_window_'+str(slide_time)+'_secs.csv', 'w', newline='')
    slide_writer = csv.DictWriter(slide_csv,
                            fieldnames=['FOLDER NAME','FILE NAME', 'START TIME', 'END TIME', 'SLIDING WINDOW TIME', 'TOL', 'PCA',
                                        'BPM'])
    slide_writer.writerow({'FOLDER NAME':'FOLDER NAME','FILE NAME': 'FILE NAME', 'START TIME': 'START TIME', 'END TIME': 'END TIME',
                     'SLIDING WINDOW TIME': 'SLIDING WINDOW TIME', 'TOL': 'TOL', 'PCA': 'PCA', 'BPM': 'BPM'})


    ml_csv = open('Outputs_ippg/Ml_input_features.csv', 'w')
    ml_writer = csv.writer(ml_csv)
    ml_writer.writerow(gen_list_ml_inp())

    return write_out,efficiency_csv, efficiency_writer,slide_csv,slide_writer,ml_csv,ml_writer

##############################################################################################
#main function
#############################################################################################

write_out,efficiency_csv, efficiency_writer,slide_csv,slide_writer,ml_csv,ml_writer = open_init_files()
option = 2
if option==0:
    SlidingWindow()
elif option==1:
    SlideAllVideos()
elif option == 2:
    SlideBruteVideos()
elif option == 3:
    efficiencyCalculation()
elif option == 4:
    MachineLearningInputGeneration()
elif option == 5:
    singleVideoFile()
elif option == 6:
    multipleVideoFiles()



close_files()



