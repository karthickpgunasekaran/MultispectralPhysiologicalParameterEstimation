from skimage import io
import numpy as np
import os
import csv
import psutil
#Set folder paths
folder = "Testing"          #set folder path
file_type = 1 # 0- tiff videos,1- tiff image sequence 
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

#Multispectral video data processing settings
frame_discard_start = 0
frame_discard_end = 0

#Get list of all files in the given directory.
def getFilesList(dirName):
    listOfFile = sorted(os.listdir(dirName))
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        allFiles.append(fullPath)
        #print("Full path ",fullPath)
    return allFiles

def getImageSequence(dirName='Testing'):
    listOfFile = [dI for dI in sorted(os.listdir(dirName)) if os.path.isdir(os.path.join(dirName,dI))]
    #listOfFile = sorted(os.listdir(dirName))
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        allFiles.append(fullPath)
    print("All the files:",allFiles)
    return allFiles

#Intermediate function to call the get all files paths from the directory
def getVideoFileList():
    global  folder
    videos_paths = getFilesList(folder)
    return  videos_paths

# When only one multispectral data needs to be analysed
def singleVideoFile():
    raw = io.imread(folder + "")

#readVideoFile() reads the video file and returns the mean of every available wavelength for each frame present and also the normalizes the frequencies across frame
#The frequencies are mapped in ascending order to the array index
def  readVideoFile(video_path,name):
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
    wavelength_frame_avg = wavelength_frame_avg.transpose()
    writeCSVFile(wavelength_frame_avg,name)
def readSequenceData(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def multipleVideoFiles():
    if file_type==0:
    	video_files = getVideoFileList()
    else:
        video_files = getImageSequence()
    for video_id in range(0,len(video_files)):
        print(psutil.virtual_memory())
        print("video id:",video_files[video_id])
        if file_type==0:
             raw = io.imread(video_files[video_id])
        else:
             raw = readSequenceData(video_files[video_id])
        readVideoFile(raw,video_files[video_id])
        raw=0

# Write CSV data to the file
def writeCSVFile(data,name):
    splits = name.split("/")
    names = splits[len(splits)-1].split(".")
    myFile = open("storecsv_lights/"+names[0]+'.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)
	
###########################MAIN###################################
multipleVideoFiles()

