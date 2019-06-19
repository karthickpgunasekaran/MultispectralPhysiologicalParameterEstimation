import csv
import numpy as np
folder = "Store_csv/fps/60/"

downsample=6 #no of times the data should be downsampled
downsampled_file ="008_files_downsample.csv" #post downsampling file name
original_file ='008_files.csv' #give the name of the file to be downsampled

store_csv = open(folder+downsampled_file, "w", newline='')
store_writer = csv.writer(store_csv)
with open(folder+original_file) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	i =0
	list_save =[]
	list_save2 = []
	list_save6 =[]   
	if downsample ==2:
		for row1 in readCSV:
			if i%2==0:
				list_save = row1
			else:
				mean_val = (np.asarray(row1, dtype=np.float32)+np.asarray(list_save,dtype=np.float32))/2
				store_writer.writerow(mean_val)
			i=i+1
	elif downsample ==3:
		for row1 in readCSV:
			if i%3==0:
				list_save = row1
			elif i%3==1:
				list_save2 = row1
			else:
				mean_val = (np.asarray(row1, dtype=np.float32)+np.asarray(list_save,dtype=np.float32)+np.asarray(list_save2,dtype=np.float32))/3
				store_writer.writerow(mean_val)
			i=i+1
	else:
		for row1 in readCSV:
			if i%downsample==(downsample-1):
				sum_v=np.asarray(row1, dtype=np.float32)
				for row in list_save6:
					sum_v = sum_v + np.asarray(row, dtype=np.float32)
				mean_val = sum_v/downsample
				store_writer.writerow(mean_val)
			elif i%downsample==0:
				list_save6.append(row1)
			i=i+1
store_csv.close()       

