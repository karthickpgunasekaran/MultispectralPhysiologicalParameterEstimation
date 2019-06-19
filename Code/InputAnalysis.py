
import matplotlib.pyplot as plt
import numpy as np

def debug_bar_plot_data(inp,inp2,title,x,y):
   # this is for plotting purpose
    if inp2 is None:
        index = np.arange(len(inp))
    else:
        index = inp2
    plt.bar(index, inp)
    plt.xlabel(x)
    plt.ylabel(y)
    #plt.xticks(index, inp)
    plt.title(title)
    plt.show()
    plt.close()

def debug_plot_waves_single(wave,name):
    #for i in range(1):
    plt.plot(wave)
    plt.title(name)
    plt.show()
    plt.close()

def debug_plot_waves(wave,name,x_axis,y_axis):
    for i in range(25):
        plt.plot(wave[i])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(name)
    plt.show()
    plt.close()

def debug_sub_plot_waves(wave):
    for j in range(0,5):
        plt.figure(j+1)
        plt.xlabel('Frame(s)')
        for i in range(0, 5):
            index =(j*5)+i
            plt.subplot(5, 1,i+1)
            #plt.title('Input Data')
            #plt.ylabel('Mean Values')
            plt.plot(wave[index])
        plt.show()
    plt.close()

def debug_PlottingRawData(filtered_wave, videoname):
   videoname = videoname.replace('tif','png')
   plt.figure(1)
   for i in range(0,25):
       plt.subplot(25, 1, i+1)
       plt.plot(filtered_wave[i])
   plt.savefig(videoname) #'{}'.format(videoname)
   plt.close()