import numpy as np
import struct

def extract_data_images(filename):
    with open(filename,'rb') as f:
        #collect info from file header
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        #reshape to a format that the neural network takes, a list of flat images
        images = np.fromfile(f,dtype = np.uint8)
        images = images.reshape(num_images,num_rows*num_rows)/255
    return images

def extract_data_labels(filename):
    def int_to_output_list(number):
        output_list = np.zeros(10)
        output_list[number] = 1
        return output_list

    with open(filename,'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_lables = struct.unpack('>I',f.read(4))[0]
        numbers = np.fromfile(f,dtype = np.uint8)
        python_list = numbers.tolist()
        for i in range(len(python_list)):
            python_list[i] = int_to_output_list(python_list[i])
        numbers = np.array(python_list)
        return numbers