from skimage import io
from skimage.filter import threshold_otsu
from skimage.color import rgb2gray
from skimage.transform import hough_line,hough_line_peaks,probabilistic_hough_line,rotate
from skimage.morphology import square,rectangle,label,closing,disk,binary_erosion,opening
import matplotlib.pyplot as plt
import numpy as np
i=rgb2gray(io.imread('s.jpg'))
t=threshold_otsu(i)
z=i>t
b=binary_erosion(1-z,rectangle(3,5))
g=np.logical_xor(b,z)
y=binary_erosion(1-g,rectangle(2,2))
o=binary_erosion(y,rectangle(1,5))
h,a,d=hough_line(o)
rows,cols=i.shape
s=hough_line_peaks(h,a,d,num_peaks=3)
f=rotate(i,90+np.mean(np.rad2deg(s[1])))
o1=rotate(o,90+np.mean(np.rad2deg(s[1])))
h,a,d=hough_line(o1)
s1=hough_line_peaks(h,a,d,num_peaks=3)
f1=rotate(f,90+np.mean(np.rad2deg(s1[1])),resize=True)
io.imsave('output1.jpg',f1)

	
