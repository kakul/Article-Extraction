import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from skimage.filter.rank import entropy,otsu
from skimage.filter import threshold_otsu
from skimage.morphology import square,rectangle,label,closing,disk,binary_erosion,opening
from skimage.color import label2rgb,rgb2gray
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from skimage import io
i=rgb2gray(io.imread('z.jpg'))
t=threshold_otsu(i)
i=i>t
z=binary_erosion(1-i,square(3))
i=1-z
i=entropy(i,rectangle(10,1))
t=threshold_otsu(i)
c=i>t
c=closing(c,rectangle(4,28))
b=c.copy()
clear_border(b)
l=label(b)
z=np.logical_xor(c,b)
l[z]=-1
iol=label2rgb(l,image=i)
fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(6,6))
ax.imshow(iol)
for region in regionprops(l,['Area','BoundingBox']):
	if region['Area']<3500:
		continue
	minr,minc,maxr,maxc=region['BoundingBox']
	rect=mp.Rectangle((minc,minr),maxc-minc,maxr-minr,fill=False,edgecolor='red',linewidth=1)
	ax.add_patch(rect)

plt.show()

