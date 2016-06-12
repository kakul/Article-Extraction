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
i=rgb2gray(io.imread('fog.jpg'))
t=threshold_otsu(i)
i=i>t
z=binary_erosion(1-i,square(3))
b=z.copy()
clear_border(b)
l=label(b)
x=np.logical_xor(z,b)
l[x]=-1
iol=label2rgb(l,image=i)
fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(6,6))
ax.imshow(iol)
for region in regionprops(l,['Area','BoundingBox']):
    if region['Area']<1500:
        continue
    minr,minc,maxr,maxc=region['BoundingBox']
    rect=mp.Rectangle((minc,minr),maxc-minc,maxr-minr,fill=False,edgecolor='red',linewidth=1)
    ax.add_patch(rect)


plt.show()
