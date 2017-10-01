
# coding: utf-8

# In[28]:

import os, sys
from PIL import Image


# In[37]:

#The desired value of the side of the square image after cropping
desire_value = 256

#resize shortside to the desire value while remain ratio
def ratio_resize (longside, shortside): 
    long_side_precent = (desire_value/float(shortside))
    longside = int(float(longside)*long_side_precent)
    return longside

# run the ratio_resize and then crop the longside to the desired value 
def ratio_cropping (image):
    if image.size[0] > image.size[1]:
        image = image.resize((ratio_resize(image.size[0], image.size[1]),desire_value), Image.ANTIALIAS)
        image = image.crop(((image.size[0]/2 - desire_value/2),0,(image.size[0]/2 + desire_value/2),desire_value))
    elif image.size[0] < image.size[1]:
        image = image.resize((desire_value, ratio_resize(image.size[1], image.size[0])), Image.ANTIALIAS)
        image = image.crop((0,(image.size[1]/2 - desire_value/2),desire_value,(image.size[1]/2 + desire_value/2)))
    else: 
        image = image.resize((desire_value,desire_value))
    return image

# loop through current directory and save the resize image to "resize" folder (need to create in advance)
for infile in os.listdir("./"):
    outfile = "./resize/" + os.path.splitext(infile)[0] + ".jpg"
    try:
        im = ratio_cropping(Image.open(infile))
        im.save(outfile, "JPEG")
    except IOError:
        print "cannot create thumbnail for '%s'" % infile


# In[ ]:



