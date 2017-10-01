
# coding: utf-8

# In[1]:

import pandas as pd

open('File Name.txt','r').readlines()[:1][0][:-2]


# In[2]:

filename = []
for line in open('File Name.txt','r'):
    singlefile = str(line[:-2])
    filename.append(singlefile)
    


# In[3]:

filename[-10:]


# In[4]:

imageurl = []
for line in open('File link.txt','r'):
    singlefile = str(line[:-2])
    imageurl.append(singlefile)
imageurl[-5:]


# In[8]:

randomdict = {}
randomdict['yes'] = 1
print randomdict


# In[9]:

import urllib
downloadFail = {}
def downloadimage(imageurl, imagename):
    for index, url in enumerate(imageurl):
        print "downloading " + imagename[index]
        try:
            urllib.urlretrieve(url, "./imagefiles/" + imagename[index] + ".jpg")
            print imagename[index] + " downloaded."
        except:
            print imagename[index] + " failed."
            downloadFail[imagename[index]] = url
            
        


# In[ ]:

downloadimage(imageurl, filename)


# In[ ]:



