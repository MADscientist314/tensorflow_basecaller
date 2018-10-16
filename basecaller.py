#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import h5py
import signal
import fast5
import fastq
import re


# In[2]:


sig = []
for x in range(10):
    a = np.loadtxt("signal/signal_{}.txt".format(x))
    sig.append(a)

print(sig[0])


# In[ ]:





# In[3]:


a = []
for x,y in enumerate(sig):
    b=[np.array(sig[x]).size]
       #, sig[x].ndim, sig[x].dtype)# gets the  number of resistances from each signal
    a.append(b)
print(a)
        
        


# In[5]:


c = np.zeros((10,1))
d = np.full((10,1),2)
if d.any()>c.any(): # I'm trying to divid the resistances in half every time and its not working...
    #np.divide(b,d)
    b=c
    print(c)


# In[6]:


# This cell imports the bases, converts them to a list, and converts them to a code of 0123 instead of ATCG
f = open("fasta/sampled_read.fasta","r") 
a = f.read()
b = (a.split(">", 11))
base = [re.sub(">|\n|\d", "",str) for str in b]
baseA = [re.sub("A","0",str) for str in base]
baseT = [re.sub("T","1",str) for str in baseA]
baseG = [re.sub("G","2",str) for str in baseT]
base_coded = [re.sub("C","3",str) for str in baseG]
print(base_coded)


# In[ ]:


base_array=(np.array(base_coded)) # this code is supposed to covert the list to a 1D array
#print(base_array[0:])
print(base_array.shape)
#for x,y in enumerate(base_array):
#    print(np.array(base_array[x]).size)
#    print(base_array[x].shape)
#    print(base_array[x].dtype)
  #  print(base_array[x])


# In[ ]:


#np.concatenate((sig,base_array), axis=0)


# In[ ]:


base_transpose = np.transpose(base_array) # you cant transpose a 1d arry into a 1d array but I was trying to append it to have the resistances???
base_transpose.T
base_transpose.shape
#for x in range(10):
    #print("base_array"(a))
 #   print(len(base_array[x]))
    #print("base_transpose length is:")
  #  print(len(base_transpose[x]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




