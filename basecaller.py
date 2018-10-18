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
import matplotlib


# In[2]:


"""
this function divides the length of whatever 
size the list is by 4 for a 4mer
"""
def BaseFunction(x,y): 
    return((len(x)//y),y)


# In[3]:


"""
this function divides the length of whatever 
size the list is by 40 for a 40mer
"""
def SigFunction(x,y): 
    return((y),((x//y)))


# In[4]:


"""
this function provides information about the array    
"""
def array_inspect(x):
    print ("Shape is",(x.shape))
    print(("Length is",len(x)))
    print(("Dimension is",x.ndim))
    print(("Total Size is",x.size))
    print(("Type is",x.dtype))
    print(("Type Name is",x.dtype.name))
    #print(("Mean is",(x).mean))


# In[5]:


"""
this function normalizes the raw signal resistances
from each read by dividing by the mean(or std?)
"""
def normalize(x,y):
    z=np.divide(x,y)
    #np.savetxt("NormalizedSigArray_{}.csv".format(),(z), delimiter=",")
    return z


# In[6]:


def divider(x,y):
    a=np.divide(x,y)
    return a


# In[7]:


#subsample=[]
#c = np.zeros((10,1))
#print(c)
#d = np.full((10,1),2)
#print(d)
#stride
#for x in 
#if d.any()>c.any(): # I'm trying to divid the resistances in half every time and its not working...
    #np.divide(b,d)
#    b=c
#    print(c)


# In[8]:


"""
This cell does the following
    1)Imports the bases
    2)trims off the new lines and digits
    3)converts them to a list
    4)converts them to a code of 0123 instead of ATCG
"""
f = open("fasta/sampled_read.fasta","r") #opens the file with the reads
a = f.read()
b = (a.split(">", 11))
base = [re.sub(">|\n|\d", "",str) for str in b]
baseA0 = [re.sub("A","0",str) for str in base]
baseT1 = [re.sub("T","1",str) for str in baseA0]
baseG2 = [re.sub("G","2",str) for str in baseT1]
base_coded = [re.sub("C","3",str) for str in baseG2]
#print(base_coded)


# In[9]:


"""
The purpose of this code is the following
1)Convert the coded reads to integer form 
2)Create an array inside the array with each read as a row
"""
reads=[]
readslength=[]
for x in range(10):#len(base_coded)):
    l=list(int (i) for i in base_coded[x])
    d=np.asarray(l)
    d.resize((BaseFunction(l,4))) #Enter the list and the desired kmer for the function
    reads.append(d) # store the reads in an array called reads
    #array_inspect(reads[x])
    c=len(reads[x])
    #print(c)
    readslength.append(c) #store the amt of kmer reads in readlength
#array_inspect(reads[x])


# In[10]:


"""
The cell does the following:
    1) Import the signal level data
    2) Normalize it to the mean fof the signal
    3) Appends the signal data from each read into a row of an array given the read count
"""
means = [] 
std = []
sig = []
for x in range(10):
    a=np.loadtxt("signal/signal_{}.txt".format(x)) # load the signal
    m=a.mean() #obtain the average signal from the read
    b=normalize(m,a) # divide each resistance by the average to normalize 
    c=b.size #obtains the number of resistance signals per read
    #print("resist",c)
    d=readslength[x] #obtains the (4)kmers count per read
    #print("kmer",d)
    e=SigFunction(c,d) #figures out the average amount of resistances per kmer
    #print("Signal",x,"2D array size={}".format(e))
    b.resize(e) # Insert the (#bases,#sig) changes the signal level n signals wide
    #we can change this number later to be a sliding window for maching learning
    sig.append(b)
    #array_inspect(sig[x])
    #np.savetxt('sig_array.csv', b, delimiter=',')


# In[12]:


"""The purpose of this cell is to merge the reads and the raw signals together"""
concat=[]
for x in range(10):
    a=reads[x].ndim 
    b=sig[x].ndim
    c=np.concatenate((reads[x],sig[x]),axis=1)
    #array_inspect(c)
    concat.append(c)
    np.savetxt("concat_1.1.csv",c, delimiter=",")


# In[13]:


"""
Any cell under this cell is trash code or experimental
"""


# In[14]:


"""
IDEA: USE AMPLICON SEQUENCING DATA TO TRAIN THE MODEL BC IT HAS A KNOWN LENGTH IN BP 
OMG USE THE PECON AMPLICON DATA BC IT IS ALL EXACTLY 3KB IN SIZE!!!!!!!!!
"""


# In[ ]:




