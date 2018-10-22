#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from numpy.random import rand
import csv
import h5py
import signal
import fast5
import fastq
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#options(no_) I'm trying to eliminate scientific notation


# In[16]:


"""
this function divides the length of whatever 
size the list is by 4 for a 4mer
"""
def BaseFunction(x,y): 
    return((len(x)//y),y)


# In[17]:


"""
this function divides the length of whatever 
size the list is by 40 for a 40mer
"""
def SigFunction(x,y): 
    return((y),((x//y)))


# In[18]:


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


# In[19]:


"""
this function normalizes the raw signal resistances
from each read by dividing by the mean(or std?)
"""
def normalize(x,y):
    z=np.divide(x,y)
    #np.savetxt("NormalizedSigArray_{}.csv".format(),(z), delimiter=",")
    return z


# In[20]:


def divider(x,y):
    a=np.divide(x,y)
    return a


# In[21]:


def fxnAxes(x):
    a=fig.add_axes([0.(x), 0.(x), 0.(x+1), 0.(x+1)])    
    return a


# In[22]:


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
baseG2 = [re.sub("C","2",str) for str in baseT1]
base_coded = [re.sub("G","3",str) for str in baseG2] 
A0=(a.count("A")) 
T1=(a.count("T"))
G2=(a.count("G"))
C3=(a.count("C"))
#print(base_coded)
names = ['A', 'T', 'G', 'C']
values = [(A0), (T1), (G2), (C3)]

plt.subplot()
plt.bar(names, values)
plt.suptitle('Nucleotides in all reads')
plt.show()


# In[23]:


"""
The purpose of this code is the following
1)Convert the coded reads to integer form 
2)Create an array inside the array with each read as a row
3)Create a 2D matric with the counts of all the possible scenarios 
presented in each row and the reads in each column
4)Create a 1D array with the readlengths
"""
kmer=[]
kmercount=[]
reads=[]
#readstring={}
readslength=[]
res=(len(base_coded))   
for x in range(res):
    l=list(int (i) for i in base_coded[x])
    for y in range(1):
        i=(l.count(y), l.count(y+1), l.count(y+2), l.count(y+3))
        kmercount.append(i)
        n=np.transpose(kmercount) # creates a 2D matrix with the bases counts as rows(ATGC) and the reads by columns
        kmer = n.view() #Create a view of the array with the same data
    d=np.asarray(l)
    d.resize((BaseFunction(d,4))) #Enter the list and the desired kmer for the function
    v=(str(d))
    reads.append(d) # store the reads in an array called reads
    c=len(d)
    readslength.append(c) #store the amt of kmer reads in readlength=


# In[24]:


"""
The purpose of this code is the following
1) visualize the readlengths
"""
plt.ioff()
for i in range(1):
    plt.title("Readlengths")
    plt.ylabel('number of bases')
    plt.xlabel('read id')
    plt.plot(kmer[i])
    plt.xticks(np.arange(0, 10))
    plt.show()


# In[25]:


"""
In this cell I was trying to create an array of all 256 possible combinations in base3  form 0000,0001, etc...
I am doing this so that I can use it as a template to count the scenarios from each read
"""
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)  


# In[26]:


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
    d=readslength[x] #obtains the kmers count per read
    #print("kmer",d)
    e=SigFunction(c,d) #figures out the average amount of resistances per kmer
    #print("Signal",x,"2D array size={}".format(e))
    b.resize(e) # Insert the (#bases,#sig) changes the signal level n signals wide
    #we can change this number later to be a sliding window for maching learning
    sig.append(b)
    #array_inspect(sig[x])
    #np.savetxt('sig_array.csv', b, delimiter=',')


# In[27]:


"""The purpose of this cell is to merge the reads and the raw signals together
I dont really remember why I was doing this, but I am now using it to export 
the csv array into another file where I will launch tensorflow


"""

concat=[]
for x in range(10):
    a=reads[x].ndim 
    b=sig[x].ndim
    reads[(x)].astype(int) #TESTTTTT
    #print(reads[x].dtype)
    c=np.concatenate(((reads[x]),sig[x]),axis=1)
    #array_inspect(c)
    concat.append(c)
    np.savetxt("concat.csv",c, delimiter=",")


# In[28]:


"""
Any cell under this cell is trash code or experimental
"""


# In[58]:





# In[59]:


#print(reads[0])
#print(sig[0])
for x in range(10):
    print("Training entries: {}, labels: {}".format(len(reads[x]), len(sig[x])))


# In[67]:


vocab_size = 256

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()


# In[ ]:


"""
I'm thinking the read identifier needs to be converted from a 2D array 
with a shape of (2078,4) to a 1D array with a shape of (2078,1) in which 
all 4 reads (ie:[0 1 2 3]) are concatenated to a single [0123] 4mer.
I am still trying to figure out how to do this, but I believe it involves 
converting the array into a string, and then concatenating the string,
then converting it back into an integer.
"""


# In[76]:



x_val =(sig[0]) #I'm trying to convert my reads and sig to tensor here and its not working
partial_x_train = (sig[0])
y_val =(reads[0])
partial_y_train =(reads[0])

print(array_inspect(partial_x_train))
print(array_inspect(partial_y_train))

print(partial_y_train)


# In[77]:


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:





# In[78]:


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[350]:


"""
IDEA: USE AMPLICON SEQUENCING DATA TO TRAIN THE MODEL BC IT HAS A KNOWN LENGTH IN BP 
OMG USE THE PECON AMPLICON DATA BC IT IS ALL EXACTLY 3KB IN SIZE!!!!!!!!!
"""


# In[ ]:





# In[ ]:




