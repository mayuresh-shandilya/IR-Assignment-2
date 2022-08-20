#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from numpy.linalg import norm
import pandas as pd
from gensim.models import Word2Vec
from csv import writer


# In[9]:


#threshold function to calculate labels
def threshold(coslst):
    thrld=[]
    thrlst=[4,5,6,7,8]
    for i in coslst:
        eachthr=[]
        for j in thrlst:
            if i>=j:
                eachthr.append(1)
            else:
                eachthr.append(0)
        thrld.append(eachthr) 
    return thrld


# In[10]:


#threshold function to calculate lables for given ground truth
def thresgd(ground):
    thrld=[]
    thrlst=['4','5','6','7','8']
    for i in ground:
        eachthr=[]
        for j in thrlst:
            if i>=j:
                eachthr.append(1)
            else:
                eachthr.append(0)
        thrld.append(eachthr) 
    return thrld


# In[11]:


#accuracy function
def accuracy(x,y):
    acc=[0]*5
    acclst=[]
    lgth=len(x)
    for i in range(lgth):
        for j in range(len(x[i])):            
            if x[i][j]==int(y[i][j]):
                acc[j]+=1
    for i in range(5):
        res=(acc[i]/lgth)
        acclst.append(res)
    return acclst


# In[12]:


#reading word similarity file
wordsimi = open('Word similarity/Word similarity/hindi.txt', 'r', encoding='UTF-8')
smlr=[]       #to store both words 
grdth=[]      #to store ground turth
allwrd=[]     #to store all words
w1=[]         #to store first word
w2=[]         #to store second word
#preprocessing the file and appending all word in single list 
for i in wordsimi.readlines():
    a=i.split(",")
    if a[0]=="\n":
           continue    
    if a[2][-1]=="\n":
        a[2]=a[2][:-1]
    allwrd.append(a[0])
    w1.append(a[0])
    allwrd.append(a[1])
    w2.append(a[1])
    smlr.append(a[:-1])
    grdth.append(a[2:][0])


# In[13]:


gthresh=thresgd(grdth)    #calculating threshold on given ground truth


# ### Glove50

# In[14]:


glv_file50 = open('hi/hi/50/glove/hi-d50-glove.txt', 'rb')
glv50=glv_file50.read().decode(errors='replace')


# In[15]:


ditglv50={}                           #dictionary to store vector for all words in given wordsimilarity file
for i in glv50.split("\n"):
    v=i.split(" ")
    if v[0] in allwrd:
        ditglv50[v[0]]=v[1:]


# In[16]:


#calculating cosine similarity
glvcos50=[]                      
for i in smlr:
    v1=np.array(ditglv50[i[0]]).astype(float)
    v2=np.array(ditglv50[i[1]]).astype(float)
    cosine = np.dot(v1,v2)/(norm(v1)*norm(v2))
    glvcos50.append(cosine*10)
    


# In[17]:


#calcuting threshold on our consine values
mygd50=threshold(glvcos50)


# In[18]:


#acuuracy of the both model and ground truth
acc_50=accuracy(mygd50,gthresh)


# In[19]:


#making lables lists of each threshold value
gd4_50=[]
gd5_50=[]
gd6_50=[]
gd7_50=[]
gd8_50=[]
for i in mygd50:
    gd4_50.append(i[0])
    gd5_50.append(i[1])
    gd6_50.append(i[2])
    gd7_50.append(i[3])
    gd8_50.append(i[4])


# In[20]:


#writing values into csv for threshold 4
glvdict4_50={"Word1":w1,"Word2":w2,"Similarity Score":glvcos50,"Ground Truth similarity score":grdth,"Label":gd4_50}

dtframe4_50 = pd.DataFrame(glvdict4_50)

dtframe4_50.to_csv("Q1 output/Q1_Glove50_similarity_4.csv",index=False)

list_of_elem=["Acuuracy",acc_50[0],"",""]
with open("Q1 output/Q1_Glove50_similarity_4.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[21]:


#writing values into csv for threshold 5
glvdict5_50={"Word1":w1,"Word2":w2,"Similarity Score":glvcos50,"Ground Truth similarity score":grdth,"Label":gd5_50}

dtframe5_50 = pd.DataFrame(glvdict5_50)

dtframe5_50.to_csv("Q1 output/Q1_Glove50_similarity_5.csv",index=False)

list_of_elem=["Acuuracy",acc_50[1],"",""]
with open("Q1 output/Q1_Glove50_similarity_5.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[22]:


#writing values into csv for threshold 6
glvdict6_50={"Word1":w1,"Word2":w2,"Similarity Score":glvcos50,"Ground Truth similarity score":grdth,"Label":gd6_50}

dtframe6_50 = pd.DataFrame(glvdict6_50)

dtframe6_50.to_csv("Q1 output/Q1_Glove50_similarity_6.csv",index=False)

list_of_elem=["Acuuracy",acc_50[2],"",""]
with open("Q1 output/Q1_Glove50_similarity_6.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[23]:


#writing values into csv for threshold 7
glvdict7_50={"Word1":w1,"Word2":w2,"Similarity Score":glvcos50,"Ground Truth similarity score":grdth,"Label":gd7_50}

dtframe7_50 = pd.DataFrame(glvdict7_50)

dtframe7_50.to_csv("Q1 output/Q1_Glove50_similarity_7.csv",index=False)

list_of_elem=["Acuuracy",acc_50[3],"",""]
with open("Q1 output/Q1_Glove50_similarity_7.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[24]:


#writing values into csv for threshold 8
glvdict8_50={"Word1":w1,"Word2":w2,"Similarity Score":glvcos50,"Ground Truth similarity score":grdth,"Label":gd8_50}

dtframe8_50 = pd.DataFrame(glvdict8_50)

dtframe8_50.to_csv("Q1 output/Q1_Glove50_similarity_8.csv",index=False)

list_of_elem=["Acuuracy",acc_50[4],"",""]
with open("Q1 output/Q1_Glove50_similarity_8.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# ### Glove100

# In[25]:


glv_file100 = open('hi/hi/100/glove/hi-d100-glove.txt', 'rb')
glv100=glv_file100.read().decode(errors='replace')


# In[26]:


ditglv100={}                                #dictionary to store vector for all words in given wordsimilarity file
for i in glv100.split("\n"):
    v=i.split(" ")
    if v[0] in allwrd:
        ditglv100[v[0]]=v[1:]


# In[27]:


#calculating cosine similarity
glvcos100=[]
for i in smlr:
    v1=np.array(ditglv100[i[0]]).astype(float)
    v2=np.array(ditglv100[i[1]]).astype(float)
    cosine = np.dot(v1,v2)/(norm(v1)*norm(v2))
    glvcos100.append(cosine*10)


# In[28]:


#calcuting threshold on our consine values
mygd100=threshold(glvcos100)


# In[29]:


#acuuracy of the both model and ground truth
acc_100=accuracy(mygd100,gthresh)


# In[30]:


#making lables lists of each threshold value
gd4_100=[]
gd5_100=[]
gd6_100=[]
gd7_100=[]
gd8_100=[]
for i in mygd100:
    gd4_100.append(i[0])
    gd5_100.append(i[1])
    gd6_100.append(i[2])
    gd7_100.append(i[3])
    gd8_100.append(i[4])


# In[31]:


#writing values into csv for threshold 4
glvdict4_100={"Word1":w1,"Word2":w2,"Similarity Score":glvcos100,"Ground Truth similarity score":grdth,"Label":gd4_100}

dtframe4_100 = pd.DataFrame(glvdict4_100)

dtframe4_100.to_csv("Q1 output/Q1_Glove100_similarity_4.csv",index=False)

list_of_elem=["Acuuracy",acc_100[0],"",""]
with open("Q1 output/Q1_Glove100_similarity_4.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[32]:


#writing values into csv for threshold 5
glvdict5_100={"Word1":w1,"Word2":w2,"Similarity Score":glvcos100,"Ground Truth similarity score":grdth,"Label":gd5_100}

dtframe5_100 = pd.DataFrame(glvdict5_100)

dtframe5_100.to_csv("Q1 output/Q1_Glove100_similarity_5.csv",index=False)

list_of_elem=["Acuuracy",acc_100[1],"",""]
with open("Q1 output/Q1_Glove100_similarity_5.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[33]:


#writing values into csv for threshold 6
glvdict6_100={"Word1":w1,"Word2":w2,"Similarity Score":glvcos100,"Ground Truth similarity score":grdth,"Label":gd6_100}

dtframe6_100 = pd.DataFrame(glvdict6_100)

dtframe6_100.to_csv("Q1 output/Q1_Glove100_similarity_6.csv",index=False)

list_of_elem=["Acuuracy",acc_100[2],"",""]
with open("Q1 output/Q1_Glove100_similarity_6.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[34]:


#writing values into csv for threshold 7
glvdict7_100={"Word1":w1,"Word2":w2,"Similarity Score":glvcos100,"Ground Truth similarity score":grdth,"Label":gd7_100}

dtframe7_100 = pd.DataFrame(glvdict7_100)

dtframe7_100.to_csv("Q1 output/Q1_Glove100_similarity_7.csv",index=False)

list_of_elem=["Acuuracy",acc_100[3],"",""]
with open("Q1 output/Q1_Glove100_similarity_7.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[35]:


#writing values into csv for threshold 8
glvdict8_100={"Word1":w1,"Word2":w2,"Similarity Score":glvcos100,"Ground Truth similarity score":grdth,"Label":gd8_100}

dtframe8_100 = pd.DataFrame(glvdict8_100)

dtframe8_100.to_csv("Q1 output/Q1_Glove100_similarity_8.csv",index=False)

list_of_elem=["Acuuracy",acc_100[4],"",""]
with open("Q1 output/Q1_Glove100_similarity_8.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# ### fastext50

# In[36]:


fasttext50= Word2Vec.load("hi/hi/50/fasttext/hi-d50-m2-fasttext.model")


# In[37]:


ftcos50=[]                               #using model to calculate cosine similarity
for i in smlr:
    v1=fasttext50.wv[i[0]]
    v2=fasttext50.wv[i[1]]
    cosine = np.dot(v1,v2)/(norm(v1)*norm(v2))
    ftcos50.append(cosine*10)


# In[38]:


#calcuting threshold on our consine values
myft50=threshold(ftcos50)


# In[39]:


#acuuracy of the both model and ground truth
acc_50=accuracy(myft50,gthresh)


# In[40]:


#making lables lists of each threshold value
ft4_50=[]
ft5_50=[]
ft6_50=[]
ft7_50=[]
ft8_50=[]
for i in myft50:
    ft4_50.append(i[0])
    ft5_50.append(i[1])
    ft6_50.append(i[2])
    ft7_50.append(i[3])
    ft8_50.append(i[4])


# In[41]:


#writing values into csv for threshold 4
ftdict4_50={"Word1":w1,"Word2":w2,"Similarity Score":ftcos50,"Ground Truth similarity score":grdth,"Label":ft4_50}

ftframe4_50 = pd.DataFrame(ftdict4_50)

ftframe4_50.to_csv("Q1 output/Q1_fasttext50_similarity_4.csv",index=False)

list_of_elem=["Acuuracy",acc_50[0],"",""]
with open("Q1 output/Q1_fasttext50_similarity_4.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[42]:


#writing values into csv for threshold 5
ftdict5_50={"Word1":w1,"Word2":w2,"Similarity Score":ftcos50,"Ground Truth similarity score":grdth,"Label":ft5_50}

ftframe5_50 = pd.DataFrame(ftdict5_50)

ftframe5_50.to_csv("Q1 output/Q1_fasttext50_similarity_5.csv",index=False)

list_of_elem=["Acuuracy",acc_50[1],"",""]
with open("Q1 output/Q1_fasttext50_similarity_5.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[43]:


#writing values into csv for threshold 6
ftdict6_50={"Word1":w1,"Word2":w2,"Similarity Score":ftcos50,"Ground Truth similarity score":grdth,"Label":ft6_50}

ftframe6_50 = pd.DataFrame(ftdict6_50)

ftframe6_50.to_csv("Q1 output/Q1_fasttext50_similarity_6.csv",index=False)

list_of_elem=["Acuuracy",acc_50[2],"",""]
with open("Q1 output/Q1_fasttext50_similarity_6.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[44]:


#writing values into csv for threshold 7
ftdict7_50={"Word1":w1,"Word2":w2,"Similarity Score":ftcos50,"Ground Truth similarity score":grdth,"Label":ft7_50}

ftframe7_50 = pd.DataFrame(ftdict7_50)

ftframe7_50.to_csv("Q1 output/Q1_fasttext50_similarity_7.csv",index=False)

list_of_elem=["Acuuracy",acc_50[3],"",""]
with open("Q1 output/Q1_fasttext50_similarity_7.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[45]:


#writing values into csv for threshold 8
ftdict8_50={"Word1":w1,"Word2":w2,"Similarity Score":ftcos50,"Ground Truth similarity score":grdth,"Label":ft8_50}

ftframe8_50 = pd.DataFrame(ftdict8_50)

ftframe8_50.to_csv("Q1 output/Q1_fasttext50_similarity_8.csv",index=False)

list_of_elem=["Acuuracy",acc_50[4],"",""]
with open("Q1 output/Q1_fasttext50_similarity_8.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# ### fasttext100

# In[46]:


fasttext100= Word2Vec.load("hi/hi/100/fasttext/hi-d100-m2-fasttext.model")


# In[47]:


ftcos100=[]                                         #using model to calculate cosine similarity
for i in smlr:
    v1=fasttext100.wv[i[0]]
    v2=fasttext100.wv[i[1]]
    cosine = np.dot(v1,v2)/(norm(v1)*norm(v2))
    ftcos100.append(cosine*10)


# In[48]:


#calcuting threshold on our consine values
myft100=threshold(ftcos100)


# In[49]:


#acuuracy of the both model and ground truth
acc_100=accuracy(myft100,gthresh)


# In[50]:


#making lables lists of each threshold value
ft4_100=[]
ft5_100=[]
ft6_100=[]
ft7_100=[]
ft8_100=[]
for i in myft100:
    ft4_100.append(i[0])
    ft5_100.append(i[1])
    ft6_100.append(i[2])
    ft7_100.append(i[3])
    ft8_100.append(i[4])


# In[51]:


#writing values into csv for threshold 4
ftdict4_100={"Word1":w1,"Word2":w2,"Similarity Score":ftcos100,"Ground Truth similarity score":grdth,"Label":ft4_100}

ftframe4_100 = pd.DataFrame(ftdict4_100)

ftframe4_100.to_csv("Q1 output/Q1_fasttext100_similarity_4.csv",index=False)

list_of_elem=["Acuuracy",acc_100[0],"",""]
with open("Q1 output/Q1_fasttext100_similarity_4.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[52]:


#writing values into csv for threshold 5
ftdict5_100={"Word1":w1,"Word2":w2,"Similarity Score":ftcos100,"Ground Truth similarity score":grdth,"Label":ft5_100}

ftframe5_100 = pd.DataFrame(ftdict5_100)

ftframe5_100.to_csv("Q1 output/Q1_fasttext100_similarity_5.csv",index=False)

list_of_elem=["Acuuracy",acc_100[1],"",""]
with open("Q1 output/Q1_fasttext100_similarity_5.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[53]:


#writing values into csv for threshold 6
ftdict6_100={"Word1":w1,"Word2":w2,"Similarity Score":ftcos100,"Ground Truth similarity score":grdth,"Label":ft6_100}

ftframe6_100 = pd.DataFrame(ftdict6_100)

ftframe6_100.to_csv("Q1 output/Q1_fasttext100_similarity_6.csv",index=False)

list_of_elem=["Acuuracy",acc_100[2],"",""]
with open("Q1 output/Q1_fasttext100_similarity_6.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[54]:


#writing values into csv for threshold 7
ftdict7_100={"Word1":w1,"Word2":w2,"Similarity Score":ftcos100,"Ground Truth similarity score":grdth,"Label":ft7_100}

ftframe7_100 = pd.DataFrame(ftdict7_100)

ftframe7_100.to_csv("Q1 output/Q1_fasttext100_similarity_7.csv",index=False)

list_of_elem=["Acuuracy",acc_100[3],"",""]
with open("Q1 output/Q1_fasttext100_similarity_7.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[55]:


#writing values into csv for threshold 8
ftdict8_100={"Word1":w1,"Word2":w2,"Similarity Score":ftcos100,"Ground Truth similarity score":grdth,"Label":ft8_100}

ftframe8_100 = pd.DataFrame(ftdict8_100)

ftframe8_100.to_csv("Q1 output/Q1_fasttext100_similarity_8.csv",index=False)

list_of_elem=["Acuuracy",acc_100[4],"",""]
with open("Q1 output/Q1_fasttext100_similarity_8.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# ### cbow50

# In[56]:


cbow50= Word2Vec.load("hi/hi/50/cbow/hi-d50-m2-cbow.model")


# In[57]:


cbowcos50=[]                                       #using model to calculate cosine similarity
for i in smlr:
    v1=cbow50.wv[i[0]]
    v2=cbow50.wv[i[1]]
    cosine = np.dot(v1,v2)/(norm(v1)*norm(v2))
    cbowcos50.append(cosine*10)


# In[58]:


#calcuting threshold on our consine values
mycb50=threshold(cbowcos50)


# In[59]:


#acuuracy of the both model and ground truth
acc_50=accuracy(mycb50,gthresh)


# In[60]:


#making lables lists of each threshold value
cb4_50=[]
cb5_50=[]
cb6_50=[]
cb7_50=[]
cb8_50=[]
for i in mycb50:
    cb4_50.append(i[0])
    cb5_50.append(i[1])
    cb6_50.append(i[2])
    cb7_50.append(i[3])
    cb8_50.append(i[4])


# In[61]:


#writing values into csv for threshold 4
cbdict4_50={"Word1":w1,"Word2":w2,"Similarity Score":cbowcos50,"Ground Truth similarity score":grdth,"Label":cb4_50}

cbframe4_50 = pd.DataFrame(cbdict4_50)

cbframe4_50.to_csv("Q1 output/Q1_cbow50_similarity_4.csv",index=False)

list_of_elem=["Acuuracy",acc_50[0],"",""]
with open("Q1 output/Q1_cbow50_similarity_4.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[62]:


#writing values into csv for threshold 5
cbdict5_50={"Word1":w1,"Word2":w2,"Similarity Score":cbowcos50,"Ground Truth similarity score":grdth,"Label":cb5_50}

cbframe5_50 = pd.DataFrame(cbdict5_50)

cbframe5_50.to_csv("Q1 output/Q1_cbow50_similarity_5.csv",index=False)

list_of_elem=["Acuuracy",acc_50[1],"",""]
with open("Q1 output/Q1_cbow50_similarity_5.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[63]:


#writing values into csv for threshold 6
cbdict6_50={"Word1":w1,"Word2":w2,"Similarity Score":cbowcos50,"Ground Truth similarity score":grdth,"Label":cb6_50}

cbframe6_50 = pd.DataFrame(cbdict6_50)

cbframe6_50.to_csv("Q1 output/Q1_cbow50_similarity_6.csv",index=False)

list_of_elem=["Acuuracy",acc_50[2],"",""]
with open("Q1 output/Q1_cbow50_similarity_6.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[64]:


#writing values into csv for threshold 7
cbdict7_50={"Word1":w1,"Word2":w2,"Similarity Score":cbowcos50,"Ground Truth similarity score":grdth,"Label":cb7_50}

cbframe7_50 = pd.DataFrame(cbdict7_50)

cbframe7_50.to_csv("Q1 output/Q1_cbow50_similarity_7.csv",index=False)

list_of_elem=["Acuuracy",acc_50[3],"",""]
with open("Q1 output/Q1_cbow50_similarity_7.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[65]:


#writing values into csv for threshold 8
cbdict8_50={"Word1":w1,"Word2":w2,"Similarity Score":cbowcos50,"Ground Truth similarity score":grdth,"Label":cb8_50}

cbframe8_50 = pd.DataFrame(cbdict8_50)

cbframe8_50.to_csv("Q1 output/Q1_cbow50_similarity_8.csv",index=False)

list_of_elem=["Acuuracy",acc_50[4],"",""]
with open("Q1 output/Q1_cbow50_similarity_8.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# ### cbow100

# In[66]:


cbow100= Word2Vec.load("hi/hi/100/cbow/hi-d100-m2-cbow.model")


# In[67]:


cbowcos100=[]                                           #using model to calculate cosine similarity
for i in smlr:
    v1=cbow100.wv[i[0]]
    v2=cbow100.wv[i[1]]
    cosine = np.dot(v1,v2)/(norm(v1)*norm(v2))
    cbowcos100.append(cosine*10)


# In[68]:


#calcuting threshold on our consine values
mycb100=threshold(cbowcos100)


# In[69]:


#acuuracy of the both model and ground truth
acc_100=accuracy(mycb100,gthresh)


# In[70]:


#making lables lists of each threshold value
cb4_100=[]
cb5_100=[]
cb6_100=[]
cb7_100=[]
cb8_100=[]
for i in mycb100:
    cb4_100.append(i[0])
    cb5_100.append(i[1])
    cb6_100.append(i[2])
    cb7_100.append(i[3])
    cb8_100.append(i[4])


# In[71]:


#writing values into csv for threshold 4
cbdict4_100={"Word1":w1,"Word2":w2,"Similarity Score":cbowcos100,"Ground Truth similarity score":grdth,"Label":cb4_100}

cbframe4_100 = pd.DataFrame(cbdict4_100)

cbframe4_100.to_csv("Q1 output/Q1_cbow100_similarity_4.csv",index=False)

list_of_elem=["Acuuracy",acc_100[0],"",""]
with open("Q1 output/Q1_cbow100_similarity_4.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[72]:


#writing values into csv for threshold 5
cbdict5_100={"Word1":w1,"Word2":w2,"Similarity Score":cbowcos100,"Ground Truth similarity score":grdth,"Label":cb5_100}

cbframe5_100 = pd.DataFrame(cbdict5_100)

cbframe5_100.to_csv("Q1 output/Q1_cbow100_similarity_5.csv",index=False)

list_of_elem=["Acuuracy",acc_100[1],"",""]
with open("Q1 output/Q1_cbow100_similarity_5.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[73]:


#writing values into csv for threshold 6
cbdict6_100={"Word1":w1,"Word2":w2,"Similarity Score":cbowcos100,"Ground Truth similarity score":grdth,"Label":cb6_100}

cbframe6_100 = pd.DataFrame(cbdict6_100)

cbframe6_100.to_csv("Q1 output/Q1_cbow100_similarity_6.csv",index=False)

list_of_elem=["Acuuracy",acc_100[2],"",""]
with open("Q1 output/Q1_cbow100_similarity_6.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[74]:


#writing values into csv for threshold 7
cbdict7_100={"Word1":w1,"Word2":w2,"Similarity Score":cbowcos100,"Ground Truth similarity score":grdth,"Label":cb7_100}

cbframe7_100 = pd.DataFrame(cbdict7_100)

cbframe7_100.to_csv("Q1 output/Q1_cbow100_similarity_7.csv",index=False)

list_of_elem=["Acuuracy",acc_100[3],"",""]
with open("Q1 output/Q1_cbow100_similarity_7.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[75]:


#writing values into csv for threshold 8
cbdict8_100={"Word1":w1,"Word2":w2,"Similarity Score":cbowcos100,"Ground Truth similarity score":grdth,"Label":cb8_100}

cbframe8_100 = pd.DataFrame(cbdict8_100)

cbframe8_100.to_csv("Q1 output/Q1_cbow100_similarity_8.csv",index=False)

list_of_elem=["Acuuracy",acc_100[4],"",""]
with open("Q1 output/Q1_cbow100_similarity_8.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# ### skipgram50

# In[76]:


sg50= Word2Vec.load("hi/hi/50/sg/hi-d50-m2-sg.model")


# In[77]:


sgcos50=[]                                            #using model to calculate cosine similarity
for i in smlr:
    v1=sg50.wv[i[0]]
    v2=sg50.wv[i[1]]
    cosine = np.dot(v1,v2)/(norm(v1)*norm(v2))
    sgcos50.append(cosine*10)


# In[78]:


#calcuting threshold on our consine values
mysg50=threshold(sgcos50)


# In[79]:


#acuuracy of the both model and ground truth
acc_50=accuracy(mysg50,gthresh)


# In[80]:


#making lables lists of each threshold value
sg4_50=[]
sg5_50=[]
sg6_50=[]
sg7_50=[]
sg8_50=[]
for i in mysg50:
    sg4_50.append(i[0])
    sg5_50.append(i[1])
    sg6_50.append(i[2])
    sg7_50.append(i[3])
    sg8_50.append(i[4])


# In[81]:


#writing values into csv for threshold 4
sgdict4_50={"Word1":w1,"Word2":w2,"Similarity Score":sgcos50,"Ground Truth similarity score":grdth,"Label":sg4_50}

sgframe4_50 = pd.DataFrame(sgdict4_50)

sgframe4_50.to_csv("Q1 output/Q1_skipgram50_similarity_4.csv",index=False)

list_of_elem=["Acuuracy",acc_50[0],"",""]
with open("Q1 output/Q1_skipgram50_similarity_4.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[82]:


#writing values into csv for threshold 5
sgdict5_50={"Word1":w1,"Word2":w2,"Similarity Score":sgcos50,"Ground Truth similarity score":grdth,"Label":sg5_50}

sgframe5_50 = pd.DataFrame(sgdict5_50)

sgframe5_50.to_csv("Q1 output/Q1_skipgram50_similarity_5.csv",index=False)

list_of_elem=["Acuuracy",acc_50[1],"",""]
with open("Q1 output/Q1_skipgram50_similarity_5.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[83]:


#writing values into csv for threshold 6
sgdict6_50={"Word1":w1,"Word2":w2,"Similarity Score":sgcos50,"Ground Truth similarity score":grdth,"Label":sg6_50}

sgframe6_50 = pd.DataFrame(sgdict6_50)

sgframe6_50.to_csv("Q1 output/Q1_skipgram50_similarity_6.csv",index=False)

list_of_elem=["Acuuracy",acc_50[2],"",""]
with open("Q1 output/Q1_skipgram50_similarity_6.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[84]:


#writing values into csv for threshold 7
sgdict7_50={"Word1":w1,"Word2":w2,"Similarity Score":sgcos50,"Ground Truth similarity score":grdth,"Label":sg7_50}

sgframe7_50 = pd.DataFrame(sgdict7_50)

sgframe7_50.to_csv("Q1 output/Q1_skipgram50_similarity_7.csv",index=False)

list_of_elem=["Acuuracy",acc_50[3],"",""]
with open("Q1 output/Q1_skipgram50_similarity_7.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[85]:


#writing values into csv for threshold 8
sgdict8_50={"Word1":w1,"Word2":w2,"Similarity Score":sgcos50,"Ground Truth similarity score":grdth,"Label":sg8_50}

sgframe8_50 = pd.DataFrame(sgdict8_50)

sgframe8_50.to_csv("Q1 output/Q1_skipgram50_similarity_8.csv",index=False)

list_of_elem=["Acuuracy",acc_50[4],"",""]
with open("Q1 output/Q1_skipgram50_similarity_8.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# ### skipgram100

# In[86]:


sg100= Word2Vec.load("hi/hi/100/sg/hi-d100-m2-sg.model")


# In[87]:


sgcos100=[]                                                 #using model to calculate cosine similarity
for i in smlr:
    v1=sg100.wv[i[0]]
    v2=sg100.wv[i[1]]
    cosine = np.dot(v1,v2)/(norm(v1)*norm(v2))
    sgcos100.append(cosine*10)


# In[88]:


#calcuting threshold on our consine values
mysg100=threshold(sgcos100)


# In[89]:


#acuuracy of the both model and ground truth
acc_100=accuracy(mysg100,gthresh)


# In[90]:


#making lables lists of each threshold value
sg4_100=[]
sg5_100=[]
sg6_100=[]
sg7_100=[]
sg8_100=[]
for i in mysg100:
    sg4_100.append(i[0])
    sg5_100.append(i[1])
    sg6_100.append(i[2])
    sg7_100.append(i[3])
    sg8_100.append(i[4])


# In[91]:


#writing values into csv for threshold 4
sgdict4_100={"Word1":w1,"Word2":w2,"Similarity Score":sgcos100,"Ground Truth similarity score":grdth,"Label":sg4_100}

sgframe4_100 = pd.DataFrame(sgdict4_100)

sgframe4_100.to_csv("Q1 output/Q1_skipgram100_similarity_4.csv",index=False)

list_of_elem=["Acuuracy",acc_100[0],"",""]
with open("Q1 output/Q1_skipgram100_similarity_4.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[92]:


#writing values into csv for threshold 5
sgdict5_100={"Word1":w1,"Word2":w2,"Similarity Score":sgcos100,"Ground Truth similarity score":grdth,"Label":sg5_100}

sgframe5_100 = pd.DataFrame(sgdict5_100)

sgframe5_100.to_csv("Q1 output/Q1_skipgram100_similarity_5.csv",index=False)

list_of_elem=["Acuuracy",acc_100[1],"",""]
with open("Q1 output/Q1_skipgram100_similarity_5.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[93]:


#writing values into csv for threshold 6
sgdict6_100={"Word1":w1,"Word2":w2,"Similarity Score":sgcos100,"Ground Truth similarity score":grdth,"Label":sg6_100}

sgframe6_100 = pd.DataFrame(sgdict6_100)

sgframe6_100.to_csv("Q1 output/Q1_skipgram100_similarity_6.csv",index=False)

list_of_elem=["Acuuracy",acc_100[2],"",""]
with open("Q1 output/Q1_skipgram100_similarity_6.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[94]:


#writing values into csv for threshold 7
sgdict7_100={"Word1":w1,"Word2":w2,"Similarity Score":sgcos100,"Ground Truth similarity score":grdth,"Label":sg7_100}

sgframe7_100 = pd.DataFrame(sgdict7_100)

sgframe7_100.to_csv("Q1 output/Q1_skipgram100_similarity_7.csv",index=False)

list_of_elem=["Acuuracy",acc_100[3],"",""]
with open("Q1 output/Q1_skipgram100_similarity_7.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[95]:


#writing values into csv for threshold 8
sgdict8_100={"Word1":w1,"Word2":w2,"Similarity Score":sgcos100,"Ground Truth similarity score":grdth,"Label":sg8_100}

sgframe8_100 = pd.DataFrame(sgdict8_100)

sgframe8_100.to_csv("Q1 output/Q1_skipgram100_similarity_8.csv",index=False)

list_of_elem=["Acuuracy",acc_100[4],"",""]
with open("Q1 output/Q1_skipgram100_similarity_8.csv", 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(list_of_elem)


# In[ ]:





# In[ ]:




