#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd


# In[14]:


ner=open("hindi_ner/hi_train.conll",encoding='utf-8')
file=ner.read()


# In[15]:


lines=file.split("# id")


# In[16]:


sent=[]
tags=[]
for i in range(len(lines)):
    if i==0:
        continue
    p=lines[i].split('\n')[1:]
    st=""
    tg=""
    for j in range(len(p)):
        if p[j]=="" :
            continue
        pair=p[j].split(" ")
        if j==0:
            st+=pair[0]
            tg+=pair[-1]
        else:
            st=st + " " +pair[0]
            tg=tg + " " +pair[-1]
    sent.append(st)
    tags.append(tg)


# In[17]:


dt={"sent":sent,"tags":tags}
df=pd.DataFrame(dt)


# In[18]:


df.to_csv("NER_Train.csv",index=False)


# In[19]:


ner=open("hindi_ner/hi_dev.conll",encoding='utf-8')
file=ner.read()


# In[20]:


lines=file.split("# id")


# In[21]:


sent=[]
tags=[]
for i in range(len(lines)):
    if i==0:
        continue
    p=lines[i].split('\n')[1:]
    st=""
    tg=""
    for j in range(len(p)):
        if p[j]=="" :
            continue
        pair=p[j].split(" ")
        if j==0:
            st+=pair[0]
            tg+=pair[-1]
        else:
            st=st + " " +pair[0]
            tg=tg + " " +pair[-1]
    sent.append(st)
    tags.append(tg)


# In[22]:


dt={"sent":sent,"tags":tags}
df=pd.DataFrame(dt)


# In[23]:


df.to_csv("NER_Test.csv",index=False)


# In[24]:


df


# In[ ]:




