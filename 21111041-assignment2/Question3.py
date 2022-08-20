#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle


# In[ ]:


file=open('hi.txt',encoding='UTF-8')


# In[5]:


vowels=["अ","आ","इ","ई","उ","ऊ","ए","ऐ","ओ","औ","अं","अः","ऋ","ॠ"]


cons=["क","ख","ग","घ","ङ","च","छ","ज","झ","ञ","ट","ठ","ड","ढ","ण","त","थ","द","ध",
          "न","प","फ","ब","भ","म","य","र","ल","व","श","ष","स","ह","ष","ञ","ऴ","ऱ","ऩ"]

matra=["ा","ि","ी","ु","ू","ृ","े","ै","ो","ौ","ं","ः","्"]

alpha=vowels + cons + matra


# In[1]:


#function to split word in characters using some conditions 

def spliting(ch):   
    addswar=[]
    for i in range(len(ch)-1):
        if ch[i]=="्":
            continue
        elif ch[i] in cons:
            if ch[i+1] in cons:
                addswar.append(ch[i] + "्")
                addswar.append("अ")
            elif ch[i+1] in vowels:
                addswar.append(ch[i] + "्")
                addswar.append("अ")
            else:
                addswar.append(ch[i] + "्")
        else :
            addswar.append(ch[i])
    i=len(ch)-1        
    if ch[i] in cons:
        addswar.append(ch[i] + "्")
        addswar.append("अ")
    elif ch[i]!="्":
        addswar.append(ch[i])
    return addswar
            


# In[ ]:


#appending all words to the list
alwrd=[]
for f in file:
    words=f.split(" ")
    for wrd in words:
        ch=''
        for hchr in wrd:
            if hchr in alpha:
                ch+=hchr
        if ch!="":
            alwrd.append(ch)
    


# In[14]:


#function to calculate ngram for character

def ngram(sngwd,n):
    bichr=''
    gram=[]
    temp=zip(*[sngwd[i:] for i in range(0,n)])
    for i in temp:
        n=len(i)
        for j in range(n):
            
#             if i[j][-1]=="्" and j==n-1:
#                 bichr+=i[j]
#             elif i[j][-1]=="्" and i[j+1]=='अ':
#                 bichr+=i[j][0]
#             elif j!=0 and i[j-1][-1]=="्" and i[j]=='अ':
#                 continue
# #             elif j!=n-1 and i[j][-1]=="्" and i[j+1][-1]=="्":
# #                 bichr+=i[j]
# #             elif j!=n-1 and i[j][-1]=="्" and i[j+1] in matra:
# #                 bichr+=i[j][0]
#             else:
            bichr+=i[j]#[0]
        gram.append(bichr)
        bichr=''
    return gram


# In[15]:


charunigram={}
charbigram={}
chartrigram={}
charquad={}
wordunigram={}
sylunigram={}
sylbigram={}
syltrigram={}
for st in alwrd:
    sngwd=spliting(st)                 #splitting words in characters
    for x in sngwd:
        if x in charunigram:           #creating charachter unigram frequency dictionary
            charunigram[x]+=1              
        else:
            charunigram[x]=1
       
    bigram=ngram(sngwd,2)
    for i in bigram:                   #calculating bigram using ngram function
        if i in charbigram:            #creating charachter bigram frequency dictionary
            charbigram[i]+=1
        else:
            charbigram[i]=1
            
    trigram=ngram(sngwd,3)            #calculating trigram using ngram function
    for i in trigram:                 #creating charachter trigram frequency dictionary
        if i in chartrigram:
            chartrigram[i]+=1
        else:
            chartrigram[i]=1
            
    qdgram=ngram(sngwd,4)             #calculating quadrigram using ngram function
    for i in qdgram:                  #creating charachter qudrigram frequency dictionary
        if i in charquad:
            charquad[i]+=1
        else:
            charquad[i]=1
    
    if st in wordunigram:            #creating word unigram frequency dictionary
        wordunigram[st]+=1
    else:
        wordunigram[st]=1
    
    
    lst=[]
    for i in range(len(st)):                   #iteraing on length og each word
        if i==0 and st[i] not in matra:        #cheaking some conditions to make bigram syllable
            em=st[i]
        elif st[i] not in matra and st[i-1]=="्":
            em+=st[i]
        elif st[i] not in matra:  
            lst.append(em)
            em=st[i]
        else:
            em+=st[i]
    lst.append(em)
    for z in lst:
        if z in sylunigram:          #creating syllables unigram frequency dictionary
            sylunigram[z]+=1
        else:
            sylunigram[z]=1
            
            
    for m in range(len(lst)-1):         #iterating on unigram syllables
        sylbi=''
        for n in range(2):              #taking 2 unigram at a time 
            sylbi+=lst[m+n]        
        if sylbi in sylbigram:          #creating syllables bigram frequency dictionary
            sylbigram[sylbi]+=1
        else:
            sylbigram[sylbi]=1
        
    for m in range(len(lst)-2):
        syltri=''
        for n in range(3):
            syltri+=lst[m+n]        
        if syltri in syltrigram:         #creating syllables trigram frequency dictionary
            syltrigram[syltri]+=1
        else:
            syltrigram[syltri]=1

        
            
        


# In[ ]:


wordbigram={}
for m in range(len(alwrd)-1):                   #iterating on all words
    wrdg=''
    for n in range(2):                          #taking  2 words in pairs
        if n==0:
            wrdg=wrdg+alwrd[m]
        else:
            wrdg+=" " + alwrd[m+n]
    if wrdg in wordbigram:                       #creating word unigram frequency dictionary
        wordbigram[wrdg]+=1
    else:
        wordbigram[wrdg]=1
    
wordtrigram={}
for m in range(len(alwrd)-1):                    #iterating on all words
    wrdg=''
    for n in range(3):                           #taking  2 words in pairs
        if n==0:
            wrdg=wrdg+alwrd[m]
        else:
            wrdg+=" " + alwrd[m+n]
    if wrdg in wordtrigram:                     #creating word bigram frequency dictionary
        wordtrigram[wrdg]+=1
    else:
        wordtrigram[wrdg]=1    


# ### making pickle files of all dictionaries

# In[ ]:


f1=open('pickle files/charunigram_pickle.pickle','wb')
pickle.dump(charunigram,f1)
f1.close()


# In[ ]:


f1=open('pickle files/charbigram_pickle.pickle','wb')
pickle.dump(charbigram,f1)
f1.close()


# In[ ]:


f1=open('pickle files/chartrigram_pickle.pickle','wb')
pickle.dump(chartrigram,f1)
f1.close()


# In[ ]:


f1=open('pickle files/charquad_pickle.pickle','wb')
pickle.dump(charquad,f1)
f1.close()


# In[ ]:


f1=open('pickle files/sylunigram_pickle.pickle','wb')
pickle.dump(sylunigram,f1)
f1.close()


# In[ ]:


f1=open('pickle files/sylbigram_pickle.pickle','wb')
pickle.dump(sylbigram,f1)
f1.close()


# In[ ]:


f1=open('pickle files/syltrigram_pickle.pickle','wb')
pickle.dump(syltrigram,f1)
f1.close()


# In[ ]:


f1=open('pickle files/unigram_words_pickle.pickle','wb')
pickle.dump(wordunigram,f1)
f1.close()


# In[ ]:


f1=open('pickle files/bigram_words_pickle.pickle','wb')
pickle.dump(wordbigram,f1)
f1.close()


# In[ ]:


f1=open('pickle files/trigram_words_pickle.pickle','wb')
pickle.dump(wordtrigram,f1)
f1.close()

