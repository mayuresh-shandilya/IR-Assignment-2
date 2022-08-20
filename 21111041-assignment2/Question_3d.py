#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pickle
import matplotlib.pyplot as plt


# In[25]:


with open('pickle files/charunigram_pickle.pickle','rb') as file:
    frequent_charunigram=pickle.load(file)
    file.close()

frequent_charunigram=dict(sorted(frequent_charunigram.items(), key=lambda x: x[1], reverse=True)[:100])

fbg=list(frequent_charunigram.keys())
with open('Q3 output/character_unigram.txt', 'w',encoding='UTF-8') as file_list:
    for ele in fbg:
        file_list.write(ele+'\n')

x=[]
y=[]
k=0
for i in range(len(frequent_charunigram)):
    x.append(i)
for i in frequent_charunigram:
    y.append(frequent_charunigram[i])
    
fig=plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)
plt.plot(x,y,label="character_unigram")
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('character_unigram')
plt.legend()
plt.savefig('Q3d output/character_unigram.jpg')
plt.show()


# In[26]:


with open('pickle files/charbigram_pickle.pickle','rb') as file:
    frequent_charbigram=pickle.load(file)
    file.close()

frequent_charbigram=dict(sorted(frequent_charbigram.items(), key=lambda x: x[1], reverse=True)[:100])

fbg=list(frequent_charbigram.keys())
with open('Q3 output/character_bigram.txt', 'w',encoding='UTF-8') as file_list:
    for ele in fbg:
        file_list.write(ele+'\n')

x=[]
y=[]
k=0
for i in range(len(frequent_charbigram)):
    x.append(i)
for i in frequent_charbigram:
    y.append(frequent_charbigram[i])
    
fig=plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)
plt.plot(x,y,label="character_bigram")
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('character_bigram')
plt.legend()
plt.savefig('Q3d output/character_bigram.jpg')
plt.show()


# In[27]:


with open('pickle files/chartrigram_pickle.pickle','rb') as file:
    frequent_chartrigram=pickle.load(file)
    file.close()

frequent_chartrigram=dict(sorted(frequent_chartrigram.items(), key=lambda x: x[1], reverse=True)[:100])

fbg=list(frequent_chartrigram.keys())
with open('Q3 output/character_trigram.txt', 'w',encoding='UTF-8') as file_list:
    for ele in fbg:
        file_list.write(ele+'\n')

x=[]
y=[]
k=0
for i in range(len(frequent_chartrigram)):
    x.append(i)
for i in frequent_chartrigram:
    y.append(frequent_chartrigram[i])
    
fig=plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)
plt.plot(x,y,label="character_trigram")
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('character_trigram')
plt.legend()
plt.savefig('Q3d output/character_trigram.jpg')
plt.show()


# In[28]:


with open('pickle files/charquad_pickle.pickle','rb') as file:
    frequent_charquad=pickle.load(file)
    file.close()

frequent_charquad=dict(sorted(frequent_charquad.items(), key=lambda x: x[1], reverse=True)[:100])

fbg=list(frequent_charquad.keys())
with open('Q3 output/character_quadgram.txt', 'w',encoding='UTF-8') as file_list:
    for ele in fbg:
        file_list.write(ele+'\n')

x=[]
y=[]
k=0
for i in range(len(frequent_charquad)):
    x.append(i)
for i in frequent_charquad:
    y.append(frequent_charquad[i])
    
fig=plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)
plt.plot(x,y,label="character_quadgram")
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('character_quadgram')
plt.legend()
plt.savefig('Q3d output/character_quadgram.jpg')
plt.show()


# In[29]:


with open('pickle files/sylunigram_pickle.pickle','rb') as file:
    frequent_sylunigram=pickle.load(file)
    file.close()

frequent_sylunigram=dict(sorted(frequent_sylunigram.items(), key=lambda x: x[1], reverse=True)[:100])

fbg=list(frequent_sylunigram.keys())
with open('Q3 output/syllable_unigram.txt', 'w',encoding='UTF-8') as file_list:
    for ele in fbg:
        file_list.write(ele+'\n')

x=[]
y=[]
k=0
for i in range(len(frequent_sylunigram)):
    x.append(i)
for i in frequent_sylunigram:
    y.append(frequent_sylunigram[i])
    
fig=plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)
plt.plot(x,y,label="syllable_unigram")
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('syllable_unigram')
plt.legend()
plt.savefig('Q3d output/syllable_unigram.jpg')
plt.show()


# In[30]:


with open('pickle files/sylbigram_pickle.pickle','rb') as file:
    frequent_sylbigram=pickle.load(file)
    file.close()

frequent_sylbigram=dict(sorted(frequent_sylbigram.items(), key=lambda x: x[1], reverse=True)[:100])

fbg=list(frequent_sylbigram.keys())
with open('Q3 output/syllable_bigram.txt', 'w',encoding='UTF-8') as file_list:
    for ele in fbg:
        file_list.write(ele+'\n')

x=[]
y=[]
k=0
for i in range(len(frequent_sylbigram)):
    x.append(i)
for i in frequent_sylbigram:
    y.append(frequent_sylbigram[i])
    
fig=plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)
plt.plot(x,y,label="syllable_bigram")
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('syllable_bigram')
plt.legend()
plt.savefig('Q3d output/syllable_bigram.jpg')
plt.show()


# In[31]:


with open('pickle files/syltrigram_pickle.pickle','rb') as file:
    frequent_syltrigram=pickle.load(file)
    file.close()

frequent_syltrigram=dict(sorted(frequent_syltrigram.items(), key=lambda x: x[1], reverse=True)[:100])

fbg=list(frequent_syltrigram.keys())
with open('Q3 output/syllable_trigram.txt', 'w',encoding='UTF-8') as file_list:
    for ele in fbg:
        file_list.write(ele+'\n')

x=[]
y=[]
k=0
for i in range(len(frequent_syltrigram)):
    x.append(i)
for i in frequent_syltrigram:
    y.append(frequent_syltrigram[i])
    
fig=plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)
plt.plot(x,y,label="syllable_trigram")
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('syllable_trigram')
plt.legend()
plt.savefig('Q3d output/syllable_trigram.jpg')
plt.show()


# In[32]:


with open('pickle files/unigram_words_pickle.pickle','rb') as file:
    frequent_unigram_words=pickle.load(file)
    file.close()

frequent_unigram_words=dict(sorted(frequent_unigram_words.items(), key=lambda x: x[1], reverse=True)[:100])

fbg=list(frequent_unigram_words.keys())
with open('Q3 output/unigram_word.txt', 'w',encoding='UTF-8') as file_list:
    for ele in fbg:
        file_list.write(ele+'\n')

x=[]
y=[]
k=0
for i in range(len(frequent_unigram_words)):
    x.append(i)
for i in frequent_unigram_words:
    y.append(frequent_unigram_words[i])
    
fig=plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)
plt.plot(x,y,label="unigram_word")
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('unigram_word')
plt.legend()
plt.savefig('Q3d output/unigram_word.jpg')
plt.show()


# In[33]:


with open('pickle files/bigram_words_pickle.pickle','rb') as file:
    frequent_bigram_words=pickle.load(file)
    file.close()

frequent_bigram_words=dict(sorted(frequent_bigram_words.items(), key=lambda x: x[1], reverse=True)[:100])

frequent_bigram_words=dict(sorted(frequent_bigram_words.items(), key=lambda x: x[1], reverse=True)[:100])
fbg=list(frequent_bigram_words.keys())
with open('Q3 output/bigram_words.txt', 'w',encoding='UTF-8') as file_list:
    for ele in fbg:
        file_list.write(ele+'\n')

x=[]
y=[]
k=0
for i in range(len(frequent_bigram_words)):
    x.append(i)
for i in frequent_bigram_words:
    y.append(frequent_bigram_words[i])
    
fig=plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)
plt.plot(x,y,label="bigram_word")
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('bigram_word')
plt.legend()
plt.savefig('Q3d output/bigram_word.jpg')
plt.show()


# In[34]:


with open('pickle files/trigram_words_pickle.pickle','rb') as file:
    frequent_trigram_words=pickle.load(file)
    file.close()

frequent_trigram_words=dict(sorted(frequent_trigram_words.items(), key=lambda x: x[1], reverse=True)[:100])

fbg=list(frequent_trigram_words.keys())
with open('Q3 output/trigram_word.txt', 'w',encoding='UTF-8') as file_list:
    for ele in fbg:
        file_list.write(ele+'\n')

x=[]
y=[]
k=0
for i in range(len(frequent_trigram_words)):
    x.append(i)
for i in frequent_trigram_words:
    y.append(frequent_trigram_words[i])
    
fig=plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)
plt.plot(x,y,label="trigram_word")
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('trigram_word')
plt.legend()
plt.savefig('Q3d output/trigram_word.jpg')
plt.show()


# ### From above all graphs we can se that "frequency is inversely proportional to the rank" hence it follows zipfian distribution

# In[ ]:




