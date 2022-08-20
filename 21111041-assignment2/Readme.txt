Plugins-
Environments: python 3.7.0


Dependencies-
Python Libraries: pandas 1.0.3, numpy 1.18.1, pickle, matplotlib, gensim

How to run : to run whole assignment please type below command on the terminal
		$ make run 

Executable Files-

1) Question 1: This .py file contains all the models implemented for each threshold
 
2) Question 2: This .py file contains the preprossesing of the dataset

3) Question 3: This .py file contains the code to which generates the dictionary and it is stored in the form of pickle file.The pickle file contains all the ngram of the character,syllable and word.There are 10 such pickle file.I made pickle file to reduce the memory load on the computer as the dataset is very big.

4) Question 3d: This .py file imports all the pickle files and computes top 100 ngram and store it in .txt file. Also this file test if frequency of character,syllable, word follows zipfian distribution.



Output files:


1)Q1 Output: This file contains the all 40 csv files generated in question 1

2)Q3 Output: This files conatains the top 100 ngram .txt files of character,syllable, word

3)Q3d Output: This file contains the graph plotted between frequency and rank of the character,syllable or word


Input Files:

1)for Question 1 : I have inserted word similarity data in the zip file but please include models data in the Zip. Also set path as hi/hi/model
2)for Question 3 : Beacuse the input dataset was too much big i have not inserted it in zip file please insert it in the zip file and name it as hi.txt
2)for Question 3d: The input files for this are stored in the "pickle files" folder .


Note: In the Question 3 as the data was so big I have splitted the hi.txt in several files and then ran that files to save it from memory errors.
	the command used to split it was $ split -l 650000 hi.txt
		   








