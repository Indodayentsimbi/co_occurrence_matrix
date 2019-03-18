from nltk.tokenize import word_tokenize
from Co_occurence_matrix import co_occurrence
import re
import numpy as np
import pandas as pd

Corpus = 'He is not lazy. He is intelligent. He is smart' #original string
Corpus = re.sub(r'[^\w\s]','',Corpus) #regular expression to remove punctuation

#vocab
vocab = word_tokenize(text=Corpus,language='english')
size = len(vocab)
#initialise empty matrix
matrix = np.zeros(shape=(size,size))

#update matrix
for row in vocab:
	for column in vocab:
		matrix[vocab.index(row),vocab.index(column)] = co_occurrence(corpus=Corpus,
																		word1=row,
																		word2=column,
																		contextwindow=2,
																		returncombinations=False)

print(pd.DataFrame(data=matrix,columns=vocab,index=vocab))
