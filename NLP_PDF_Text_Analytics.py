
# coding: utf-8
'''
Author : Jasmi Patel
Create Date : 2/20/19
Version : 1 (Initial Version)
'''
# # NLP Text Analytics on PDF Documents

# Data Source - https://www.philadelphiafed.org/research-and-data/real-time-center/greenbook-data/pdf-data-set

# Import Python basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Libraries for reading PDFs
import PyPDF2 
import os, sys

# Import NLTK Text Processing Libraries
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

directory = 'C:\\Users\\l1jxp04\\AI Notebooks\\NLP\\NLP_Demo\\GreenSheets-2012\\GreenSheets-2012'

# For loop to read all PDF files available in directory 
allwords = []
allcount = 0
alltext = ""
i = 0
for file in os.listdir(directory):
    i += 1
    if not file.endswith(".pdf"):
        continue
    pdf_filename =  os.path.join(directory,file)  
       
    #open file
    pdfFileObj = open(pdf_filename,'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    num_pages = pdfReader.numPages
        
    #Read content of PDF as Text
    count = 0
    text = ""
    while count < num_pages:
        pageObj = pdfReader.getPage(count)
        count +=1
        text += pageObj.extractText()
    if text != "":
       text = text
    alltext = alltext + text
    # Extracting word Tokens from Text
    allcount = allcount + count
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
   #Lemmatization 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    #Remove Stopwords and Punctuations
    punctuation = ['(',')',';',':','[',']',',']
    stop_words = stopwords.words('english')
    
    keywords = [word for word in tokens if not word in stop_words and  not word in punctuation and word.isalpha()]
    #print(keywords)
    allwords += keywords 
    

df = pd.DataFrame(allwords)

wordstr = ' '.join(str(e) for e in allwords)


# ## 1. WordCloud

# Word Cloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance.

from wordcloud import WordCloud, STOPWORDS 

wordcloud = WordCloud(max_font_size=60).generate(alltext)

plt.figure(figsize=(16,12))
# plot wordcloud in matplotlib
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud on Keywords from All PDFs")
plt.show()

## Term Frequency - Inverse Term Frequency

# Term Frequency(TF), you just count the number of words occurred in each document.
# IDF(Inverse Document Frequency) measures the amount of information a given word provides across the document.
# 

from sklearn.feature_extraction.text import TfidfVectorizer

# Create the tf-idf feature matrix
tfidf = TfidfVectorizer()
text_data = np.array(allwords)

feature_matrix = tfidf.fit_transform(text_data)
feature_matrix.toarray()

# Show tf-idf feature matrix
tfidf.get_feature_names()

# Create data frame
#pd.DataFrame(feature_matrix.toarray(), columns=tfidf.get_feature_names())


# ## 2. Lexical dispersion plot

# This is the plot of a word vs the offset of the word in the text corpus.The y-axis represents the word. Each word has a strip representing entire text in terms of offset, and a mark on the strip indicates the occurrence of the word at that offset, a strip is an x-axis. The positional information can indicate the focus of discussion in the text. 

topics = ['projection', 'federal', 'percent','tealbook','economic']

from nltk.draw.dispersion import dispersion_plot
dispersion_plot(allwords, topics)


# ## 3. Frequency distribution plot

import nltk
from nltk.probability import FreqDist

fqdist = FreqDist(allwords)

freqdist = nltk.FreqDist(allwords)
plt.figure(figsize=(16,5))
freqdist.plot(50)

# Most Frequent 10 words in all Text

freqdist.most_common(10)

## Draw a bar chart with the count of the most common 20 words
x, y = zip(*freqdist.most_common(n=20))
plt.figure(figsize=(16,5))
plt.bar(range(len(x)), y, color = 'Orange', tick_label = y)
plt.xticks(range(len(x)), x)
plt.title('Frequency Count of Top 20 Words')
plt.xlabel('Frequent Words')
plt.ylabel('Count')
plt.show()
plt.savefig('Most Frequent 10 words.jpeg')

# Least common 5 words

# In[27]:


freqdist.most_common()[-5:]

# Get Bigrams from text
bigrams = nltk.bigrams(allwords)
# Calculate Frequency Distribution for Bigrams
freq_bi = nltk.FreqDist(bigrams)

# Draw a bar chart with the count of the most common 20 words
x,y = zip(*freq_bi.most_common(n=20))
plt.figure(figsize=(16,5))
plt.barh(range(len(x)), y, color = 'Maroon')
#plt.barh(x,range(len(x)), color = 'Orange', tick_label = y)
#plt.xticks(range(len(x)), x)
y_pos = np.arange(len(x))
plt.yticks(y_pos, x)
plt.title('Frequency Count of Top 20 Bi-Grams')
plt.ylabel('Frequent Words')
plt.xlabel('Count')
plt.show()
plt.savefig('Most Frequent 20 Bi-Grams.jpeg')

# Get Trigrams from text
trigrams = nltk.trigrams(allwords)
# Calculate Frequency Distribution for Bigrams
freq_tri = nltk.FreqDist(trigrams)

# Print and plot most common bigrams
freq_tri.most_common(20)

# Draw a bar chart with the count of the most common 50 words
x,y = zip(*freq_tri.most_common(n=20))
plt.figure(figsize=(16,5))
plt.barh(range(len(x)), y, color = 'Darkblue')
#plt.barh(x,range(len(x)), color = 'Orange', tick_label = y)
#plt.xticks(range(len(x)), x)
y_pos = np.arange(len(x))
plt.yticks(y_pos, x)
plt.title('Frequency Count of Top 20 Tri-Grams')
plt.ylabel('Frequent Words')
plt.xlabel('Count')
plt.show()
plt.savefig('Most Frequent 20 Tri-Grams.jpeg')

# ## 4. Word Length Distribution Plot

# #### This plot is word length on x-axis vs number of words of that length on the y-axis. This plot helps to visualise the composition of different word length in the text corpus.

from nltk.probability import ConditionalFreqDist

cfdist = ConditionalFreqDist((len(word), word) for word in allwords )
cfdist.plot()


# ## t-SNE Corpus Visualization

#  Visualizing document similarity is to use t-distributed stochastic neighbor embedding

from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf  = TfidfVectorizer()

docs   = tfidf.fit_transform(allwords)

# Create the visualizer and draw the vectors
tsne = TSNEVisualizer()
tsne.fit(docs)
tsne.poof()

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

text_counts= cv.fit_transform(df[0])
