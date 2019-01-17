--------------------------------------------------Code Start---------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 23:13:53 2018

@author: Jasmi
Description: NLP code for Bis Data
Here is the data:
  - https://www.bis.org/list/research/index.htm
  - https://www.bis.org/cbhub/index.htm
  - https://www.bis.org/list/wpapers/index.htm
"""
import PyPDF2 
import os, sys

#import textract

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def searchInPDF(filename, key):
    occurrences = 0
    pdfFileObj = open(filename,'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    num_pages = pdfReader.numPages
    count = 0
    text = ""
    while count < num_pages:
        pageObj = pdfReader.getPage(count)
        count +=1
        text += pageObj.extractText()
    if text != "":
       text = text
#   else:
  #     text = textract.process(filename, method='tesseract', language='eng')
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    punctuation = ['(',')',';',':','[',']',',']
    stop_words = stopwords.words('english')
    keywords = [word for word in tokens if not word in stop_words and  not word in punctuation]
    for k in keywords:
        if key == k: occurrences+=1
    return occurrences

directory = 'C:\\My NLP Projects\\Bix Project\\'
#pdf_filename = '0330.pdf'
for file in os.listdir(directory):
    if not file.endswith(".pdf"):
        continue
    pdf_filename =  os.path.join(directory,file)
    search_for = 'Word'
    result = searchInPDF(pdf_filename,search_for)
    print(result)

--------------------------------------------------Code End---------------------------------------------------------

