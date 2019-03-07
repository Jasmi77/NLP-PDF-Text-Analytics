


# Import Libraries
import PyPDF2 
import os, sys
import re

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
punctuation = ['(',')',';',':','[',']',',']
stop_words = stopwords.words('english')

ca_one = str(sys.argv[1])
ca_two = str(sys.argv[2])

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
	
directory =  ca_one #'C:\\Users\\l1jxp04\\AI Notebooks\\NLP\\NLP_Demo\\GreenSheets-2012\\GreenSheets-2012'
# 'C:\Users\l1jxp04\AI Notebooks\NLP\NLP_Demo\GreenSheets-2012\GreenSheets-2012'

for file in os.listdir(directory):
    if not file.endswith(".pdf"):
        continue
    pdf_filename =  os.path.join(directory,file)
    search_for =  ca_two #'Percent'
    result = searchInPDF(pdf_filename,search_for)
    print("Filename : " + file)
    print("Word count :" + str(result))