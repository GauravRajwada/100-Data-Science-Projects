# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 08:56:34 2020

@author: Gaurav
"""

from flask import Flask, render_template, request
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import nltk
import seaborn as sb
import pandas as pd
# nltk.download("wordnet", "whatever_the_absolute_path_to_myapp_is/nltk_data/")
classifier=pickle.load(open("sentiment_model1.pkl","rb"))
cv=pickle.load(open("sentiment_vectorizer.pkl","rb"))


lm=WordNetLemmatizer()

emotions={0:"Angry",1:"Sad",2:'Fear',3:"Surprise",4:"Joy",5:"Love"}

app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
@app.route("/",methods=['GET'])

def Home():
    return render_template('index.html')
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = '0'
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/predict",methods=['POST'])
def predict():
    
    if request.method=='POST':
        review=request.form['review']
        if review.isnumeric():
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
            
        else:
            corpus=[]
            # review="i hate you you are vary bad"
            text=re.sub('[^a-zA-Z]'," ",review)
            text_lower=text.lower()
            lower_list=text_lower.split()
            lower_list=[lm.lemmatize(i) for i in lower_list if i not in set(stopwords.words('english'))]
            clean_text=" ".join(lower_list)
            corpus.append(clean_text)
            x=cv.transform(corpus).toarray()
            output=classifier.predict(x)
            output=emotions[output[0]]
            string="""
                Natural language process take following steps:\n

                    1. Removing special charecter: "{}" \n
                    2. Lowering all words: "{}"\n
                    3. Lemmatization- It usually refers to remove inflectional endings only 
                        and to return the base: "{}"\n
                    4. Making Corpus: "{}"\n
                    5. Making Bag of words: "{}"\n
                    6. Finally prediction: "{}"\n
            """.format(text,text_lower,clean_text,corpus,x,output)
            print(string)
            prob=pd.DataFrame({'Emotions': ['Angry',"Sad","Fear","Surprise","Joy","Love"], 'Probability': classifier.predict_proba(x)[0,:]})
            prob=prob.sort_values(by='Probability',ascending=False)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize =(10, 7)) 
  
            plt.bar(prob['Emotions'], prob['Probability']) 
            plt.savefig("static/people_photo/plot.png")

            path="static/people_photo/plot.png"
         
            return render_template('result.html',review=review,
                                   text=text,
                                   text_lower=text_lower,
                                   lower_list=lower_list,
                                   clean_text=clean_text,
                                   corpus=corpus,
                                   vector=x,
                                   output=output,ax=path
                                   )
    else:
        return render_template('index.html')
    


if __name__=="__main__":
    app.run()








