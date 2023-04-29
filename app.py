import flask
import pickle
import re
import pandas as pd
import nltk 
from flask import Flask,render_template

# use pickle to load in the pretrained model
with open(f'model/sid.pkl','rb') as f:
 model = pickle.load(f)




# function for the review test processing
def pre_process(text):
    # Remove links
    text = re.sub('http://\S+|https://\S+', '', text)
    text = re.sub('http[s]?://\S+', '', text)
    text = re.sub(r"http\S+", "", text)

    # Convert HTML references
    text = re.sub('&amp', 'and', text)
    text = re.sub('&lt', '<', text)
    text = re.sub('&gt', '>', text)

    # Remove new line characters
    text = re.sub('[\r\n]+', ' ', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove multiple space characters
    text = re.sub('\s+',' ', text)
    
    # Convert to lowercase
    text = text.lower()
    return text
    


app = flask.Flask(__name__,template_folder='templates')

@app.route('/', methods=['GET','POST'])
def index():
     
     if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    
     if flask.request.method == 'POST':
        
        text = flask.request.form['text']

        df = pd.DataFrame([text], columns=['text'])


        df['text'] = df['text'].apply(pre_process)

        final_text = df['text']

        

        prediction = model.polarity_scores(final_text)
        compound = prediction['compound']
        
        return flask.render_template('index.html', result=compound, original_input={'Review':text})
     

     if __name__ == '__main__':
        app.debug = True
        app.run

   




        

 