'''
!pip install keras==2.3.1
!pip install tensorflow==2.0.0
!pip install keras_layer_normalization
'''
import speech_recognition as sr 
#from sklearn.externals import joblib
import joblib
from datetime import datetime
from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import pickle
import numpy as np
import os
import keras
from keras import backend as K
import pyttsx3 
from datetime import date
# the magic words
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

#import tensorflow as tf
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras_layer_normalization import LayerNormalization
import pickle
from keras import layers , activations , models , preprocessing
from keras.models import load_model
from keras import utils


#load all models
vect = joblib.load('vectorizer.pkl')
category_clf = joblib.load('category_clf.pkl')
lastText = ' '
print('this part worked',lastText)    
enc_gen = load_model('enc_generic1.h5',custom_objects={'LayerNormalization': LayerNormalization()})
dec_gen = load_model('dec_generic1.h5',custom_objects={'LayerNormalization': LayerNormalization()})
wi_gen = pickle.load(open("wi_gen.pkl","rb"))
iw_gen = pickle.load(open("iw_gen.pkl","rb"))

enc_gre = load_model('enc_greetings.h5',custom_objects={'LayerNormalization': LayerNormalization()})
dec_gre = load_model('dec_greetings.h5',custom_objects={'LayerNormalization': LayerNormalization()})
wi_gre = pickle.load(open("wi_greetings.pkl","rb"))
iw_gre = pickle.load(open("iw_greetings.pkl","rb"))

enc_help = load_model('enc_help.h5',custom_objects={'LayerNormalization': LayerNormalization()})
dec_help = load_model('dec_help.h5',custom_objects={'LayerNormalization': LayerNormalization()})
wi_help = pickle.load(open("wi_help.pkl","rb"))
iw_help = pickle.load(open("iw_help.pkl","rb"))

enc_lon = load_model('enc_lonely.h5',custom_objects={'LayerNormalization': LayerNormalization()})
dec_lon = load_model('dec_lonely.h5',custom_objects={'LayerNormalization': LayerNormalization()})
wi_lon = pickle.load(open("wi_lon.pkl","rb"))
iw_lon = pickle.load(open("iw_lon.pkl","rb"))


def clean_text(text):	#Clean text by removing unnecessary characters and altering the format of words.
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:{}`+=~|.!?,]", "", text)   
    return text



sample_rate = 48000
chunk_size = 2048
r = sr.Recognizer() 
mic_list = sr.Microphone.list_microphone_names() 
device_id = 0# APP STARTS HERE


app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/startrec")
def startornot():
    with sr.Microphone(device_index = device_id, sample_rate = sample_rate, chunk_size = chunk_size) as source: 
        r.adjust_for_ambient_noise(source) 

    return 'true'

@app.route("/recog")
def speechrecog():

    with sr.Microphone(device_index = device_id, sample_rate = sample_rate, chunk_size = chunk_size) as source: 
        audio = r.listen(source)    

    print('recorded')
    
    try: 
        text = r.recognize_google(audio)
        return text
  
    except:
        return '00'


@app.route("/get")

def get_bot_response():
    userText = request.args.get('msg')
    combined_text = userText #+ lastText
    #lastText = userText
    #category = category_clf.predict(vect.transform(userText))
    category = category_clf.predict((vect.transform([combined_text])).toarray())

    if(category == 'generic1'):
        enc_model = enc_gen
        dec_model = dec_gen
        word_to_index = wi_gen
        index_to_word = iw_gen
    elif (category == 'greetings'):
        enc_model = enc_gre
        dec_model = dec_gre
        word_to_index = wi_gre
        index_to_word = iw_gre
    elif (category == 'help'):
        enc_model = enc_help
        dec_model = dec_help
        word_to_index = wi_help
        index_to_word = iw_help
    else:
        enc_model = enc_lon
        dec_model = dec_lon
        word_to_index = wi_lon
        index_to_word = iw_lon
    
    # STRING TO TOKENS

    maxlen_input=15
    maxlen_output=15

    unknown_token = '<unk>'

    tokens_list=[]
    message = clean_text(str('<beg> ') + userText + str(' <end>'))
    tokens_list.append([word_to_index[w] if w in word_to_index else word_to_index[unknown_token] for w in message.split()])
    
    print(tokens_list)
    print(sequence.pad_sequences(tokens_list , maxlen=maxlen_input , padding='post'))    
    text =  sequence.pad_sequences(tokens_list , maxlen=maxlen_input , padding='post')

    # PREDICT

    states_values = enc_model.predict( text )
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = word_to_index['<beg>']
    stop_condition = False
    decoded_translation = ''
    final_trans = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values ) #
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        stop_condition = (sampled_word_index == word_to_index['<end>'])  
        decoded_translation = decoded_translation + index_to_word[sampled_word_index] + str(' ')        
 
        if len(decoded_translation.split()) > maxlen_output:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    lword = ''
    for w in decoded_translation.split()[1:]:
      if(w != lword and w!='<end>'):
        final_trans = final_trans + w + str(' ')        
        lword = w

    if (final_trans=='' or final_trans==' '):
        final_trans = 'please tell me more...'

    if('time' in userText):
        t = str(datetime.time(datetime.now()))

        t = re.sub(r":", " ", t)

        h = t.split()
        ampm = 'am'
        
        if(int(h[0])>12):
            h[0] = str(int(h[0])-12)
            ampm='pm'
            
        tim = h[0]+':'+h[1]+' '+ampm        
        final_trans = 'the time is '+tim
    
    if('date' in userText):
    	final_trans = 'the date is '+str(datetime.date(datetime.now()).strftime("%d %B, %Y").lower())
    elif('day' in userText):
    	my_day = datetime.weekday(datetime.date(datetime.now()))
    	cal = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    	final_trans = 'the day is '+str(cal[my_day].lower())


    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-GB_HAZEL_11.0')
    '''
    for voice in voices:
        print(voice.id, voice.name, voice.languages , voice.gender)
    '''
    engine.say(final_trans+' ')
    engine.runAndWait()



	
    return( final_trans )

    


if __name__ == "__main__":
    app.run(threaded=False)
