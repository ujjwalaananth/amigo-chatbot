# amigo-chatbot
Chatbot Web App using Context Identification and Sequence-to-Sequence Models


## Requirements
!pip install keras==2.3.1 
!pip install tensorflow==2.0.0 
!pip install keras_layer_normalization 
!pip install gTTS 
!pip install SpeechRecognition 
!pip install pyaudio 

### Methodology
There are three main steps in the pipeline:
- Categorizing the intent of speech
- Category-specific NLP model to generate response
- Web App Framework to integrate model along with speech-related functionalities.

### Web application framework

The web application framework integrated five important components:
- Associated word dictionaries: 
These dictionaries are based on training data, and contain the word indices which are used for tokenization
- Speech-to-Text Recognition and Transcription:
We used Google Speech API to record audio from the microphone. It adjusts for ambient noise and automatically stops recording when there is no input. This input is transcribed in Javascript and passed as input to our loaded chatbot model.
- Classifier for categorizing intent:
Given a TF-IDF vector of current and previous input data, the saved GaussianNB model accurately predicted conversation category. Accordingly, we used the chatbot model.
- The chatbot model:
The trained chatbot model, called Amigo (friend) and its associated pickle files were imported and used for predictions in the python code. 
- Text-to-Speech Synthesis:
The output of the chatbot model is a string, which is then voiced using the gTTS, or Google Text-to-Speech API. The output is also printed on the screen. 

### Future scope: 
- Collecting more conversational data, which would allow us to train deeper encoder and decoder networks without overfitting. This would also aid context memory within the models. 
- Adding functionality for specific use cases, such as alerting emergency contacts if required in the context of needing assistance, or offering location-based suggestions or directions.
- Adding more general functionality as well, such as using voice commands to set reminders for appointments, medication, etc.

### References 
- J. Ducharme, “One in Three Seniors Is Lonely. Here's How It’s Hurting Their Health”, TIME, March 4, 2019 (https://time.com/5541166/loneliness-old-age/) 
- A. Steptoe, A. Shankar, P. Demakakos, J. Wardle, “Social isolation, loneliness, and all-cause mortality in older men and women,” Proc. National Acad Sci USA, 2013. vol. 110, no. 15, pp. 5797-801. 
- A. P. Dickens, S.H. Richards, C.J. Greaves, et al., “Interventions targeting social isolation in older people: a systematic review,” BMC Public Health, vol. 11, no. 647, 2011, doi:10.1186/1471-2458-11-647 
- P. Khosravi, A, Rezvani, A. Wiewiora, "The impact of technology on older adults’ social isolation,"  Computers in Human Behaviour, vol. 63, pp. 594-603, 2016. 
- W. Wang, S. Hosseini, A. H. Awadallah, P. N, Bennett, C. Quirk, “Context-Aware Intent Identification in Email Conversations,” Proc. 42Nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) 2019}, Paris, France, pp. 585--594, doi: 10.1145/3331184.3331260. 
- I. Sutskever, O. Vinyals, Q. Le, “Sequence to Sequence Learning with Neural Networks,” Proc 27th International Conference on Neural Information Processing Systems (NIPS) 2014, Montreal, Canada, pp. 3104-3112. [Online]. Available: http://dl.acm.org/citation.cfm?id=2969033.2969173. 


