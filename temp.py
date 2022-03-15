from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from gtts import gTTS
from playsound import playsound  
import pyttsx3
#import os

app = Flask(__name__)

dic = {0: '10', 1: '100', 2: '20', 3: '200', 4: '2000', 5: '50', 6: '500'}

model = load_model('model.h5')

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(150, 150))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 150, 150, 3)
    p = np.argmax(model.predict(i), axis=-1)
    return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':

        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)
        textMsg = p+"Rupees"
        
        txt_speech = pyttsx3.init()
        
        txt_speech.say(textMsg)
        txt_speech.runAndWait()
        
        #myobj = gTTS(text=textMsg,lang='en',slow=False)


        #myobj.save("welcome.mp3")
        #print(myobj)

    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True)
    #playsound("welcome.mp3")
    #os.system("start welcome.mp3")