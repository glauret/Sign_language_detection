# CV2 module

import cv2
from handdetector import HandDetector
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


# Import component
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

#builtin python
import av
import queue
import urllib
import urllib.request
import os
from pathlib import Path

#add_fronta
app_formal_name = "🧙 Sign detection 🐍"

# Start the app in wide-mode

st.set_page_config(
    layout="wide", page_title=app_formal_name,
)

# Starting random state
random_state = 78

# Number of results to return
top_k = 12

# initialise the title
title_element = st.empty()
info_element = st.empty()
bottom_element = st.empty()

#title
title = f"<h1 style='text-align: center; font-family:verdana;'>Sign Language Detection</h1>"
#info
info = '''
<style>{
div[data-testid="stHorizontalBlock"] > div:first-of-type {
  background-image: url("https://images.pexels.com/photos/7516363/pexels-photo-7516363.jpeg?cs=srgb&dl=pexels-shvets-production-7516363.jpg&fm=jpg")
}
}</style>
<p>A real-time sign language translator permit communication between the deaf
community and the general public. 🤙</p>
<p>We hereby present the development and implementation of an American Sign
Language fingerspelling translator based on a
convolutional neural network. 🚀</p>
<p>Made with 💙 by <a href='https://github.com/jvesp/sld'>Detection language team</a></p>'''.strip()
"""
[![Star](https://img.shields.io/github/stars/jvesp/sld.svg?logo=github&style=social)](https://github.com/jvesp/sld)
"""

title_element.markdown(title, unsafe_allow_html=True)
info_element.write(info, unsafe_allow_html=True)

#backgroud image
empty_col_bg, col_bg = st.columns([0.5, 1.5])



st.markdown("<br>", unsafe_allow_html=True)



how_work="""Jumpstart your machine learning code:<br>
1. Select your device<br>
2. Click on start<br>
3. Show your hands and do magic! :sparkles:<br>

---
"""
# création de deux colonnes
col1, empty_mid, col2 = st.columns([1.2, 0.2 , 1.5])
col1.image("./images/img_sign_main.JPG")

bottom_element.write(how_work, unsafe_allow_html=True)

#dictionary of traduction letters
dict_letter = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L', 11:'M',
               12:'N', 13:'O', 14:'P', 15:'Q', 16:'R', 17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y' }

COLORS = np.random.uniform(0, 255, size=(len(dict_letter), 3))

dict_colors = {}
for i in range(24):
    dict_colors[i] = COLORS[i]

#Set up STUN servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#getting cwd
HERE = Path(__file__).parent
print(os.listdir(HERE))
if not 'model.h5' in os.listdir(HERE):
    txt = st.warning("Téléchargement du modèle")
    print("loading model")
    url = 'https://www.dropbox.com/s/sffb5ew98us9gxa/model_resnet50_V2_8830.h5?dl=1'
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open('model.h5', 'wb') as f:
        f.write(data)
    print("model loaded")
    txt.success("Téléchargement terminé")


#@st.cache(allow_output_mutation=True)
@st.experimental_singleton
def load_mo():
    model = load_model('model.h5')
    return model

# Your class where you put the intelligence
class SignPredictor(VideoProcessorBase):

    def __init__(self) -> None:
        # Hand detector
        self.hand_detector = HandDetector(detectionCon=0.8, maxHands=1)
        self.model = load_mo()
        self.counter = 0
        self.l=[]
        self.word=[]
        self.result_queue_word = queue.Queue()

    def find_hands(self, image):

        #add the rectangle in your image around the hands
        hands, image_hand = self.hand_detector.findHands(image)

        if hands:
            bbox1 = hands[0]["bbox"]  # Bounding box info x,y,w,h
            x, y, w, h = bbox1

            #récup img plus grande que la bbox1 de base
            x_square = int(x - 0.2 * w)
            y_square = int(y - 0.2 * h)
            w_square = int(x + 1.2 * w)
            h_square = int(y + 1.2 * h)

            #anticipe erreur de x, y négatifs
            if x_square < 0 :
                x_square = 0
            if y_square < 0:
                y_square = 0
            if h_square < 0:
                h_square = 0
            if w_square < 0:
                w_square = 0

            hand_img = image_hand[y_square:h_square,x_square:w_square]  # image of the hand
            img_hand_resize = np.array(
                tf.image.resize_with_pad(hand_img, 256, 256))  # resize image to match model's expected sizing
            img_hand_resize = img_hand_resize.reshape(1, 256, 256, 3)
            img_hand_resize = tf.math.divide(img_hand_resize, 255)

            #couleur img_main
            channels = tf.unstack(img_hand_resize, axis=-1)
            img_hand_resize = tf.stack([channels[2], channels[1], channels[0]],
                                       axis=-1)

            prediction = self.model.predict(img_hand_resize)[0]

            probabs = round(prediction[np.argmax(prediction)], 2)
            pred = np.argmax(prediction)

            self.counter +=1
            if self.counter % 1 == 0:
                cv2.putText(image_hand, f'{dict_letter[pred]}',
                            (int(x_square + 1.5 * w), int(h_square - 0.7 * h)),
                            cv2.FONT_HERSHEY_PLAIN, 2, dict_colors[pred], 2)
                cv2.putText(image_hand, f'{str(probabs)}',
                            (int(x_square + 1.5 * w), int(h_square - 0.5 * h)),
                            cv2.FONT_HERSHEY_PLAIN, 2, dict_colors[pred], 2)

                if probabs > 0.90:
                    self.l.append(dict_letter[pred])

                # COLORING BOX
                cv2.rectangle(
                    image_hand, (x_square, y_square),
                    (w_square, h_square),
                    (dict_colors[pred]), 2)

                # WORD CREATION
                if len(self.l)==10:
                    self.word.append(max(set(self.l), key=self.l.count))
                    self.result_queue_word.put(self.word)# QUEUE IN STREAMLIT
                    self.l=[]
        else:
            if self.word:
                if self.word[-1] != " ":
                    self.word.append(" ")
                    self.result_queue_word.put(self.word)

        return hands, image_hand

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        hands, annotated_image = self.find_hands(image)
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

with col2:
    webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=SignPredictor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

# Final word
final_word = ""

if webrtc_ctx.state.playing:
    labels_placeholder = st.empty()
    while True:
        if webrtc_ctx.video_processor:
            try:
                result = webrtc_ctx.video_processor.result_queue_word.get(timeout=1.0)
                final_word = ""
                for value in result:
                    final_word = final_word + value
                labels_placeholder.title(final_word)
            except queue.Empty:
                result = final_word
        else:
            break
