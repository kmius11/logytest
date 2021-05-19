from flask import Flask, jsonify
from flask import Flask
from flask import render_template, redirect, url_for,flash
from flask import request, Response

import cv2 as cv
import os
import time
import imutils
import numpy as np
from imutils.video import VideoStream

app= Flask(__name__)

faceClassif = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')
videoStream = VideoStream(src=0).start()
#port es el puerto donde quiero alojar la aplicacion
#El debug true actualiza la aplicacion y se realizan cambios.
@app.route("/")
def index():   
    return render_template("index.html")

  
def generateFrames():
    
    ##frameCapt = cv2.VideoCapture(0)
           
    while True:
        #ret,frame = frameCapt.read()
        frame = videoStream.read()
        #frame = imutils.resize(frame, width=600,height=400, conts=3)
        #(w, h, c) = frame.shape
        #syntax: cv2.resize(img, (width, height))
        frame = cv.resize(frame,(600, 400))
        #print(w, h)
        #print(frame.shape)        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)        
        auxFrame = frame.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:

            cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv.resize(rostro,(150,150), interpolation=cv.INTER_CUBIC)                          
                                   
            #frame = cv2.imencode('.jpg', frame)[1].tobytes()
            cv.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
            cv.putText(frame,'Rostro detectado... Por favor ingrese su usuario',(10,20), 2, 0.5,(0,0,0),1,cv.LINE_AA)
            
            #cv2.imshow('frame',frame)              
        (flag, encodedImage) = cv.imencode(".jpg", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_stream")
def video_stream():
    print("FN REGISTRO")    
    return Response(generateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ =='__main__':
    #app.run(host="0.0.0.0",port=4000,debug=True)
    app.run()
