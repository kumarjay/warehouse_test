# -*- coding: utf-8 -*-
"""
Created on Firday Jun 14 2019
@author: Jay Prakash Kumar
"""
import argparse
import logging
import tkinter as tk
from tkinter import *
import numpy as np
import pandas as pd
import cv2
import os
from PIL import ImageTk, Image
import pafy
import youtube_dl

from datetime import datetime
from configuration import configuration_model

from detectron2.utils.visualizer import Visualizer
import time




class VideoTrain:
    def __init__(self, logFileName, warehouse_metadata, id, name):

        self.logFileName = logFileName
        self.warehouse_metadata= warehouse_metadata
        self.window = tk.Tk()
        # helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
        self.window.title("Warehouse Equipment Analysis")
        self.id= id
        self.name= name

        # this removes the maximize button
        self.window.resizable(0, 0)
        window_height = 550
        window_width = 800

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))

        self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        # window.geometry('880x600')
        self.window.configure(background='#ffffff')

        # window.attributes('-fullscreen', True)

        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        header = tk.Label(self.window, text="Predict on Videos and Live WebCam....", width=80, height=2, fg="white", bg="SlateGray",
                          font=('times', 18, 'bold', 'underline'))
        header.place(x=0, y=0)
        employeeInfo = tk.Label(self.window, text="Employee Info", width=10, height=2, fg="white", bg="SaddleBrown", font=('times', 15))
        employeeInfo.place(x=80, y=80)

        displayVariable = StringVar()
        self.employeeIDTxt = tk.Label(self.window, width=40, text=' ID : '+ self.id+'     NAME :  '+self.name, bg="white", fg="black",
                               font=('times', 15, 'bold'))
        self.employeeIDTxt.place(x=205, y=80)


        webcamVideo = tk.Button(self.window, text="Predict using WebCam", fg="white", bg="SaddleBrown", width=30,
                            height=2,
                            activebackground="SlateGray", font=('times', 15, ' bold '))
        webcamVideo.place(x=80, y=150)

        predVideo = tk.Button(self.window, text="Predict on Video", fg="white", bg="SaddleBrown", width=30,
                             height=2,
                             activebackground="SlateGray", font=('times', 15, ' bold '))
        predVideo.place(x=80, y=250)

        urlLInk = tk.Label(self.window, text="URL Link :", width=10, fg="white", bg="SlateGray", height=2,
                           font=('times', 15))
        urlLInk.place(x=80, y=330)

        self.urlLInkTxt = tk.Entry(self.window, width=40, bg="white", fg="black", font=('times', 15, ' bold '))
        self.urlLInkTxt.place(x=205, y=330)

        youtubeVideo = tk.Button(self.window, text="Predict on YouTube Video", command=self.collectImagesFromCamera, fg="white", bg="SaddleBrown",
                             width=30,
                             height=2,
                             activebackground="SlateGray", font=('times', 15, ' bold '))
        youtubeVideo.place(x=80, y=400)


        quitWindow = tk.Button(self.window, text="Quit", command=self.close_window, fg="white", bg="DarkRed", width=10, height=2,
                               activebackground="SlateGray", font=('times', 15, 'bold'))
        quitWindow.place(x=620, y=400)

        link2 = tk.Label(self.window, text="Jay prakash kumar", fg="red", )
        link2.place(x=670, y=520)
        # link2.pack()
        link2.bind("<Button-1>", lambda e: self.callback(""))
        label = tk.Label(self.window)

        self.window.mainloop()

        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename=self.logFileName,
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')

    def collectImagesFromCamera(self):
        # initialize video stream
        #cap = cv2.VideoCapture('http://25.123.144.9:8080/video')
        predictor= configuration_model()

        #link = 'https://www.youtube.com/watch?v=C3G49oh0_nY'
        try:
            link = (self.urlLInkTxt.get())
            vPafy = pafy.new(link)
            play = vPafy.getbest()
        except:
            self.message = tk.Label(self.window, text="Please enter a url", bg="#bbc7d4", fg="black", width=58, height=2,
                                    activebackground="#bbc7d4",
                                    font=('times', 15))
            self.message.place(x=80, y=460)

        cap = cv2.VideoCapture(play.url)

        t1= time.time()
        while True:
            ret, frame = cap.read()
            #top = Toplevel()
            if time.time()-t1 >= 0.5:

                frame= cv2.resize(frame, (800,600))
                output = predictor(frame)
                objs = output['instances'].get('pred_classes')

                visualizer = Visualizer(frame[:, :, ::-1], metadata=self.warehouse_metadata, scale=0.5)

                out = visualizer.draw_instance_predictions(output['instances'].to('cpu'))

                img_out = Image.fromarray(out.get_image()[:, :, ::-1])
                print('out is....', out)
                my_image1 = ImageTk.PhotoImage(image=img_out)

                cv2.imshow("Frame", out.get_image()[:,:,::-1])
                t1= time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def close_window(self):
        self.window.destroy()