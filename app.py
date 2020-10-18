# -*- coding: utf-8 -*-
"""
Created on Firday Jun 14 2019
@author: Jay Prakash Kumar
"""
import argparse
import logging
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

import tkinter.font as font
import webbrowser
import random
from train_video import VideoTrain


from warehouse_box import Box
from configuration import configuration_model
#import warehouse_box as box

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random




from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import itertools
import pandas as pd

import os



class WarehouseModule:
    def __init__(self, logFileName):

        self.logFileName = logFileName
        self.window = tk.Tk()
        #self.root= tk.Tk()
        # helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
        self.window.title("Warehouse Equipment Analysis")

        # this removes the maximize button
        self.window.resizable(0, 0)
        window_height = 600
        window_width = 880

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

        header = tk.Label(self.window, text="Warehouse Objects Prediction", width=80, height=2, fg="white", bg="SlateGray",
                          font=('times', 18, 'bold', 'underline'))
        header.place(x=0, y=0)




        # my_label = Label(top, text=self.window.filename).pack()
        my_image = ImageTk.PhotoImage(Image.open('datasets/vinayak_1_0032.jpg'))

        my_image_label = tk.Label(image=my_image)
        my_image_label.place(y=60)

        employeetID = tk.Label(self.window, text="Emp ID", width=10, height=2, fg="white", bg="SaddleBrown",
                            font=('times', 15))
        employeetID.place(x=80, y=80)

        displayVariable = StringVar()



        self.employeeIDTxt = tk.Entry(self.window, width=20, text=displayVariable, bg="white", fg="black",
                               font=('times', 15, 'bold'))
        self.employeeIDTxt.place(x=205, y=80)

        employeeName = tk.Label(self.window, text="Emp Name", width=10, fg="white", bg="SaddleBrown", height=2, font=('times', 15))
        employeeName.place(x=450, y=80)

        self.empIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empIDTxt.place(x=575, y=80)

        text= 'This is a warehouse object prediction software. ' \
              'You can predict different objects used in warehouse.'
        messageDisplay = tk.Label(self.window, text=text, fg="white", bg="SaddleBrown", height=5, font=('times', 12))
        messageDisplay.place(x= 40, y=140)


        takeImg = tk.Button(self.window, text="Predict Images", command=self.predImg, fg="white", bg="SaddleBrown", width=15,
                            height=2,
                            activebackground="LightSlateGray", font=('times', 15, ' bold '))
        takeImg.place(x=120, y=400)

        takeVideoImg = tk.Button(self.window, text="Predict Video", command=self.predictVideo, fg="white", bg="SaddleBrown", width=15,
                             height=2,
                             activebackground="LightSlateGray", font=('times', 15, ' bold '))
        takeVideoImg.place(x=400, y=400)


        quitWindow = tk.Button(self.window, text="Quit", command=self.close_window, fg="white", bg="DarkRed", width=10, height=2,
                               activebackground="LightSlateGray", font=('times', 15, 'bold'))
        # quitWindow1 = tk.Button(self.root, text="Quit", command=self.close_root, fg="white", bg="#363e75", width=10,
        #                        height=2,
        #                        activebackground="#118ce1", font=('times', 15, 'bold'))
        quitWindow.place(x=650, y=510)
        #quitWindow1.place(x=650, y=510)

        link2 = tk.Label(self.window, text="Jay prakash kumar", fg="red", )
        link2.place(x=750, y=580)
        # link2.pack()
        link2.bind("<Button-1>", lambda e: self.callback(""))
        label = tk.Label(self.window)

        self.window.mainloop()

        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename=self.logFileName,
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')




    def predImg(self):
        global my_image, my_image1   # Need to def inside variable as global bec python sees it as garbage

        #root= Tk()
        self.window.filename= filedialog.askopenfilename(initialdir= os.getcwd(), title='Select a Image',
                                                         filetypes=(('png files', '*.png'),
                                                                    ('All files', '*.*')))

        predictor = configuration_model()


        top= Toplevel()
        my_label = Label(top, text=self.window.filename).pack()
        my_image = ImageTk.PhotoImage(Image.open(self.window.filename))
        print('name....', my_image)
        print('label...', my_label)

        img = cv2.imread(self.window.filename)
        print('image name....', self.window.filename)
        print(img.shape)
        output = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata= warehouse_metadata, scale=0.5)
        out= visualizer.draw_instance_predictions(output['instances'].to('cpu'))
        img_out = Image.fromarray(out.get_image()[:,:,::-1])
        print('out is....', out)
        my_image1 = ImageTk.PhotoImage(image= img_out)


        # display1 = tk.Label(imageFrame)
        # display1.grid(row=1, column=0, padx=10, pady=2)
        # display1.imgtk = imgtk  # Shows frame for display 1
        # display1.configure(image=imgtk)

        #cv2.imshow("images", out.get_image()[:, :, ::-1])
        #cv2.imshow('image', img)

        print('image name1....', my_image1)
        #my_image_label= tk.Label(image= my_image, width=my_image.width(), height=my_image.height()).pack()
        panel = tk.Label(top, image=my_image1)
        panel.pack(side="bottom", fill="both", expand="yes")
        #my_image_label.place(x= 250, y= 100)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    def predictVideo(self):
        id= (self.employeeIDTxt.get())
        name= (self.empIDTxt.get())
        print('id is.....', id)

        video= VideoTrain('Jay kumar', warehouse_metadata, id, name)




    def close_window(self):
        self.window.destroy()

    def close_root(self):
        self.root.destroy()

    def callback(self, url):
        webbrowser.open_new(url)

classes= ['Pallet Jacks', 'Rolling Ladder', 'Wire Mesh', 'Bulk Box', 'Totes',
       'Dump Hopper', 'Bin', 'Yard Ramp']

for d in ["train", "test"]:
    DatasetCatalog.register("experiment1/" + d, lambda d=d: Box.get_warehouse_box("/var/warehouse/resized/images/data_14.csv", "//var/warehouse/resized/images/"+ d +"/"))
    MetadataCatalog.get("experiment1/" + d).set(thing_classes=classes)
warehouse_metadata = MetadataCatalog.get("experiment1/train")
print('metadata....', warehouse_metadata)



# dataset_dicts = get_warehouse_box('/home/jay/warehouse/resized/images/train_labels.csv',
#                                   '/home/jay/warehouse/resized/images/train/')
#
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=warehouse_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow("images", out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)
#
# cv2.destroyAllWindows()



logFileName = "ProceduralLog.txt"
regStrtnModule = WarehouseModule(logFileName)
