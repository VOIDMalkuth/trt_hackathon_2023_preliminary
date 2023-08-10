import random

import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import cv2
import datetime
from canny2image_TRT import hackathon

hk = hackathon(export_calib_data=True)
hk.initialize()
for j in range(2):
    for i in range(20):
        path = "/home/player/pictures_croped/bird_"+ str(i) + ".jpg"
        img = cv2.imread(path)
        start = datetime.datetime.now().timestamp()
        new_img = hk.process(img,
                random.choice(["a bird", "a cat", "dog", "an aeroplane"]), 
                "best quality, extremely detailed", 
                "longbody, lowres, bad anatomy, bad hands, missing fingers", 
                1, 
                256, 
                random.randint(15, 20),
                False, 
                1, 
                9, 
                random.randint(0, 10000000), 
                0.0, 
                100, 
                200)
        end = datetime.datetime.now().timestamp()


