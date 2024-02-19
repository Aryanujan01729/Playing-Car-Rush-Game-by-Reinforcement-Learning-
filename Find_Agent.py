
import os
import sys
import time
import random 
import pygame as pg
import cv2
import numpy as np

AGENT_THRESHOLD=1000

def find_contour(RGB):
            COLOR=["BLUE","LIGHT GREEN","YELLOW","ORANGE","DARK GREEN","PURPLE","MEDIUM GREEN","LIGHT RED","RED"]
            CAR_IMG=["police","traffic_car_2","yellow_car","traffic_car_4","green_car","purple_car","traffic_car_1","traffic_car_3","red_car"]
            CENTROID=[]
            PIXEL_COLOR=[]
            for img in CAR_IMG:
                car_image = pg.image.load(os.path.join(sys.path[0], f'assets/{img}.png'))
                car_image = pg.transform.scale(car_image, (75, 140))
                bbox = cv2.boundingRect(car_image)
                top_left, bottom_right = (bbox[0], bbox[1]),(bbox[0]+bbox[2], bbox[1]+bbox[3])
                cv2.rectangle(car_image, top_left, bottom_right, (255, 0, 0), 2)
                (startX,startY)= top_left
                (endX,endY)= bottom_right
                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                #CENTROID.append((cX,cY))
                color = car_image.get_at((cX, cY))
                if color == RGB:
                    return img



def find_object_frequency(cur_frame):
            CAR_IMG={"police":0 ,"traffic_car_2":0 ,"yellow_car":0 ,"traffic_car_4":0 ,"green_car":0 ,"purple_car":0 ,"traffic_car_1":0 ,"traffic_car_3":0 ,"red_car":0 }
            CAR_IMG=["police","traffic_car_2","yellow_car","traffic_car_4","green_car","purple_car","traffic_car_1","traffic_car_3","red_car"]
            IMAGE_PATH='assets\'
            IMAGES = {name: cv2.imread(IMAGE_PATH +'{}.png'.format(name),-1)#.convert_alpha()
            for name in CAR_IMG}
            for name in CAR_IMG:
                if IMAGES[name].shape[-1] == 4:
                    IMAGES[name][np.where(IMAGES[name][:, :, 3] == 0)] = (0, 0, 0, 255)
            method = cv2.TM_CCOEFF_NORMED
            TM_THRESHOLD = .78
            for img in CAR_IMG:
                frame1=cur_frame.copy()
                tW, tH = img.shape[::-1]
                res = cv2.matchTemplate(img,frame1,3)
                (yCoords, xCoords) = np.where(res >= TM_THRESHOLD)
                (startX,startY)=(xCoords, yCoords)
                (endX,endY)=(xCoords+tW, yCoords+tH)
                cX=(startX+endX)/2
                cY=(startY+endY)/2
                RGB=frame1.get_at((cX, cY))
                CAR_NAME=find_contour(RGB)
                CAR_IMG[CAR_NAME]=CAR_IMG[CAR_NAME]+1
                if CAR_IMG[CAR_NAME] >= AGENT_THRESHOLD:
                    return CAR_NAME