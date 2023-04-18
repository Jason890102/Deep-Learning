import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import glob
import time
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Data:
    def __init__(self):
        return
    
    def file_split(self, file):
        os.mkdir(file) # 創建資料夾
        imagelist = os.listdir('C:/Users/User/Desktop/Deep Learning/Assignment-1 Image classiication/images')
        f = open(file + '.txt') # 打開train.txt檔

        for i in range(22, 50):
            os.mkdir(file + '/' + file + str(i)) # 創建50個類別的資料夾

            for txt in f.readlines(): #逐行讀取test.txt

                for image in glob.glob('images/' + imagelist[i] + '/*.*'): # 讀取images下所有檔案

                    if (txt[17:32] == image[17:32]): # 對比txt及image名稱一不一樣

                        if '.JPEG ' in txt[17:34]: # 1位數
                            shutil.copy('images/' + imagelist[i] + '/' + image[17:28] +'.JPEG', './' + file + '/'+ file + str(i))
                        elif '.JPEG' in txt[17:34]: # 2位數
                            shutil.copy('images/' + imagelist[i] + '/' + image[17:29] +'.JPEG', './' + file + '/'+ file + str(i)) # 複製檔案至test目錄
                        elif '.JPE' in txt[17:34]: # 3位數
                            shutil.copy('images/' + imagelist[i] + '/' + image[17:30] +'.JPEG', './' + file + '/'+ file + str(i))
                        elif '.JP' in txt[17:34]: # 4位數
                            shutil.copy('images/' + imagelist[i] + '/' + image[17:31] +'.JPEG', './' + file + '/'+ file + str(i))
                        elif '.J' in txt[17:34]: # 5位數
                            shutil.copy('images/' + imagelist[i] + '/' + image[17:32] +'.JPEG', './' + file + '/'+ file + str(i))

        f.seek(0)
    
    
    def dataset_split(self, file):
        if file == 'train':
            global  train_labels, train_images
            starttime = int(time.time())
            train_images = []
            image_list = os.listdir('train/')
            image_list.sort(key = lambda x: int(x[5:]))


            for i in range(len(image_list)):
                j = 0
                for images in glob.glob('train/' + image_list[i] + '/*'):
                    j+=1
                    if j == 121: # 只取前120張
                        break
                    img = cv2.imread(images) # 圖片讀檔
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉灰階
                    img = cv2.resize(img, (256, 256))
                    train_images.append(img)

            train_images = np.array(train_images)



            train_labels = []
            image_list = os.listdir('train/')
            image_list.sort(key = lambda x: int(x[5:]))

            for i in range(len(image_list)):
                j = 0
                for labels in glob.glob('train/' + image_list[i] + '/*'):
                    j+=1
                    if j == 121:
                        break
                    train_labels.append(i)
            train_labels = np.array(train_labels)

            endtime = int(time.time()) # 計時結束
            print("train_images:",train_images.shape)
            print(type(train_images))
            print("train_labels:",len(train_labels))
            print(type(train_labels))
            print('花了',endtime-starttime,'s')
            
            return train_images, train_labels
            
        elif file == 'validation':
            global validation_labels, validation_images
            starttime = int(time.time()) # 計時開始
            validation_images = []
            image_list = os.listdir('validation/')
            image_list.sort(key = lambda x: int(x[10:]))

            for i in range(len(image_list)):
                for images in glob.glob('validation/' + image_list[i] + '/*'):
                    img = cv2.imread(images) # 圖片讀檔
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉灰階
                    img = cv2.resize(img, (256, 256))
                    validation_images.append(img)

            validation_images = np.array(validation_images)



            validation_labels = []
            image_list = os.listdir('validation/')
            image_list.sort(key = lambda x: int(x[10:]))


            for i in range(len(image_list)):
                for labels in glob.glob('validation/' + image_list[i] + '/*'):
                    validation_labels.append(i)

            validation_labels = np.array(validation_labels)

            endtime = int(time.time()) # 計時結束
            print("validation_images:",validation_images.shape)
            print(type(validation_images))
            print("validation_labels:",len(validation_labels))
            print(type(validation_labels))
            print('花了',endtime-starttime,'s')
            
            return validation_images, validation_labels
            
        elif file == 'test':
            global test_labels, test_images
            starttime = int(time.time())
            # images
            test_images = []
            image_list = os.listdir('test/')
            image_list.sort(key = lambda x: int(x[4:]))

            for i in range(len(image_list)):
                for images in glob.glob('test/' + image_list[i] + '/*'):
                    img = cv2.imread(images) # 圖片讀檔
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉灰階
                    img = cv2.resize(img, (256, 256))
                    test_images.append(img)

            test_images = np.array(test_images)

            # labels
            test_labels = []
            image_list = os.listdir('test/')
            image_list.sort(key = lambda x: int(x[4:]))


            for i in range(len(image_list)):
                for labels in glob.glob('test/' + image_list[i] + '/*'):
                    test_labels.append(i)

            test_labels = np.array(test_labels)
            endtime = int(time.time())

            print("test_images:",test_images.shape)
            print(type(test_images))
            print("test_labels:",len(test_labels))
            print(type(test_labels))
            print('花了',endtime-starttime,'s')
            
            return test_images, test_labels
