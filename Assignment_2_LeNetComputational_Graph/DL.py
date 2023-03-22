class DL:
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
            global  train_labels, d2_train_images
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
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉灰階
                    img = cv2.resize(img, (256, 256))
                    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
                    train_images.append(hog_image)

            train_images = np.array(train_images)
            print(train_images.shape)
            nsamples, nx, ny = train_images.shape
            d2_train_images = train_images.reshape((nsamples,nx*ny))
            print(type(d2_train_images))
            print(d2_train_images.shape)
            print(d2_train_images)


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
            endtime = int(time.time())
            print('花了',endtime-starttime,'s')
            print(type(train_labels))
            print(len(train_labels))
            print(train_labels)
            
            return train_labels, d2_train_images
            
        elif file == 'validation':
            global validation_labels, d2_validation_images
            starttime = int(time.time())
            validation_images = []
            image_list = os.listdir('validation/')
            image_list.sort(key = lambda x: int(x[10:]))

            for i in range(len(image_list)):
                for images in glob.glob('validation/' + image_list[i] + '/*'):
                    img = cv2.imread(images) # 圖片讀檔
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉灰階
                    img = cv2.resize(img, (256, 256))
                    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
                    validation_images.append(hog_image)

            validation_images = np.array(validation_images)
            print(validation_images.shape)
            nsamples, nx, ny = validation_images.shape
            d2_validation_images = validation_images.reshape((nsamples,nx*ny))
            print(type(d2_validation_images))
            print(d2_validation_images.shape)
            print(d2_validation_images)


            validation_labels = []
            image_list = os.listdir('validation/')
            image_list.sort(key = lambda x: int(x[10:]))


            for i in range(len(image_list)):
                for labels in glob.glob('validation/' + image_list[i] + '/*'):
                    validation_labels.append(i)

            validation_labels = np.array(validation_labels)

            endtime = int(time.time())
            print(type(validation_labels))
            print(len(validation_labels))
            print(validation_labels)
            print('花了',endtime-starttime,'s')
            
            return validation_labels, d2_validation_images
            
        elif file == 'test':
            global test_labels, d2_test_images
            starttime = int(time.time())
            test_images = []
            image_list = os.listdir('test/')
            image_list.sort(key = lambda x: int(x[4:]))

            for i in range(len(image_list)):
                for images in glob.glob('test/' + image_list[i] + '/*'):
                    img = cv2.imread(images) # 圖片讀檔
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉灰階
                    img = cv2.resize(img, (256, 256))
                    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
                    test_images.append(hog_image)

            test_images = np.array(test_images)
            print(test_images.shape)
            nsamples, nx, ny = test_images.shape
            d2_test_images = test_images.reshape((nsamples,nx*ny))
            print(type(d2_test_images))
            print(d2_test_images.shape)
            print(d2_test_images)


            test_labels = []
            image_list = os.listdir('test/')
            image_list.sort(key = lambda x: int(x[4:]))


            for i in range(len(image_list)):
                for labels in glob.glob('test/' + image_list[i] + '/*'):
                    test_labels.append(i)

            test_labels = np.array(test_labels)
            endtime = int(time.time())

            print(type(test_labels))
            print(len(test_labels))
            print(test_labels)
            print('花了',endtime-starttime,'s')
            
            return test_labels, d2_test_images
