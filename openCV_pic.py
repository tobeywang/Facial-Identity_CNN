
def hello(txt):
    print('your input:'+txt)

#抓臉
import os
from PIL import Image
import numpy as np
import cv2 as cv
import glob
import logging 

def getImg():
    flist = glob.glob('img/*')
    return flist

def getIdolImg():
    flist=getImg()
    for (i, f) in enumerate(flist):
        mask_1=cv.imread(f)
        #haarcascade_frontalface_default.xml:Haar級聯數據
        face_cascade = cv.CascadeClassifier('openCV\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
        #圖片數據 ScaleFactor：每次搜尋方塊減少的比例 minNeighbers：每個目標至少檢測到幾次以上，才可被認定是真數據。minSize：設定數據搜尋的最小尺寸 (不設定不限制) ex:minSize=(40,40)
        faces = face_cascade.detectMultiScale(mask_1, 1.3, 5)
        if(len(faces)==0):
            logging.warning('%s :face is not exist' % (f))
            pass
        face_int=1
        for (x,y,w,h) in faces:
            #x,y,w,h = faces[0]
            img = Image.open(f)
            crpim = img.crop((x,y, x + w, y + h)).resize((64,64))#截臉
            #tag face:圖片數據 兩個對角座標 線的顏色 線的粗細
            face_img=cv.rectangle(mask_1,(x,y),(x+w,y+h),(14,201,255),2)#框臉

            if not os.path.exists('img_face'):
                os.mkdir('img_face')
            base=os.path.basename(f)
            name = 'img_face/%s_face_%d.jpg' %(os.path.splitext(base)[0],face_int)
            crpim.save(name)#截臉
            face_int=face_int+1
        if not os.path.exists('img_face_plt'):
            os.mkdir('img_face_plt')
        name = 'img_face_plt/{}_face.jpg'.format(os.path.splitext(base)[0])
        cv.imwrite(name,face_img)#框臉

# #TF
import tensorflow as tf
#CNN Model Build
#import about cnn model have to package
from keras.models import Sequential 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Dense
def load_model(modelname):
    classifier= tf.keras.models.load_model(modelname)
    return classifier

def build_model():
    # Initialising the CNN
    classifier = Sequential()

    # Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64,
    3), activation = 'relu'))

    # Max Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Convolution
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

    # Max Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))


    # Flattening
    classifier.add(Flatten())

    # Fully Connected
    classifier.add(Dense(units = 128, activation = 'relu')) 
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 2, activation = 'softmax'))

    classifier.compile(optimizer = 'adam', 
                            loss ='categorical_crossentropy', 
                        metrics = ['accuracy'])
    return classifier

def build_data_unknow():
    unknow = glob.glob("valid/*")
    return unknow

def cnnLayer(classnum):
    ''' create cnn layer'''
    # 第一层
    W1 = weightVariable([3, 3, 3, 32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5) # 32 * 32 * 32 多个输入channel 被filter内积掉了

    # 第二层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5) # 64 * 16 * 16

    # 第三层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5) # 64 * 8 * 8

    # 全连接层
    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512, classnum])
    bout = weightVariable([classnum])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

#import keras have to packages
from keras.preprocessing.image import ImageDataGenerator
##show_unknow
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt 
##
def build_keras_model(IsTrain):
    cv.OPENCV_OPENCL_DEVICE=False
    #CNN FIT
    if(IsTrain):
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,     
                                    zoom_range = 0.2,      
                                    horizontal_flip = True 
                                    )
        test_datagen = ImageDataGenerator(rescale = 1./255)
        
        training_set = train_datagen.flow_from_directory(
        r'train/', target_size = (64, 64),
        batch_size = 10,
        class_mode = 'categorical')
        test_set = test_datagen.flow_from_directory(
        r'test/', target_size = (64, 64),
        batch_size = 10,
        class_mode = 'categorical')
        
        classifier=build_model()
        history = classifier.fit_generator(training_set,
                            nb_epoch=10,
                            nb_val_samples=10,
                            steps_per_epoch = 30,
                            verbose = 1,
                            validation_data = test_set)
        classifier.save('model_one.h5')
    else:
        #CNN Load
        classifier=load_model('model_best.h5')
        
    #MODEL LABEL
    """
    transform_dic = {
    'andy'  : 'Andy',
    'jeff'    : 'Jeff'
    }
    name_dic = {v:transform_dic.get(k) for k,v in training_set.class_indices.items()}
    """
    name_dic = {
    0  : 'Andy',
    1    : 'Jeff'
    }
    
    ##
    #show_unknow
    ##
    #(x + int(w/3)-70, y-10) 字的位置
    font = cv.FONT_HERSHEY_PLAIN
    unknow=build_data_unknow()
    unknow_int=1
    list_img_filepath=[]
    for (i, f) in enumerate(unknow):
        logging.info('%s :unknow face' % (f))
        mask_1=cv.imread(f)
        face_cascade = cv.CascadeClassifier('openCV\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(mask_1, 1.3, 5)
        if(len(faces)==0):
            logging.warning('%s :face is not exist' % (f))
            pass
        #x,y,w,h = faces[0]
        for (x,y,w,h) in faces:
            #x,y,w,h = faces[0]
            img = Image.open(f)
            crpim = img.crop((x,y, x + w, y + h)).resize((64,64))
            face_img=cv.rectangle(mask_1,(x,y),(x+w,y+h),(14,201,255),2)#框臉
        img = Image.open(f)
        base=os.path.basename(f)
        print('run %s' % (base))
        for x,y,w,h in faces:
            box = (x, y, x+w, y+h)
            crpim = img.crop(box).resize((64,64))
            target_image = image.img_to_array(crpim)
            target_image = np.expand_dims(target_image, axis = 0)
            res = classifier.predict_classes(target_image)[0]
            #取得概率
            num=classifier.predict(target_image)
            label=np.argmax(num, axis=1)[0] #label
            print(num)
            print(label)
            cv.rectangle(mask_1,(x,y),(x+w,y+h),(14,201,255),2)
            cv.putText(mask_1,name_dic.get(res), (x + int(w/3)-30, y-3), font, 1.5, (14,201,255), 2)
        #plt.figure(figsize=(8,6))
        #plt.imshow(cv.cvtColor(mask_1, cv.COLOR_BGR2RGB))
        name = '/static/{}_face.jpg'.format(os.path.splitext(base)[0])
        i_result=1
        while(os.path.isfile(name)):
            name='/static/%s_face_%d.jpg'%(os.path.splitext(base)[0],i_result)
            i_result=i_result+1
        cv.imwrite(name,mask_1)
        list_img_filepath.append(name)
    
    logging.info('%d face' % (unknow_int))
    return list_img_filepath
#build_keras_model()
#getIdolImg()
#hello('testing python')
