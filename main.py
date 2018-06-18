import face_recognition
import cv2
import numpy as np
from imutils import face_utils
import argparse
import imutils
import dlib
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import numpy as np
import os
import progressbar
from time import sleep

PATH = '/home/shaaran/PycharmProjects/robot_emotion/'
os.chdir('/home/shaaran/PycharmProjects/robot_emotion/')
sz = 224
arch = resnet34

def gitsearch():
# This part contains the main code.

        path_place  = '/home/shaaran/Downloads/Obama_out_-_President_Barack_Obama_s_hilarious_final_White_House_correspondents_dinner_speech-youtube-NxFkEj7KPC0-43-0-301.mp4' #file destination
        video_capture = cv2.VideoCapture(path_place) #starts the web cam if you attach it externally use 1 or 2 , use trail and error .For using the downloaded video replace with path_place
        detector = dlib.get_frontal_face_detector() #pretrained model for detecting frontal face
        predict_path = '/home/shaaran/PycharmProjects/shape_predictor_68_face_landmarks.dat'
        predictor = dlib.shape_predictor(predict_path) # initialzing the predictor
        count = 0 # counter for loop
        tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1) #transforamtions for getting a large and varied dataset from small datset
        data = ImageClassifierData.from_paths(PATH, tfms=tfms) #apply transforms to data 
        print(data.classes) #prints the available emotions 

        learn = ConvLearner.pretrained(arch, data, precompute=True) #Uses pretrianed in first case
        print('loading requirements......')
        print('This has been made by shaaran alias devshaaran, if you are using this code anywhere for research or educational purposes, please give reference.ENJOY!')
        learn.precompute=False #precomputation is made false for deeper recognition
        #learn.fit(1e-1, 1)
        learn.fit(1e-1, 3, cycle_len=1) #model is fit 
        learn.load('224_all')
        print('loading done !')
        
        #progress bar for all emotions *Incomplete*
        bar_happy = progressbar.ProgressBar(maxval=1,widgets=[progressbar.Bar('=', '[', ']'), 'happy', progressbar.Percentage()]) 
        bar_neutral = progressbar.ProgressBar(maxval=1, widgets=[progressbar.Bar('=', '[', ']'), 'neutral',progressbar.Percentage()])
        bar_sad = progressbar.ProgressBar(maxval=1, widgets=[progressbar.Bar('=', '[', ']'), 'sad',progressbar.Percentage()])
        bar_surprise = progressbar.ProgressBar(maxval=1, widgets=[progressbar.Bar('=', '[', ']'), 'surprise',progressbar.Percentage()])
        bar_happy.start()
        bar_neutral.start()
        bar_sad.start()
        bar_surprise.start()


# Initialize some variablesface_locations = []

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face detection processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame, model="cnn")
            counts = 0
            counts += 1

            # Display the results
            for top, right, bottom, left in face_locations:
                # Scale back up face locanp.exp(preds[0][3])*100tions since the frame we detected in was scaled to 1/4 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                lower_red = np.array([0,0,253])
                upper_red = np.array([0,0,255])

                # Extract the region of the image that contains the face
                face_image = frame[top:bottom, left:right]
                mask = cv2.inRange(face_image, lower_red, upper_red)
                res = cv2.bitwise_and(face_image, face_image, mask=mask)
                cv2.imshow('vid', face_image)
                cv2.imshow('res', res)
                count += 1
                cv2.imwrite('0.jpg',res)
                #cv2.imwrite((output_loc + '\\' + str(count)+ str(counts) + '.jpg'), res)

                try:

                    # learn = ConvLearner.pretrained(arch, data, precompute=True)
                    trn_tfms, val_tfms = tfms_from_model(arch, sz)
                    im = val_tfms(open_image('0.jpg'))
                    learn.precompute = False
                    preds = learn.predict_array(im[None])
                    
                    #Use below only for debuggng !
                    #print(preds)
                    #print(np.exp(preds)[0][0])
                    #qprint(data.classes[np.argmax(preds)])

                    #updating the percentages

                    bar_happy.update(np.exp(preds[0][0]))
                    bar_sad.update(np.exp(preds[0][2]))
                    bar_neutral.update(np.exp(preds[0][1]))
                    bar_surprise.update(np.exp(preds[0][3]))

                    #put text on video
                    cv2.putText(frame,'happy : ' + str(int(np.exp(preds[0][0])*100)) + '%' ,(top-40,left-30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0), 1)
                    cv2.putText(frame,'neutral : ' + str(int(np.exp(preds[0][1])*100)) + '%', (top-40, left), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
                    cv2.putText(frame, 'sad : ' + str(int(np.exp(preds[0][2])*100)) + '%', (top-40 , left+30 ), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
                    cv2.putText(frame, 'surprise : ' + str(int(np.exp(preds[0][3])*100)) + '%', (top-40, left + 60), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)

                except Exception as e:
                    print(e)

            cv2.imshow('Video', frame) #shows image 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                bar_surprise.finish()
                bar_neutral.finish()
                bar_sad.finish()
                bar_happy.finish()
                break
        video_capture.release()
        cv2.destroyAllWindows()


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):

    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def dlibfound():

    detector = dlib.get_frontal_face_detector()
    predict_path = 'D:\Python_mytext\cv2_notes\shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predict_path)
    video_capture = cv2.VideoCapture(0)

    while True:
        
        ret, image = video_capture.read()
        #image = cv2.imread(args["image"])
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        
        for (i, rect) in enumerate(rects):

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped_image = image[x:x+w,y:y+h]
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv2.imshow('cropped', cropped_image)
            cv2.imwrite('0.jpg',cropped_image)
                
        cv2.imshow("Output", image) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video_capture.release()

gitsearch()


gitsearch()
