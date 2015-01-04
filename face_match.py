import os
import sys
import pyaudio
import wave
import cv2
import numpy as np
import time

import random
rand = random.Random()

chunk = 1024

soundpath = "/home/tato/src/py/facerec/sounds/master/sound.wav"

def play_sound():
    f = wave.open(soundpath,"rb")
    #instantiate PyAudio
    p = pyaudio.PyAudio()
    #open stream
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)
    #read data
    data = f.readframes(chunk)

    #paly stream
    while data != '':
        stream.write(data)
        data = f.readframes(chunk)

    #stop stream
    stream.close()
    #close PyAudio
    p.terminate()

# added a names list(z)
def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.
    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes
    Returns:
        A list [X,y,z]
            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
            z: A list of person-names, indexed by label
    """
    c = 0
    X,y,z = [], [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                    try:
                        im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                        if (len(im)==0):dd
                            continue # not an image
                        # resize to given size (if given)
                        if (sz is not None):
                            im = cv2.resize(im, sz)
                        X.append(np.asarray(im, dtype=np.uint8))
                        y.append(c)
                    except IOError, (errno, strerror):
                        print "I/O error({0}): {1}".format(errno, strerror)
                    except:
                        print "Unexpected error:", sys.exc_info()[0]
                        raise
            c = c+1
            z.append(subdirname)
    return [X,y,z]



# reload the images & retrain the model.
# note, that lbp would give you the possibility of
# just updating the model with additonal data instead.
def retrain( imgpath, model,sz ) :
    # read in the image data. This must be a valid path!
    X,y,names = read_images(imgpath,sz)
    if len(X) == 0:
        print "image path empty", imgpath
        return [[],[],[]]
    # Learn the model. Remember our functio                    try:n returns Python lists,
    # so we use np.asarray to turn them into NumPy lists to make
    # the OpenCV wrapper happy:
    # Also convert labels to 32bit integers. This is a workaround for 64bit machines,
    model.train(np.asarray(X), np.asarray(y, dtype=np.int32))
    return [X,y,names]



if __name__ == "__main__":
    # You'll need the path to your image folder, also we need to find
    # a haar/lbpcascade for detecting faces, e.g. opencv\data\haarcascades\haarcascade_frontalface_alt2.xml
    if len(sys.argv) < 3:
        print "USAGE: facerec_online.py </path/to/images> <path/to/cascadefile>"
        sys.exit()

    print "  press 'esc' to quit"
    print "  press 'a' to append a new face to the database"
    print "      (you'll be prompted for a name on the console)"
    print "  press 't' to retrain the model (if you appended faces there)"

    # create the img folder, if nessecary
    imgdir = sys.argv[1]
    try:
        os.mkdir(imgdir)
    except:
        pass # dir already existed

    # default face size, all faces in the db need to be the same.
    face_size=(90,90)

    # open the webcam
    cam = cv2.VideoCapture(0)
    if ( not cam.isOpened() ):
         print "no cam!"
         sys.exit()
    print "cam: ok."

    # load the cascadefile:
    cascade = cv2.CascadeClassifier(sys.argv[2])
    if ( cascade.empty() ):
         print "no cascade!"
         sys.exit()
    print "cascade:",sys.argv[2]

    # Create the model. We are going to use the default
    # parameters for this simple example, please read the documentation
    #model = cv2.createEigenFaceRecognizer()
    #model = cv2.createFisherFaceRecognizer()
    model = cv2.createLBPHFaceRecognizer()

    # train it from faces in the imgdir:
    images,labels,names = retrain(imgdir,model,face_size)
    print "trained:",len(images),"images",len(names),"persons"
    master = 1
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # try to detect a face in the img:
        rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

        # roi will keep the cropped face image ( if there was one )
        roi = None
        for x, y, w, h in rects:
            # crop & resize it
            roi = cv2.resize( gray[y:y+h, x:x+h], face_size )
            # give some visual feedback for the cascade detection
            cv2.rectangle(img, (x,y),(x+w,y+h), (255, 0, 0))
            if len(images)>0:
                # model.predict is going to return the predicted label and
                # the associated confidence:
                [p_label, p_confidence] = model.predict(np.asarray(roi))
                name = "unknown"
                if p_label != -1 : name = names[p_label]
                cv2.putText( img, "%s %.2f" % (name, p_confidence),(x+10,y+20), cv2.FONT_HERSHEY_PLAIN,1.3, (0,0,255))
                if name != "master":
                    master = 1
                else:
                        if name == "master" and master == 1:
                            play_sound()
                            master = 0

            break # use only 1st detected

        cv2.imshow('facedetect', img)

        k = cv2.waitKey(5) & 0xFF

        # bailout on 'esc'
        if k == 27: break

        # 'a' pressed, add person to the database
        if (k == 97) and (roi!=None):
            print "please input the name: "
            name = sys.stdin.readline().strip('\r').strip('\n')
            # make a folder for that person:
            dirname = os.path.join(imgdir,name)
            try:
                os.mkdir(dirname)
            except:
                pass # dir already existed
            # save image
            path=os.path.join(dirname,"%d.png" %(rand.uniform(0,10000)))
            print "added:",path
            cv2.imwrite(path,roi)

        # if enough new data was collected, retrain the model
        if (k == 116): # 't' pressed
            images,labels,names = retrain(imgdir,model,face_size)
            print "trained:",len(images),"images",len(names),"persons"
