import glob
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'

# Path to landmarks and face recognition model files
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_PATH = 'dlib_face_recognition_resnet_model_v1.dat'

# Initialize face detector, facial landmarks detector 
# and face recognizer
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(PREDICTOR_PATH)
faceRecognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# read image
testImages = glob.glob('test-images/*.jpg')

def calcFaceDescriptor(img, face):
    # Find facial landmarks for each detected face
    shape = shapePredictor(img, face)

    # Compute face descriptor using neural network defined in Dlib.
    # It is a 128D vector that describes the face in img identified by shape.
    faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)

    # Convert face descriptor from Dlib's format to list, then a NumPy array
    faceDescriptorList = [x for x in faceDescriptor]
    faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
    faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]
    return faceDescriptorNdarray

index = np.load('index.pkl', allow_pickle=True)
faceDescriptorsEnrolled = np.load('descriptors.npy')

for test in testImages:
    im = cv2.imread(test)
    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    #####################
    #  YOUR CODE HERE

    # load descriptors and index file generated during enrollment
    
    faces = faceDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    faceDescriptor = calcFaceDescriptor(im, faces[0])

    # Calculate Euclidean distances between face descriptor calculated on face dectected
    # in current frame with all the face descriptors we calculated while enrolling faces
    distances = np.linalg.norm(faceDescriptorsEnrolled - faceDescriptor, axis=1)
    # Calculate minimum distance and index of this face
    argmin = np.argmin(distances)  # index
   
    celeb_name = index[argmin]["name"]
    celeb_img  = cv2.imread(index[argmin]["image"])
    ####################
    
    plt.subplot(121)
    plt.imshow(imDlib)
    plt.title("test img")
    
    #TODO - display celeb image which looks like the test image instead of the black image. 
    plt.subplot(122)
    plt.imshow(celeb_img[:,:,::-1])
    plt.title("Celeb Look-Alike={}".format(celeb_name))
    plt.show()