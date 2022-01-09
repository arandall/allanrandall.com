import os
import dlib
import cv2
import numpy as np
import _pickle as cPickle

# Path to landmarks and face recognition model files
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_PATH = 'dlib_face_recognition_resnet_model_v1.dat'

# Initialize face detector, facial landmarks detector 
# and face recognizer
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(PREDICTOR_PATH)
faceRecognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# Root folder of the dataset
faceDatasetFolder = 'celeb_mini'
# Label -> Name Mapping file
labelMap = np.load("celeb_mapping.npy", allow_pickle=True).item()

# Each subfolder has images of a particular celeb
subfolders = os.listdir(faceDatasetFolder)

faceDescriptors = None
index = []

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

def enrol(imgPath, label):
    global faceDescriptors, index
    
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    faces = faceDetector(img)
    if len(faces) != 1:
        print("{} Face(s) found in {}. Skipping".format(len(faces), imgPath))
        return
    
    faceDescriptor = calcFaceDescriptor(img, faces[0])
    
    # Stack face descriptors (1x128) for each face in images, as rows
    if faceDescriptors is None:
        faceDescriptors = faceDescriptor
    else:
        faceDescriptors = np.concatenate((faceDescriptors, faceDescriptor), axis=0)
        
    index.append({
        "name": labelMap[label], 
        "label": label,
        "image": imgPath,
    })


for i, label in enumerate(subfolders):
    for imgName in os.listdir(os.path.join(faceDatasetFolder, label)):
        if not (imgName.lower().endswith('jpg') or imgName.lower().endswith('jpeg')):
            continue
        imgPath = os.path.join(faceDatasetFolder, label, imgName)
        enrol(imgPath, label)

# Write descriptors and index to disk
np.save('descriptors.npy', faceDescriptors)
# index has image paths in same order as descriptors in faceDescriptors
with open('index.pkl', 'wb') as f:
    cPickle.dump(index, f)

print("enrolment complete")