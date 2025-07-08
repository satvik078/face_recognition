import cv2
import face_recognition
import pickle
import os

# Importing student images
folderPath = '/Users/satvikpandey/Downloads/project/face recognition/images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []
for path in pathList:
    # Only process image files
    if path.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join(folderPath, path))
        if img is not None:
            imgList.append(img)
            studentIds.append(os.path.splitext(path)[0])
        else:
            print(f"Warning: Could not load image {path}")
    else:
        # Remove .DS_Store or other unwanted files
        if path == '.DS_Store':
            try:
                os.remove(os.path.join(folderPath, path))
                print(f"Removed unwanted file: {path}")
            except Exception as e:
                print(f"Could not remove {path}: {e}")
print(studentIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                encodeList.append(encodings[0])
            else:
                print("Warning: No face found in an image, skipping.")
        except Exception as e:
            print(f"Error encoding image: {e}")
    return encodeList

print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print(encodeListKnown)
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")
