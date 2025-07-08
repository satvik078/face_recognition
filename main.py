import cv2
import pickle
import face_recognition
import numpy as np

#url= "http://172.23.4.29:4747/video"
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

#load the encoding file
file=open('EncodeFile.p','rb')
encodeListKnownWithIds=pickle.load(file)
file.close()
encodeListKnown,studentIds=encodeListKnownWithIds
#print(studentIds)
print("Encode file loaded ")

modeType=0
counter=0
id=-1
imgstudent=[]


while True:
    success, img = cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)

    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)


    faceCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)


    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print("matches:",matches)
        print("faceDis",faceDis)

        matchIndex=np.argmin(faceDis)
        print("matchIndex",matchIndex)
       
        if matches[matchIndex]:
                # print("Known Face Detected")
                # print(studentIds[matchIndex])
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                img = cv2.rectangle(img,(x1,y1),(x2,y2),(50,50,255),1)
                cv2.putText(img,studentIds[matchIndex],(x1,y1),fontFace=2, fontScale=1,color=(255,255,0,255),thickness=2)

    cv2.imshow("frame", img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    