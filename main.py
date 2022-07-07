import cv2
import numpy as np
import  face_recognition
import os
from datetime import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


#fetch service account key JOSN file content
cred = credentials.Certificate('serviceAccountKey.json')

#Initialiaze the app with  the service account , granting adim privileges
firebase_admin.initialize_app(cred,{
    'databaseURL': 'https://attendance-f271e-default-rtdb.firebaseio.com'
})

usersIds=[]

my_names = {"pknhNh4ycpM9XoGYvkrVw9e7wji1": "Kareem Aboelatta",
            "9s58BuNR7Ncn17R1D80OuKMZ7Be2": "Abdalla reda",
            "rmVt27S5H8cbhQrlxjhiPsBnIOH3": "Marlene Marlow"}

#read data
handle =db.reference('users/').get()
for id in handle:
    usersIds.append(id)
print(usersIds)



path='ImageAttendance'

images=[]
classNames=[]

#to get image path
myList=os.listdir(path)
print(myList)

# to get image name and add it to our  list
for cl in myList:
    curImg= cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


#encode images
def findEncodings(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# add this attendance to database
def markAttendance(name, id):
    now = datetime.now()
    dtString = now.strftime('%d-%B-%Y')

    #updlaod day he attended in his data
    ref = db.reference('users/').child(id)
    ref.child('attendance').child(dtString).set(1)
    print(id)



#use our function to encode our images and but into one list which is "encodeListKnown"
encodeListKnown= findEncodings(images)
print('Encoding Complete')

# now open the camera
cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()

    #resize our pic for our performance speed  100*100  -> 25*25
    imgS= cv2.resize(img,(0,0),None,0.25,0.25)
    imgS= cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)


    #get face location to know where our face
    facesCurFrame=face_recognition.face_locations(imgS)
    #encode current face on camera
    encodesCurFrame= face_recognition.face_encodings(imgS,facesCurFrame)



    #compare this face with our employee faces
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex=np.argmin(faceDis)


        # check if it match
        # now we want to draw a borders for these faces
        if matches[matchIndex]:
            id=classNames[matchIndex]  # kareem for example we get name from this id
            print(my_names[id])     #100*100 ->  25*25    *4 =100*100
            y1,x2,y2,x1 =faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, ( x1,y1 ) , ( x2,y2 ) , (255,0,255) ,2)
            cv2.rectangle(img, ( x1,y2-35 ) , ( x2,y2 ) , (255,0,255) ,cv2.FILLED)
            cv2.putText(img, my_names[id] , (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(my_names[id],id)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)



