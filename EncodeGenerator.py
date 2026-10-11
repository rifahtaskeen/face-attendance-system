import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://smart-attendance-system-745b6-default-rtdb.firebaseio.com/",
    'storageBucket': "your-project-id.appspot.com"
})
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(f"📁 Found {len(pathList)} images")

imgList = []
studentIds = []

for path in pathList:
    img = cv2.imread(os.path.join(folderPath, path))
    if img is not None:
        imgList.append(img)
        student_id = os.path.splitext(path)[0]
        studentIds.append(student_id)
        
        try:
            fileName = f'Images/{path}'
            bucket = storage.bucket()
            blob = bucket.blob(fileName)
            blob.upload_from_filename(os.path.join(folderPath, path))
            print(f"✅ Uploaded: {path}")
        except Exception as e:
            print(f"⚠️ Could not upload {path}: {e}")
    else:
        print(f"⚠️ Could not read: {path}")

def findEncodings(imagesList):
    encodeList = []
    for idx, img in enumerate(imagesList):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodes = face_recognition.face_encodings(img)
            if encodes:
                encodeList.append(encodes[0])
                print(f"✅ Encoded: {studentIds[idx]}")
            else:
                print(f"⚠️ No face found in {studentIds[idx]}")
        except Exception as e:
            print(f"❌ Error encoding {studentIds[idx]}: {e}")
    return encodeList

print("🔍 Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print(f"✅ Encoding Complete: {len(encodeListKnown)} faces encoded")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("💾 File Saved: EncodeFile.p")
