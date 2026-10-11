import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
import dlib

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "Firebase_URL",
    'storageBucket': "your-project-id.appspot.com"
})

bucket = storage.bucket()

# Download facial landmark predictor
predictor_path = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(predictor_path):
    import urllib.request
    import bz2
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    try:
        print("📥 Downloading facial landmark predictor...")
        urllib.request.urlretrieve(url, predictor_path + ".bz2")
        with bz2.BZ2File(predictor_path + ".bz2") as fr, open(predictor_path, 'wb') as fw:
            fw.write(fr.read())
        os.remove(predictor_path + ".bz2")
        print("✅ Landmark predictor downloaded")
    except Exception as e:
        print(f"⚠️ Could not download predictor: {e}")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def detect_mask(face_landmarks):
    if not face_landmarks or len(face_landmarks) < 68:
        return False, "Unable to detect"
    mouth_points = face_landmarks[48:68]
    mouth_points = np.array(mouth_points)
    mouth_area = cv2.contourArea(mouth_points)
    if mouth_area < 100:
        return True, "Mask detected"
    else:
        return False, "No mask detected"

def detect_eyewear(face_landmarks, frame):
    if not face_landmarks or len(face_landmarks) < 68:
        return False, "Unable to detect"
    left_eye_points = face_landmarks[36:42]
    right_eye_points = face_landmarks[42:48]
    left_eye = np.array(left_eye_points)
    right_eye = np.array(right_eye_points)
    left_eye_bbox = (min(left_eye[:, 0]) - 10, min(left_eye[:, 1]) - 10,
                    max(left_eye[:, 0]) + 10, max(left_eye[:, 1]) + 10)
    right_eye_bbox = (min(right_eye[:, 0]) - 10, min(right_eye[:, 1]) - 10,
                     max(right_eye[:, 0]) + 10, max(right_eye[:, 1]) + 10)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_eye_region = gray[left_eye_bbox[1]:left_eye_bbox[3], 
                          left_eye_bbox[0]:left_eye_bbox[2]]
    right_eye_region = gray[right_eye_bbox[1]:right_eye_bbox[3], 
                           right_eye_bbox[0]:right_eye_bbox[2]]
    if left_eye_region.size > 0 and right_eye_region.size > 0:
        left_avg = np.mean(left_eye_region)
        right_avg = np.mean(right_eye_region)
        if left_avg < 50 and right_avg < 50:
            return True, "Sunglasses detected"
        if left_avg > 150 and right_avg > 150:
            return True, "Eyeglasses detected"
    return False, "No eyewear detected"

def detect_face_attributes(frame, face_location):
    top, right, bottom, left = face_location
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dlib_faces = detector(rgb_frame)
    for dlib_face in dlib_faces:
        landmarks = predictor(rgb_frame, dlib_face)
        landmark_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_points.append((x, y))
        mask_detected, mask_label = detect_mask(landmark_points)
        eyewear_detected, eyewear_label = detect_eyewear(landmark_points, frame)
        return {
            "mask": mask_detected,
            "mask_label": mask_label,
            "eyewear": eyewear_detected,
            "eyewear_label": eyewear_label
        }
    return {"mask": False, "mask_label": "Unknown", "eyewear": False, "eyewear_label": "Unknown"}

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print(f"✅ Loaded {len(encodeListKnown)} face encodings")

modeType = 0
counter = 0
id = -1
imgStudent = []
studentInfo = {}

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]
                
                # Detect mask and eyewear
                attributes = detect_face_attributes(img, (y1, x2, y2, x1))
                if attributes['mask']:
                    cvzone.putTextRect(imgBackground, f"😷 {attributes['mask_label']}", 
                                     (55 + x1, 162 + y1 - 30), scale=0.7, thickness=1, offset=5)
                if attributes['eyewear']:
                    cvzone.putTextRect(imgBackground, f"🕶️ {attributes['eyewear_label']}", 
                                     (55 + x1, 162 + y2 + 30), scale=0.7, thickness=1, offset=5)
                
                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

        if counter != 0:
            if counter == 1:
                studentInfo = db.reference(f'Students/{id}').get()
                try:
                    blob = bucket.get_blob(f'Images/{id}.png')
                    if blob:
                        array = np.frombuffer(blob.download_as_string(), np.uint8)
                        imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                except Exception as e:
                    print(f"⚠️ Could not load image: {e}")
                    imgStudent = np.zeros((216, 216, 3), dtype=np.uint8)
                if studentInfo:
                    datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                       "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                    if secondsElapsed > 30:
                        ref = db.reference(f'Students/{id}')
                        studentInfo['total_attendance'] += 1
                        ref.child('total_attendance').set(studentInfo['total_attendance'])
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:
                if 10 < counter < 20:
                    modeType = 2
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                if counter <= 10 and studentInfo:
                    cv2.putText(imgBackground, str(studentInfo.get('total_attendance', 0)), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo.get('major', '')), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo.get('standing', '')), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo.get('year', '')), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo.get('starting_year', '')), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    (w, h), _ = cv2.getTextSize(studentInfo.get('name', ''), cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackground, str(studentInfo.get('name', '')), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
                    if imgStudent is not None and imgStudent.size > 0:
                        imgBackground[175:175 + 216, 909:909 + 216] = cv2.resize(imgStudent, (216, 216))
                counter += 1
                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = {}
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance System", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
