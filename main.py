
import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
import requests
from datetime import datetime
from collections import deque
import onnxruntime as ort
import time

from dotenv import load_dotenv
from pathlib import Path
import cloudinary

import firebase_admin
from firebase_admin import credentials, db


# ================= PROJECT CONFIGURATION =================

projectPath = Path(__file__).resolve().parent

# Load local environment variables.
load_dotenv(projectPath / ".env")

# Firebase configuration is read from .env.
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")

if not FIREBASE_DATABASE_URL:
    raise RuntimeError(
        "FIREBASE_DATABASE_URL is missing from your .env file."
    )

# Keep this credentials file local and out of GitHub.
firebase_key_path = projectPath / "serviceAccountKey.json"

if not firebase_key_path.exists():
    raise FileNotFoundError(
        "Firebase credentials not found. "
        "Place serviceAccountKey.json in the project folder."
    )

cred = credentials.Certificate(str(firebase_key_path))

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": FIREBASE_DATABASE_URL
    })

# Cloudinary configuration.
# API secrets must remain in the local .env file.
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)


# ================= FILE PATHS =================

resourcesPath = projectPath / "Resources"
modelsPath = projectPath / "models"

backgroundPath = resourcesPath / "background.png"
folderModePath = resourcesPath / "Modes"

encodeFilePath = projectPath / "EncodeFile.p"
antiSpoofModelPath = modelsPath / "best_model_quantized.onnx"


# ================= WEBCAM =================

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread(str(backgroundPath))

if imgBackground is None:
    raise FileNotFoundError(
        f"Background image not found: {backgroundPath}"
    )


# ================= ANTI-SPOOFING MODEL =================

if not antiSpoofModelPath.exists():
    raise FileNotFoundError(
        f"Anti-spoofing model not found: {antiSpoofModelPath}"
    )

anti_spoof_session = ort.InferenceSession(
    str(antiSpoofModelPath),
    providers=["CPUExecutionProvider"]
)

anti_spoof_input = anti_spoof_session.get_inputs()[0].name

# Require 4 live-face predictions out of 5 checks.
liveness_history = deque(maxlen=5)

# Initial logit-difference threshold.
LIVENESS_THRESHOLD = 0.5

print("Anti-spoofing model loaded successfully!")


# ================= FACE RECOGNITION STATE =================

processing = False
FACE_MATCH_THRESHOLD = 0.40

attendance_authorized = False

recognized_id_history = deque(maxlen=5)

modeType = 0
counter = 0
id = -1
imgStudent = []
frame_count = 0
faceCurFrame = []
encodeCurFrame = []
message = ""
message_until = 0


# ================= FACE ANTI-SPOOFING =================

def check_live_face(img, face_location):
    top, right, bottom, left = face_location

    # Face locations are based on the quarter-size frame.
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    height, width = img.shape[:2]

    # Expand the face crop slightly.
    face_w = right - left
    face_h = bottom - top
    margin = int(max(face_w, face_h) * 0.35)

    x1 = max(0, left - margin)
    y1 = max(0, top - margin)
    x2 = min(width, right + margin)
    y2 = min(height, bottom + margin)

    face_crop = img[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    size = 128
    h, w = face_crop.shape[:2]
    scale = size / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    face_crop = cv2.resize(face_crop, (new_w, new_h))

    pad_w = size - new_w
    pad_h = size - new_h

    face_crop = cv2.copyMakeBorder(
        face_crop,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_w // 2,
        pad_w - pad_w // 2,
        cv2.BORDER_REFLECT_101
    )

    tensor = (
        face_crop.transpose(2, 0, 1).astype(np.float32) / 255.0
    )
    tensor = np.expand_dims(tensor, axis=0)

    output = anti_spoof_session.run(
        None,
        {anti_spoof_input: tensor}
    )

    real_score = float(output[0][0][0])
    spoof_score = float(output[0][0][1])

    return real_score - spoof_score


# ================= LOAD DASHBOARD MODES =================

modePathList = sorted(os.listdir(folderModePath))
imgModeList = []

# Ignore hidden files such as .DS_Store.
for path in modePathList:
    if path.startswith("."):
        continue

    image_path = folderModePath / path
    mode_image = cv2.imread(str(image_path))

    if mode_image is None:
        print("Failed to load image:", image_path)
    else:
        imgModeList.append(mode_image)

if not imgModeList:
    raise RuntimeError(
        f"No dashboard mode images found in {folderModePath}"
    )


# ================= LOAD FACE ENCODINGS =================

print("Loading Encode File...")

if not encodeFilePath.exists():
    raise FileNotFoundError(
        "EncodeFile.p not found. "
        "Generate the face encodings before running the application."
    )

with open(encodeFilePath, "rb") as file:
    encodeListKnownWithIds = pickle.load(file)

encodeListKnown, studentIds = encodeListKnownWithIds

print("Encode File Loaded")


# ================= MAIN ATTENDANCE LOOP =================

try:
    while True:

        success, img = cap.read()

        if not success or img is None:
            continue

        # Reduce image size to lower computation.
        imgS = cv2.resize(
            img, (0, 0), None, 0.25, 0.25
        )

        imgRGB = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect faces every 2 frames.
        frame_count += 1

        if frame_count % 2 == 0 or processing:
            faceCurFrame = face_recognition.face_locations(imgRGB)
            encodeCurFrame = face_recognition.face_encodings(
                imgRGB, faceCurFrame
            )
        else:
            faceCurFrame = []
            encodeCurFrame = []

        imgBackground = cv2.imread(str(backgroundPath))

        if imgBackground is None:
            raise RuntimeError("Unable to load dashboard background.")

        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        # Block attendance if multiple faces are detected.
        if faceCurFrame and not processing and len(faceCurFrame) > 1:
            print("MULTIPLE FACES DETECTED - Attendance blocked")

            recognized_id_history.clear()
            liveness_history.clear()
            attendance_authorized = False

            message = "Only one face allowed"
            message_until = time.time() + 0.3

        # Process a single detected face.
        if faceCurFrame and not processing and len(faceCurFrame) == 1:

            for encodeFace, faceLoc in zip(
                encodeCurFrame, faceCurFrame
            ):

                faceDis = face_recognition.face_distance(
                    encodeListKnown, encodeFace
                )

                if len(faceDis) == 0:
                    continue

                matchIndex = int(np.argmin(faceDis))
                best_distance = float(faceDis[matchIndex])

                # Reject uncertain face matches.
                if best_distance > FACE_MATCH_THRESHOLD:
                    recognized_id_history.clear()
                    continue

                candidate_id = studentIds[matchIndex]

                # Require consistent identity across 5 checks.
                recognized_id_history.append(candidate_id)

                if len(recognized_id_history) < 5:
                    message = "Verifying identity..."
                    message_until = time.time() + 0.3
                    continue

                if len(set(recognized_id_history)) != 1:
                    recognized_id_history.clear()
                    continue

                # Check liveness before authorizing attendance.
                live_score = check_live_face(img, faceLoc)

                if live_score is None:
                    liveness_history.clear()

                    message = "Cannot verify face"
                    message_until = time.time() + 0.3
                    continue

                liveness_history.append(
                    live_score >= LIVENESS_THRESHOLD
                )

                # Require 4 positive results out of 5 checks.
                if len(liveness_history) < 5:
                    message = "Checking face liveness..."
                    message_until = time.time() + 0.3
                    continue

                if sum(liveness_history) < 4:
                    print(
                        "SPOOF REJECTED - Attendance not allowed"
                    )

                    recognized_id_history.clear()
                    liveness_history.clear()

                    message = "Spoof detected - Attendance blocked"
                    message_until = time.time() + 0.3
                    continue

                liveness_history.clear()

                # Authorize attendance only after verification.
                attendance_authorized = True

                y1, x2, y2, x1 = faceLoc

                y1, x2, y2, x1 = (
                    y1 * 4,
                    x2 * 4,
                    y2 * 4,
                    x1 * 4
                )

                bbox = (
                    55 + x1,
                    162 + y1,
                    x2 - x1,
                    y2 - y1
                )

                imgBackground = cvzone.cornerRect(
                    imgBackground, bbox, rt=0
                )

                id = candidate_id
                processing = True
                recognized_id_history.clear()

                if counter == 0:
                    cvzone.putTextRect(
                        imgBackground,
                        "Loading",
                        (275, 400)
                    )

                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)

                    counter = 1
                    modeType = 1

                break

        # Update attendance only after successful verification.
        if counter != 0 and attendance_authorized:

            if counter == 1:

                studentInfo = db.reference(
                    f"Students/{id}"
                ).get()

                if not studentInfo:
                    print("Student record not found:", id)

                    counter = 0
                    modeType = 0
                    processing = False
                    attendance_authorized = False
                    continue

                print(studentInfo)

                # Fetch the registered student's image.
                image_url = studentInfo.get("Image URL")

                if not image_url:
                    print("Student image URL is missing.")
                    imgStudent = []
                else:
                    try:
                        response = requests.get(
                            image_url, timeout=10
                        )
                        response.raise_for_status()

                        array = np.frombuffer(
                            response.content, np.uint8
                        )
                        imgStudent = cv2.imdecode(
                            array, cv2.IMREAD_COLOR
                        )

                    except requests.RequestException as e:
                        print("Student image download failed:", repr(e))
                        imgStudent = []

                # Update attendance in Firebase.
                try:
                    last_attendance = studentInfo.get(
                        "Last Attendance Time"
                    )

                    if last_attendance:
                        datetimeObject = datetime.strptime(
                            last_attendance,
                            "%Y-%m-%d %H:%M:%S"
                        )

                        secondsElapsed = (
                            datetime.now() - datetimeObject
                        ).total_seconds()
                    else:
                        secondsElapsed = 31

                    print("Student ID:", id)
                    print(
                        "Previous attendance:",
                        studentInfo.get("Total Attendance")
                    )
                    print("Seconds elapsed:", secondsElapsed)

                    if secondsElapsed > 30:
                        new_attendance = (
                            int(studentInfo.get("Total Attendance", 0))
                            + 1
                        )

                        ref = db.reference(f"Students/{id}")
                        ref.update({
                            "Total Attendance": new_attendance,
                            "Last Attendance Time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        })

                        print("Attendance updated successfully!")
                        print("New attendance count:", new_attendance)

                        studentInfo["Total Attendance"] = new_attendance

                    else:
                        print(
                            "Attendance not updated: "
                            "30-second restriction."
                        )

                        modeType = 3
                        counter = 0
                        processing = False
                        attendance_authorized = False

                        imgBackground[
                            44:44 + 633, 808:808 + 414
                        ] = imgModeList[modeType]

                except Exception as e:
                    print("Attendance update failed:", repr(e))

            if modeType != 3:

                if 10 < counter < 20:
                    modeType = 2

                imgBackground[
                    44:44 + 633, 808:808 + 414
                ] = imgModeList[modeType]

                if counter <= 10:

                    cv2.putText(
                        imgBackground,
                        str(studentInfo["Total Attendance"]),
                        (861, 125),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        1
                    )

                    cv2.putText(
                        imgBackground,
                        str(studentInfo["Branch"]),
                        (1006, 550),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                    cv2.putText(
                        imgBackground,
                        str(id),
                        (1006, 493),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                    cv2.putText(
                        imgBackground,
                        str(studentInfo["Standing"]),
                        (910, 625),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (100, 100, 100),
                        1
                    )

                    cv2.putText(
                        imgBackground,
                        str(studentInfo["Year"]),
                        (1025, 625),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (100, 100, 100),
                        1
                    )

                    cv2.putText(
                        imgBackground,
                        str(studentInfo["Starting Year"]),
                        (1125, 625),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (100, 100, 100),
                        1
                    )

                    # Center the student's name.
                    (w, h), _ = cv2.getTextSize(
                        studentInfo["Name"],
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        1
                    )

                    offset = (414 - w) // 2

                    cv2.putText(
                        imgBackground,
                        str(studentInfo["Name"]),
                        (808 + offset, 445),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (50, 50, 50),
                        1
                    )

                    # Display the student's image if available.
                    if (
                        isinstance(imgStudent, np.ndarray)
                        and imgStudent.size > 0
                    ):
                        imgBackground[
                            175:175 + 216, 909:909 + 216
                        ] = cv2.resize(imgStudent, (216, 216))

                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    processing = False
                    attendance_authorized = False
                    studentInfo = []
                    imgStudent = []

                    imgBackground[
                        44:44 + 633, 808:808 + 414
                    ] = imgModeList[modeType]

        else:
            if not processing:
                modeType = 0
                counter = 0

        # Display temporary status messages.
        if time.time() < message_until and message:
            cvzone.putTextRect(
                imgBackground,
                message,
                (50, 200),
                scale=0.85,
                thickness=1
            )

        cv2.imshow("Face Attendance", imgBackground)
        cv2.waitKey(1)

finally:
    cap.release()
    cv2.destroyAllWindows()
