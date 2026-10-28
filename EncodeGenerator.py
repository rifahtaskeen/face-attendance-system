
import cv2
import face_recognition
import pickle
import os

from dotenv import load_dotenv
from pathlib import Path
import cloudinary
import cloudinary.uploader

import firebase_admin
from firebase_admin import credentials, db


# ================= PROJECT CONFIGURATION =================

projectPath = Path(__file__).resolve().parent

# Load local environment variables.
load_dotenv(projectPath / ".env")

# Firebase configuration.
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")

if not FIREBASE_DATABASE_URL:
    raise RuntimeError(
        "FIREBASE_DATABASE_URL is missing from your .env file."
    )

firebaseKeyPath = projectPath / "serviceAccountKey.json"

if not firebaseKeyPath.exists():
    raise FileNotFoundError(
        "serviceAccountKey.json not found in the project folder."
    )

cred = credentials.Certificate(str(firebaseKeyPath))

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": FIREBASE_DATABASE_URL
    })


# ================= CLOUDINARY CONFIGURATION =================

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

config = cloudinary.config()

if not config.cloud_name or not config.api_key or not config.api_secret:
    raise RuntimeError(
        "Cloudinary credentials are missing from your .env file."
    )


# ================= FIREBASE DATABASE =================

ref = db.reference("Students")


# ================= LOAD STUDENT IMAGES =================

folderPath = projectPath / "Images"

if not folderPath.is_dir():
    raise FileNotFoundError(
        f"Student images folder not found: {folderPath}"
    )

PathList = sorted([
    file for file in os.listdir(folderPath)
    if file.lower().endswith((".png", ".jpg", ".jpeg"))
])

print(f"Found {len(PathList)} image files.")

imgList = []
studentIds = []

for path in PathList:
    filePath = folderPath / path
    img = cv2.imread(str(filePath))

    if img is None:
        print(f"Could not load image: {path}")
        continue

    student_id = os.path.splitext(path)[0]

    imgList.append(img)
    studentIds.append(student_id)

print(f"Loaded {len(imgList)} student images.")


# ================= GENERATE FACE ENCODINGS =================

def findEncodings(imagesList, ids):
    encodeList = []
    validIds = []

    for img, student_id in zip(imagesList, ids):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(imgRGB)

        if not encodings:
            print(f"No face detected for ID {student_id}")
            continue

        encodeList.append(encodings[0])
        validIds.append(student_id)

    return encodeList, validIds


print("Encoding started...")

encodeListKnown, validStudentIds = findEncodings(
    imgList,
    studentIds
)

encodeListKnownWithIds = [
    encodeListKnown,
    validStudentIds
]

print("Encoding complete.")
print(f"Successfully encoded {len(validStudentIds)} faces.")


# ================= UPLOAD IMAGES TO CLOUDINARY =================

for student_id in validStudentIds:
    filePath = folderPath / f"{student_id}"

    # Find the original image extension.
    matching_files = [
        name for name in PathList
        if os.path.splitext(name)[0] == student_id
    ]

    if not matching_files:
        continue

    filePath = folderPath / matching_files[0]

    try:
        result = cloudinary.uploader.upload(
            str(filePath),
            folder="face-attendance",
            public_id=student_id,
            overwrite=True
        )

        image_url = result.get("secure_url")

        if image_url:
            ref.child(student_id).update({
                "Image URL": image_url
            })
            print(f"Image uploaded for student ID: {student_id}")
        else:
            print(f"No image URL returned for ID: {student_id}")

    except Exception as e:
        print(
            f"Image upload failed for ID {student_id}: {e}"
        )


# ================= SAVE FACE ENCODINGS =================

encodeFilePath = projectPath / "EncodeFile.p"

with open(encodeFilePath, "wb") as file:
    pickle.dump(encodeListKnownWithIds, file)

print("EncodeFile.p saved successfully.")
