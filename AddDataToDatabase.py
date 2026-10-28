
import os
from pathlib import Path

from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db


# ================= PROJECT CONFIGURATION =================

projectPath = Path(__file__).resolve().parent

load_dotenv(projectPath / ".env")

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

ref = db.reference("Students")


# ================= SAMPLE STUDENT DATA =================
# Use fictional records in the public repository.
# Add actual student information locally when needed.

data = {
    "100": {
        "Name": "Sample Student 100",
        "Branch": "Computer Science",
        "Starting Year": 2023,
        "Ending Year": 2027,
        "Total Attendance": 0,
        "Standing": "G",
        "Year": 4,
        "Last Attendance Time": "2000-01-01 00:00:00"
    },
    "101": {
        "Name": "Sample Student 101",
        "Branch": "Computer Science",
        "Starting Year": 2023,
        "Ending Year": 2027,
        "Total Attendance": 0,
        "Standing": "G",
        "Year": 4,
        "Last Attendance Time": "2000-01-01 00:00:00"
    }
}


# ================= OPTIONAL TEST STUDENT RECORDS =================

image_folder = projectPath / "Images"

# Generate records only for test images that exist locally.
for i in range(100, 200):
    image_path = image_folder / f"{i}.png"

    if not image_path.exists():
        continue

    student_id = str(i)

    data[student_id] = {
        "Name": f"Test Student {i}",
        "Branch": "Computer Science",
        "Starting Year": 2023,
        "Ending Year": 2027,
        "Total Attendance": 0,
        "Standing": "G",
        "Year": 4,
        "Last Attendance Time": "2000-01-01 00:00:00"
    }


# ================= UPLOAD TO FIREBASE =================

if not data:
    print("No student records to upload.")
else:
    for student_id, student_info in data.items():
        ref.child(student_id).update(student_info)

    print(f"Uploaded {len(data)} student records to Firebase.")
