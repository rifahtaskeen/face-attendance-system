import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://your-project-id-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "STU001": {
        "name": "John Doe",
        "major": "Computer Science",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "G",
        "year": 1,
        "last_attendance_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "email": "john@example.com",
        "department": "CSE"
    },
    "STU002": {
        "name": "Jane Smith",
        "major": "Electronics",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "G",
        "year": 1,
        "last_attendance_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "email": "jane@example.com",
        "department": "ECE"
    },
    "STU003": {
        "name": "Bob Johnson",
        "major": "Mechanical",
        "starting_year": 2022,
        "total_attendance": 0,
        "standing": "G",
        "year": 2,
        "last_attendance_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "email": "bob@example.com",
        "department": "ME"
    }
}

print("📤 Adding students to Firebase...")
for key, value in data.items():
    ref.child(key).set(value)
    print(f"✅ Added: {key} - {value['name']}")

print("✅ All students added successfully!")
