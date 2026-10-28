# Smart Attendance System Using Face Recognition and Anti-Spoofing

## 1. Project Overview

The Smart Attendance System is a Python-based application that automates student attendance using real-time face recognition. It incorporates anti-spoofing verification to help prevent attendance through photographs or other spoofing attempts.

The system stores student details in Firebase Realtime Database and uses Cloudinary for cloud-based image storage, with local storage available as a backup for images.

## 2. Objectives

- Automate attendance recording using facial recognition.
- Reduce manual effort and errors in attendance management.
- Improve attendance reliability using anti-spoofing verification.
- Support face recognition under different conditions, including attempts involving masks or eyewear.
- Store student information and attendance records in real time.

## 3. Technologies Used

- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Face Recognition:** `face_recognition`
- **Anti-Spoofing:** ONNX Runtime with a pre-trained model
- **Database:** Firebase Realtime Database
- **Cloud Image Storage:** Cloudinary
- **Local Storage:** Used as a backup for images
- **User Interface:** OpenCV and CVZone

## 4. System Workflow

1. Capture video frames through a webcam.
2. Detect faces in the video stream.
3. Perform anti-spoofing verification.
4. Compare the detected face with the registered face encodings.
5. Identify the student when the recognition confidence meets the configured threshold.
6. Retrieve the student's details from Firebase Realtime Database.
7. Record attendance with the relevant date and time.
8. Store and retrieve student images using Cloudinary, with local image backup.

## 5. Storage Architecture

- **Firebase Realtime Database:** Stores student information and attendance records.
- **Cloudinary:** Stores student images in the cloud.
- **Local Storage:** Maintains a backup of student images.

Student information is added manually through the project's data-entry script. A separate admin panel is not currently implemented.

## 6. Project Structure

```text
face-attendance-system/
├── Resources/
│   └── Modes/
├── models/
│   └── best_model_quantized.onnx
├── main.py
├── AddDataToDatabase.py
├── EncodeGenerator.py
├── export_faces.py
├── face_landmarker.task
├── .gitignore
└── README.md
```

## 7. Prerequisites

- Python installed on your system
- A webcam
- A Firebase project with Realtime Database enabled
- A Cloudinary account
- The required Python packages

## 8. Installation and Setup

**Step 1: Clone the repository**

```bash
git clone https://github.com/rifahtaskeen/face-attendance-system.git
cd face-attendance-system
```

**Step 2: Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Step 3: Install the dependencies**

Install the Python packages required by the project. A `requirements.txt` file should be added to the repository to make this step reproducible.

**Step 4: Configure credentials**

Create a local `.env` file for the Firebase database URL and Cloudinary credentials. Place your Firebase service-account JSON file in the project directory.

Do not upload `.env` or `serviceAccountKey.json` to GitHub.

**Step 5: Prepare student data**

Add student information using `AddDataToDatabase.py` and place the corresponding student images in the expected local image directory.

**Step 6: Generate face encodings**

Run `EncodeGenerator.py` to generate the face encodings required for recognition.

**Step 7: Start the application**

```bash
python main.py
```

## 9. Security and Privacy

- Never commit Firebase service-account credentials or API secrets.
- Keep personal student images and biometric face encodings private.
- Obtain appropriate consent before collecting or processing facial data.
- Restrict database access using appropriate Firebase security rules.

## 10. Limitations and Future Enhancements

- Improve recognition accuracy under varying lighting and face angles.
- Strengthen anti-spoofing performance against different presentation attacks.
- Improve recognition when faces are partially covered by masks or eyewear.
- Add a dedicated administrative interface for managing student records and attendance reports.
- Evaluate the system under different real-world conditions.

## 11. Disclaimer

This project is developed for academic and demonstration purposes. Recognition accuracy and anti-spoofing effectiveness depend on environmental conditions, model performance, and system configuration.
