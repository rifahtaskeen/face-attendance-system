#!/usr/bin/env python3

from flask import Flask, render_template, jsonify, request, redirect, url_for
from firebase_admin import credentials, db
import firebase_admin
import base64
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os


app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your-secret-key-change-this'


try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://smart-attendance-system-745b6-default-rtdb.firebaseio.com/"
    })
    print("✅ Firebase connected")
except Exception as e:
    print(f"⚠️ Firebase error: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    message = None
    error = None

    if request.method == 'POST':

        student_id = request.form.get('student_id', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        major = request.form.get('major', '').strip()

        try:
            year = int(request.form.get('year', 1))
        except:
            year = 1

        try:
            starting_year = int(request.form.get('starting_year', 2024))
        except:
            starting_year = 2024

        face_image = request.form.get('face_image', '')

        # -----------------------------
        # VALIDATION
        # -----------------------------

        if not student_id or not name or not face_image:
            error = "Student ID, Name and Face Image are required!"

        else:

            try:

                # Remove data:image/... prefix
                if ',' in face_image:
                    face_image = face_image.split(',')[1]

                # Decode Base64 image
                image_bytes = base64.b64decode(face_image)

                # Convert image to OpenCV format
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    error = "Invalid image. Please capture the image again."

                else:

                    # Convert BGR → RGB
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # -----------------------------
                    # FACE DETECTION
                    # -----------------------------

                    face_locations = face_recognition.face_locations(rgb_img)

                    if len(face_locations) == 0:

                        error = "No face detected. Please capture your face clearly."

                    elif len(face_locations) > 1:

                        error = "Multiple faces detected. Please make sure only one person is visible."

                    else:

                        # -----------------------------
                        # FACE ENCODING
                        # -----------------------------

                        encodings = face_recognition.face_encodings(
                            rgb_img,
                            face_locations
                        )

                        if not encodings:

                            error = "Could not create face encoding. Please try again."

                        else:

                            face_encoding = encodings[0].tolist()

                            # -----------------------------
                            # COMPRESS IMAGE
                            # -----------------------------

                            success, compressed_image = cv2.imencode(
                                '.jpg',
                                img,
                                [cv2.IMWRITE_JPEG_QUALITY, 70]
                            )

                            if not success:

                                error = "Could not process the face image."

                            else:

                                image_base64 = base64.b64encode(
                                    compressed_image.tobytes()
                                ).decode('utf-8')

                                # -----------------------------
                                # SAVE TO FIREBASE
                                # -----------------------------

                                ref = db.reference(
                                    f'Students/{student_id}'
                                )

                                ref.set({

                                    'name': name,

                                    'email': email,

                                    'major': major,

                                    'year': year,

                                    'starting_year': starting_year,

                                    'total_attendance': 0,

                                    'standing': 'G',

                                    'last_attendance_time': '',

                                    'active': True,

                                    # Face recognition data
                                    'face_encoding': face_encoding,

                                    # Compressed face image
                                    'face_image': image_base64,

                                    'face_image_type': 'image/jpeg'

                                })

                                message = (
                                    f"Student {name} registered successfully!"
                                )

                                print(
                                    f"✅ Registered student: "
                                    f"{student_id} - {name}"
                                )

            except Exception as e:

                print(f"❌ Registration error: {e}")

                error = f"Error: {str(e)}"

    return render_template(
        'register.html',
        message=message,
        error=error
    )


@app.route('/attendance')
def attendance():

    try:

        ref = db.reference('attendance_records')
        records = ref.get() or {}

        attendance_list = []

        for key, record in records.items():

            attendance_list.append({

                'id': key,

                'student_id': record.get('student_id', ''),

                'name': record.get('name', ''),

                'time': record.get('time', ''),

                'date': record.get('date', '')

            })

        return render_template(
            'attendance.html',
            records=attendance_list
        )

    except:

        return render_template(
            'attendance.html',
            records=[]
        )


@app.route('/students')
def students():

    try:

        ref = db.reference('Students')

        students_data = ref.get() or {}

        student_list = []

        for student_id, data in students_data.items():

            student_list.append({

                'student_id': student_id,

                'name': data.get('name', ''),

                'major': data.get('major', ''),

                'year': data.get('year', ''),

                'attendance': data.get('total_attendance', 0),

                'last_attendance':
                    data.get(
                        'last_attendance_time',
                        'Never'
                    )[:16]

            })

        return render_template(
            'students.html',
            students=student_list
        )

    except:

        return render_template(
            'students.html',
            students=[]
        )


@app.route('/stats')
def stats():

    try:

        students_ref = db.reference('Students')
        students_data = students_ref.get() or {}

        attendance_ref = db.reference('attendance_records')
        records = attendance_ref.get() or {}

        # Current date
        today = datetime.now().strftime("%Y-%m-%d")

        # Count only today's attendance
        today_attendance = sum(
            1
            for record in records.values()
            if record.get('date') == today
        )

        stats = {

            'total_students':
                len(students_data),

            'today_attendance':
                today_attendance,

            'active_students':
                sum(
                    1
                    for student in students_data.values()
                    if student.get('active', True)
                ),

            'known_faces':
                sum(
                    1
                    for student in students_data.values()
                    if student.get('face_encoding')
                )

        }

        return render_template(
            'stats.html',
            stats=stats
        )

    except Exception as e:

        print(f"❌ Statistics error: {e}")

        return render_template(
            'stats.html',
            stats={
                'total_students': 0,
                'today_attendance': 0,
                'active_students': 0,
                'known_faces': 0
            }
        )


@app.route('/api/students')
def api_students():

    try:

        ref = db.reference('Students')

        return jsonify(
            ref.get() or {}
        )

    except:

        return jsonify({})


@app.route('/api/attendance')
def api_attendance():

    try:

        ref = db.reference('attendance_records')

        return jsonify(
            ref.get() or {}
        )

    except:

        return jsonify({})


@app.route('/api/recognize', methods=['POST'])
@app.route('/api/recognize', methods=['POST'])
def recognize_face():

    try:

        data = request.get_json()

        if not data or 'image' not in data:

            return jsonify({
                'success': False,
                'message': 'No image received'
            })

        face_image = data['image']

        # Remove Base64 prefix
        if ',' in face_image:
            face_image = face_image.split(',')[1]

        # Decode image
        image_bytes = base64.b64decode(face_image)

        nparr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return jsonify({
                'success': False,
                'message': 'Invalid image'
            })

        # ==========================================
        # FACE RECOGNITION
        # ==========================================

        rgb_img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        face_locations = face_recognition.face_locations(
            rgb_img,
            number_of_times_to_upsample=2,
            model="hog"
        )

        print(
            f"🔎 Face locations detected: "
            f"{len(face_locations)}"
        )

        if not face_locations:

            return jsonify({
                'success': True,
                'face_count': 0,
                'recognized': False,
                'message': 'No face detected'
            })

        # ==========================================
        # CREATE FACE ENCODINGS
        # ==========================================

        face_encodings = face_recognition.face_encodings(
            rgb_img,
            face_locations
        )

        # ==========================================
        # GET REGISTERED STUDENTS
        # ==========================================

        students_ref = db.reference('Students')

        students = students_ref.get() or {}

        known_encodings = []
        known_students = []

        for student_id, student in students.items():

            encoding = student.get('face_encoding')

            if encoding:

                known_encodings.append(
                    np.array(encoding)
                )

                known_students.append({
                    'student_id': student_id,
                    'name': student.get(
                        'name',
                        'Unknown'
                    )
                })

        if not known_encodings:

            return jsonify({
                'success': False,
                'face_count': len(face_locations),
                'recognized': False,
                'message': 'No registered face encodings found'
            })

        # ==========================================
        # FACE RECOGNITION
        # ==========================================

        recognized_name = None
        recognized_id = None
        best_distance = 1.0

        for face_encoding in face_encodings:

            distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_index = np.argmin(distances)

            distance = float(
                distances[best_index]
            )

            # Recognition threshold
            if (
                distance < 0.50
                and distance < best_distance
            ):

                best_distance = distance

                recognized_name = known_students[
                    best_index
                ]['name']

                recognized_id = known_students[
                    best_index
                ]['student_id']

        # ==========================================
        # NOT RECOGNIZED
        # ==========================================

        if not recognized_name:

            return jsonify({
                'success': True,
                'face_count': len(face_locations),
                'recognized': False,
                'message': 'Unknown face'
            })

        # ==========================================
        # RECOGNIZED
        # ==========================================

        now = datetime.now()

        today = now.strftime(
            "%Y-%m-%d"
        )

        current_time = now.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ==========================================
        # CHECK IF ALREADY MARKED TODAY
        # ==========================================

        attendance_ref = db.reference(
            'attendance_records'
        )

        existing_records = (
            attendance_ref.get() or {}
        )

        already_marked = False

        for record in existing_records.values():

            if (
                record.get('student_id')
                == recognized_id
                and
                record.get('date')
                == today
            ):

                already_marked = True

                break

        # ==========================================
        # MARK ATTENDANCE
        # ==========================================

        if not already_marked:

            new_record = attendance_ref.push()

            new_record.set({

                'student_id':
                    recognized_id,

                'name':
                    recognized_name,

                'date':
                    today,

                'time':
                    current_time,

                'face_distance':
                    round(
                        best_distance,
                        4
                    )

            })

            # Update student's attendance count
            student_ref = db.reference(
                f'Students/{recognized_id}'
            )

            student_data = (
                student_ref.get() or {}
            )

            current_attendance = int(
                student_data.get(
                    'total_attendance',
                    0
                )
            )

            student_ref.update({

                'total_attendance':
                    current_attendance + 1,

                'last_attendance_time':
                    current_time

            })

            print(
                f"✅ Attendance marked: "
                f"{recognized_name} "
                f"({recognized_id})"
            )

            attendance_status = "marked"

        else:

            print(
                f"ℹ️ Already marked today: "
                f"{recognized_name}"
            )

            attendance_status = "already_marked"

        # ==========================================
        # RESPONSE
        # ==========================================

        return jsonify({

            'success': True,

            'face_count':
                len(face_locations),

            'recognized':
                True,

            'student_id':
                recognized_id,

            'name':
                recognized_name,

            'distance':
                round(
                    best_distance,
                    4
                ),

            'attendance':
                attendance_status

        })

    except Exception as e:

        print(
            f"❌ Recognition error: {e}"
        )

        return jsonify({

            'success': False,

            'message': str(e)

        })


if __name__ == '__main__':

    print("=" * 60)
    print("🌐 SMART ATTENDANCE SYSTEM - WEB INTERFACE")
    print("=" * 60)
    print("📱 Access at: http://localhost:5000")
    print("📝 Register students at: http://localhost:5000/register")
    print("=" * 60)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
