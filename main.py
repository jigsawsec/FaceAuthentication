import cv2
import face_recognition

# Load the known faces
known_faces = []
known_faces_encodings = []
for name in ["person1","person2"]:
    image = cv2.imread("path/to/"+name+".jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    known_faces.append(image)
    known_faces_encodings.append(face_recognition.face_encodings(image)[0])

# Capture video from camera
cap = cv2.VideoCapture(0)

while True:
    # Get frame from camera
    ret, frame = cap.read()

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find the faces in the frame
    faces = face_recognition.face_locations(rgb_frame)

    # Find the encodings of the faces in the frame
    face_encodings = face_recognition.face_encodings(rgb_frame, faces)

    # Compare the encodings of the faces in the frame to the known faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            name = "Authenticated"
        else:
            name = "Unauthorized"
            
        # Draw a rectangle around the face
        (top, right, bottom, left) = faces[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name of the person
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Recognized Faces", frame)
        cv2.waitKey(0)

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
