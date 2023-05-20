import cv2
import mediapipe as mp
import numpy as np
# time for fps calculation
import time
# UDP sockets
import socket
import struct

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

fps_start_time = 0
fps = 0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ("127.0.0.1", 4242)
buf = bytearray(8 * 6)

while cap.isOpened():
    # Flags to improve performance

    # Read Camera
    success, image = cap.read()
    # MediaPipe change = BGR (cv2) to RGB (mediapipe)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    # Return to cv2 format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert numpy to array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)
            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -10:
                text = "Left"
            elif y > 10:
                text = "Right"
            elif x < -10:
                text = "Down"
            elif x > 10:
                text = "Top"
            else:
                text = "Forward"

            data = [y*2.5, x*1.5, z*5, 0, 0, 0]
            struct.pack_into('dddddd', buf, 0, *data)
            sock.sendto(buf, address)

            cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # FPS

            fps_end_time = time.time()
            time_diff = fps_end_time - fps_start_time
            fps = 1/time_diff
            fps_start_time = fps_end_time

            fps_text = "FPS: {:.2f}".format(fps)
            cv2.putText(image, fps_text, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # XYZ display

            cv2.putText(image, f"X: {str(np.round(x, 2))}", (480, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f"Y: {str(np.round(y, 2))}", (480, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f"Z: {str(np.round(z, 2))}", (480, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display Nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

            cv2.line(image, p1, p2, (0, 0, 255), 2)

            cv2.imshow('Head Pose Estimation', image)

            # y -> left/right
            # x -> top/down
            # z -> distance
            print(f"X: {x} Y: {y} Z: {z}")

    # press esc
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
