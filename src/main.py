import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Buffer for smoothing (gesture must be stable for a few frames)
gesture_buffer = deque(maxlen=10)

# Function to compute angle between three points
def calculate_angle(a, b, c):
    """Calculate the angle at point b formed by points a and c."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def recognize_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark

    # Finger joints for angle calculation
    finger_joints = {
        'thumb': (2, 3, 4),
        'index': (5, 6, 8),
        'middle': (9, 10, 12),
        'ring': (13, 14, 16),
        'pinky': (17, 18, 20)
    }

    # Calculate angles for all fingers
    angles = {}
    for finger, (a, b, c) in finger_joints.items():
        angles[finger] = calculate_angle(landmarks[a], landmarks[b], landmarks[c])

    # Define extended threshold
    extended_threshold = 160

    # Determine finger states
    is_thumb_extended = angles['thumb'] > 150
    is_index_extended = angles['index'] > extended_threshold
    is_middle_extended = angles['middle'] > extended_threshold
    is_ring_extended = angles['ring'] > extended_threshold
    is_pinky_extended = angles['pinky'] > extended_threshold

    # Gesture Detection

    # Thumbs Up (check first)
    if is_thumb_extended and not (is_index_extended or is_middle_extended or is_ring_extended or is_pinky_extended):
        return "Thumbs Up"

    # Fist: All fingers curled (thumb also curled)
    if (angles['index'] < 150 and angles['middle'] < 150 and angles['ring'] < 150 and angles['pinky'] < 150 and not is_thumb_extended):
        return "Fist"

    # Peace Sign
    if is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
        return "Peace Sign"

    # Open Palm
    if all([is_thumb_extended, is_index_extended, is_middle_extended, is_ring_extended, is_pinky_extended]):
        return "Open Palm"

    return "Unknown Gesture"


# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():  
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True

        # BGR to RGB
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        # print(results)

        gesture = 'No Hand'
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(45, 104, 46), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(50, 4, 50), thickness=2, circle_radius=2),
                )
                detected_gesture = recognize_gesture(hand_landmarks)
                gesture_buffer.append(detected_gesture)

        # Smoothing: Use the most frequent gesture in buffer
        if gesture_buffer:
            gesture = max(set(gesture_buffer), key=gesture_buffer.count)

        cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break 

cap.release()
cv2.destroyAllWindows()