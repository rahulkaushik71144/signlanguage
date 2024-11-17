import cv2
import mediapipe as mp
import time

# Initialize Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

# Function to count fingers based on landmarks
def count_fingers(lm_list):
    fingers = []

    # Thumb
    if lm_list[4][1] > lm_list[3][1]:
        fingers.append(1)  # Thumb is extended
    else:
        fingers.append(0)  # Thumb is not extended

    # Other fingers (Index, Middle, Ring, Pinky)
    for i in range(8, 21, 4):
        if lm_list[i][2] < lm_list[i - 2][2]:  # If the finger tip is above the base joint
            fingers.append(1)  # Finger is extended
        else:
            fingers.append(0)  # Finger is not extended
    
    return fingers.count(1)  # Count how many fingers are extended

def main():
    cap = cv2.VideoCapture(0)  # Start webcam
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Convert the image to RGB for hand detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        # If hands are detected, process landmarks
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                # Count fingers using landmarks
                finger_count = count_fingers(lm_list)

                # Display the finger count on the image
                cv2.putText(img, f'Fingers: {finger_count}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                
                # Draw hand landmarks
                mp_draw.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS
        cv2.putText(img, f'FPS: {int(fps)}', (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Show the image with annotations
        cv2.imshow("Hand Gesture and Finger Count", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
