from deepface import DeepFace
import cv2

cap = cv2.VideoCapture(0)
print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        print("Full Result:", result)
        dominant_emotion = result[0]['dominant_emotion']

        # Display the emotion on the frame
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("Error:", e)

    # Show frame
    cv2.imshow("Webcam Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
