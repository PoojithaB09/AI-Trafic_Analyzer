from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Open computer webcam (index 0)
# To use phone webcam in MacOS use index (1)
cam = cv2.VideoCapture(0)

# Set window name
window_name = "YOLOv8 Real-Time Detection"

while cam.isOpened():
    read, frame = cam.read()  # Read video frame and return if successfully read
    if not read:
        break  # Exit if video is unavailable

    # Run YOLOv8 on the frame and detect objects in real time with a confidence interval of 0.5
    # This confidence interval filters out weak detections
    results = model(frame, conf=0.5)

    # Count the number of persons detected (class '0' corresponds to 'person')
    persons = 0
    for result in results[0].boxes:
        # If the detected object is a person (class '0')
        if result.cls[0] == 0:
            persons += 1
            
    # Draw detections on the frame
    annotated_frame = results[0].plot()
    
    # Display the count of persons on the frame 
    cv2.putText(annotated_frame, f"Persons detected: {persons}", (10, 30), # corresponds to position
                cv2.FONT_ITALIC, 1, (255, 0, 255), 2, cv2.LINE_AA) #c2.LINE_AA ensures text is not aliased
    
    # Display instructions to quit
    cv2.putText(annotated_frame, f"To exit press 'q' on your keyboard", (10, 70), # corresponds to position
                cv2.FONT_ITALIC, 1, (0, 0, 255), 2, cv2.LINE_AA) #c2.LINE_AA ensures text is not aliased
    
    # Show the annotated video feed
    cv2.imshow(window_name, annotated_frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    
    
    

# Release resources
cam.release()
cv2.destroyAllWindows()

