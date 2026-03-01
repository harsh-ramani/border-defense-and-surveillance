import cv2
import numpy as np
from ultralytics import YOLO
import time

def main():
    # Load the YOLOv8 model (downloads automatically if not present)
    model = YOLO("yolov8n.pt")
    
    # Initialize video capture. 0 typically refers to the default webcam.
    cap = cv2.VideoCapture(0)

    # Give the camera some time to warm up
    time.sleep(2)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Define the coordinates for the "virtual border"
    border_x = None

    # Define the classes we care about. In COCO dataset:
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
    target_classes = [0, 1, 2, 3, 5, 7]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Get frame dimensions if not already set
        if border_x is None:
            height, width, _ = frame.shape
            border_x = width // 2

        # Run YOLO inference on the frame
        results = model(frame, verbose=False)
        
        intrusion_detected = False

        # Extract bounding boxes, classes, and confidences
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get class ID
                cls_id = int(box.cls[0])
                
                # Filter for target classes (people and vehicles)
                if cls_id in target_classes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0])

                    # Check if the bounding box crosses the border
                    if x1 < border_x and x2 > border_x:
                        intrusion_detected = True
                        color = (0, 0, 255) # Red for intrusion
                    else:
                        color = (0, 255, 0) # Green for safe

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw center point
                    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                    
                    # Add label
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw the virtual border
        border_color = (0, 0, 255) if intrusion_detected else (0, 255, 255)
        cv2.line(frame, (border_x, 0), (border_x, height), border_color, 3)

        # Display alert text if intrusion is detected
        if intrusion_detected:
            cv2.putText(frame, "ALERT: INTRUSION DETECTED!", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show the processed frames
        cv2.imshow("Border Defense Surveillance System (YOLO)", frame)

        # Exit on 'q' key press
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
