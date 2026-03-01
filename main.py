import cv2
import numpy as np
import time

def main():
    # Initialize video capture. 0 typically refers to the default webcam.
    # You can replace 0 with a video file path (e.g., 'border_video.mp4')
    cap = cv2.VideoCapture(0)

    # Give the camera some time to warm up
    time.sleep(2)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Create background subtractor for motion detection
    # Using MOG2 which handles shadows well
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Define the coordinates for the "virtual border"
    # We'll set a vertical line. Let's start with an arbitrary x-coordinate.
    # This will be updated once we get the frame dimensions.
    border_x = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Get frame dimensions if not already set
        if border_x is None:
            height, width, _ = frame.shape
            # Initialize border in the middle of the screen
            border_x = width // 2

        # Apply background subtraction
        fg_mask = back_sub.apply(frame)

        # Threshold to remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise (small dots)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate mask to merge adjacent blobs
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        # Find contours of moving objects
        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        intrusion_detected = False

        for contour in contours:
            # Ignore small contours (adjust this value based on distance and camera resolution)
            if cv2.contourArea(contour) < 2000:
                continue

            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the center point of the object for tracking
            center_x = x + w // 2
            center_y = y + h // 2

            # Check if the object crosses the border line
            # Logic: If the object's x coordinate + width crosses the border from left to right, 
            # or if its x coordinate crosses from right to left.
            # For simplicity, we trigger alert if any part of the bounding box intersects the border line.
            if x < border_x and (x + w) > border_x:
                intrusion_detected = True
            
            # Draw bounding box and center point
            color = (0, 0, 255) if intrusion_detected else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # Draw the virtual border
        border_color = (0, 0, 255) if intrusion_detected else (0, 255, 255)
        cv2.line(frame, (border_x, 0), (border_x, height), border_color, 3)

        # Display alert text if intrusion is detected
        if intrusion_detected:
            cv2.putText(frame, "ALERT: INTRUSION DETECTED!", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show the processed frames
        cv2.imshow("Border Defense Surveillance System", frame)
        # cv2.imshow("Foreground Mask", fg_mask) # Uncomment to see the motion mask for debugging

        # Exit on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
