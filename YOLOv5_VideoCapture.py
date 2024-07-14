
import torch
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the YOLOv5 small model


# Function to perform object detection on a video
def detect_objects_in_video(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or 'mp4v' depending on your preference
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference
        results = model(frame)
        
        # Render results on the frame
        for img_result in results.render():
            out.write(cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))
        
        # Display the frame with detections
        cv2.imshow('YOLOv5 Object Detection', img_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to {output_path}")



# Example usage for video:
video_path = 'rosewatr.mov'
output_path = 'rosewatr_objs.mp4'
detect_objects_in_video(video_path, output_path)