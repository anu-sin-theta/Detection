import torch
import cv2
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection

# Set the device to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the YOLOS model and image processor onto the device
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small').to(device)
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")

# Open the video stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width of the captured frames to 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height of the captured frames to 480

# Initialize a set to keep track of detected objects
detected_objects = set()

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame and convert it to a PIL Image
    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run object detection inference on the image
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move the inputs tensor to the device
    outputs = model(**inputs)

    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # Print the number of detected objects
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    objects = set([model.config.id2label[label.item()] for label in results["labels"]])
    new_objects = objects.difference(detected_objects)
    detected_objects = detected_objects.union(objects)
    if len(new_objects) > 0:
        print(f"Detected {len(new_objects)} : {', '.join(new_objects)}")

    # Draw bounding boxes on the frame
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    for box in results["boxes"]:
        box = [int(i) for i in box.tolist()]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video stream and destroy the window
cap.release()
cv2.destroyAllWindows()