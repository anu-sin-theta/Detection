from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width of the captured frames to 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height of the captured frames to 480
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # print results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()