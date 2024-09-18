from ultralytics import YOLO
import cv2
import os
import pandas as pd

def loadModel(model_pth):
    return YOLO(model_pth)

def loadImage(image_pth):
    return cv2.imread(image_pth)

def resizeImage(image, target_size=(1088, 1088)):
    image = cv2.resize(image, target_size)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def predict(model, image):
    results = model(image)
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    return boxes, scores

def calculate_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def show(image, boxes, scores, save_path=None):
    for box, score in zip(boxes, scores):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image, f'Confidence: {score:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def process_images(image_folder, model, run_directory):
    all_dacs_details = pd.DataFrame(columns=["Image Name", "Original Size", "Number of DACS", "Box Area", "Detection Probability"])
    
    for filename in os.listdir(image_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".tif")):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            original_size = image.shape[:2] 
            resized_image = cv2.resize(image, (1088, 1088))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            results = model(resized_image)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()

            number_of_dacs = len(boxes)
            for index, (box, score) in enumerate(zip(boxes, scores)):
                box_area = calculate_area(box)
                dacs_detail = {
                    "Image Name": filename if index == 0 else "",
                    "Original Size": f"{original_size[1]} x {original_size[0]}" if index == 0 else "",
                    "Number of DACS": number_of_dacs if index == 0 else None,
                    "Box Area": box_area,
                    "Detection Probability": score
                }
                all_dacs_details = pd.concat([all_dacs_details, pd.DataFrame([dacs_detail])], ignore_index=True)
            output_path = os.path.join(run_directory, filename)
            for box, score in zip(boxes, scores):
                cv2.rectangle(resized_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                label = f"{score:.2f}"
                cv2.putText(resized_image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(output_path, cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
    excel_path = os.path.join(run_directory, "Detailed_DACS_analysis.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        all_dacs_details.to_excel(writer, sheet_name='DACS Details', index=False)

def getNextRunDirectory(base_path="runs"):
    os.makedirs(base_path, exist_ok=True)
    previous_runs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    run_number = 1
    while f"run{run_number}" in previous_runs:
        run_number += 1
    new_run_dir = os.path.join(base_path, f"run{run_number}")
    os.makedirs(new_run_dir, exist_ok=True)
    return new_run_dir

def main(image_folder, model_pth):
    model = loadModel(model_pth)
    run_directory = getNextRunDirectory()
    process_images(image_folder, model, run_directory)

if __name__ == '__main__':
    image_folder = "/Users/zhangsen/Desktop/ye/test_image"
    model_pth = "/Users/zhangsen/Desktop/ye/model/model_full_2.pt"
    main(image_folder, model_pth)
