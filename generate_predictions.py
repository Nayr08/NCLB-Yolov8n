from ultralytics import YOLO
import os

# --- Configuration for Inference ---
# IMPORTANT: UPDATE THIS PATH to your actual best.pt file
MODEL_PATH = r'C:\Users\User\Desktop\COMPUTER ENGR\NLB_Research\best.pt'

# Path to the folder containing your testing images
TEST_IMAGES_PATH = r'C:\Users\User\Desktop\COMPUTER ENGR\NLB_Research\Phone'

# --- Output Settings ---
# Base directory where results will be saved.
# A subfolder will be created here (e.g., 'C:\...\model_predictions\my_phone_test\labels')
OUTPUT_BASE_DIR = r'C:\Users\User\Desktop\COMPUTER ENGR\NLB_Research\model_predictions'

# Name for the specific run's output folder
RUN_NAME = 'my_phone_test'

# Minimum confidence score for a prediction to be considered and saved
CONF_THRESHOLD = 0.25

# IoU threshold for Non-Maximum Suppression (NMS) to remove duplicate boxes
IOU_THRESHOLD = 0.45

def run_inference_and_save_predictions():
    """
    Loads the YOLOv8 model, runs inference on test images,
    and saves predictions to text files.
    """
    # 1. Load the model
    print(f"Loading YOLOv8 model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    # 2. Define where the prediction .txt files will be saved
    # This is where your model's bounding box predictions will appear as .txt files
    # E.g., C:\Users\User\Desktop\COMPUTER ENGR\NLB_Research\model_predictions\my_phone_test\labels
    prediction_labels_dir = os.path.join(OUTPUT_BASE_DIR, RUN_NAME, 'labels')
    os.makedirs(prediction_labels_dir, exist_ok=True) # Ensure the directory exists

    print(f"\nStarting inference on images in: {TEST_IMAGES_PATH}")
    print(f"Prediction .txt files will be saved to: {prediction_labels_dir}")
    print(f"Confidence Threshold: {CONF_THRESHOLD}, IoU Threshold (NMS): {IOU_THRESHOLD}")

    # 3. Run inference with specified settings
    # The key arguments here are save_txt=True and save_conf=True
    results = model.predict(
        source=TEST_IMAGES_PATH,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=False,        # Set to True if you also want to save images with drawn boxes
        save_txt=True,     # <--- ESSENTIAL: Saves bounding box coordinates to .txt files
        save_conf=True,    # <--- ESSENTIAL: Includes confidence scores in the .txt files
        project=OUTPUT_BASE_DIR, # Base directory for the output
        name=RUN_NAME,       # Subfolder for this run (e.g., 'my_phone_test')
        exist_ok=True        # Allow saving to an existing folder
    )

    print("\nInference complete!")
    print(f"You can now find your model's prediction text files in: {prediction_labels_dir}")

if __name__ == "__main__":
    run_inference_and_save_predictions()