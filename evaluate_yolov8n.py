import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import datetime

GT_FOLDER = "eval/labels"
PRED_FOLDER = "eval/predictions"
SAVE_DIR = "runs/image_eval"

os.makedirs(SAVE_DIR, exist_ok=True)

def load_boxes(file_path):
    """Load bounding boxes from YOLO format file"""
    boxes = []
    try:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x, y, w, h = map(float, parts[:5])
                    boxes.append([cls, x, y, w, h])
    except:
        pass
    return boxes

def has_nlb(file_path):
    """Check if file contains any NLB detections"""
    return len(load_boxes(file_path)) > 0

def main():
    gt_files = sorted([f for f in os.listdir(GT_FOLDER) if f.endswith(".txt")])
    
    # Image-level classification metrics
    TP = FP = TN = FN = 0
    y_true = []
    y_pred = []

    for file in gt_files:
        gt_path = os.path.join(GT_FOLDER, file)
        pred_path = os.path.join(PRED_FOLDER, file)

        actual = has_nlb(gt_path)
        predicted = has_nlb(pred_path)

        y_true.append(1 if actual else 0)
        y_pred.append(1 if predicted else 0)

        if actual and predicted:
            TP += 1
        elif actual and not predicted:
            FN += 1
        elif not actual and predicted:
            FP += 1
        elif not actual and not predicted:
            TN += 1

    total = TP + FP + TN + FN
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = (TP + TN) / total if total > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Print metrics
    print("\n" + "="*60)
    print("NORTHERN CORN LEAF BLIGHT DETECTION EVALUATION")
    print("="*60)
    
    print("\nImage-Level Classification Metrics:")
    print(f"Total Images: {total}")
    print(f"True Positives: {TP}")
    print(f"False Negatives: {FN}")
    print(f"False Positives: {FP}")
    print(f"True Negatives: {TN}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")

    # Save results
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(SAVE_DIR, f"metrics_{now}.txt"), "w") as f:
        f.write("NORTHERN CORN LEAF BLIGHT DETECTION EVALUATION\n")
        f.write("="*60 + "\n\n")
        
        f.write("Image-Level Classification Metrics:\n")
        f.write(f"Total Images: {total}\n")
        f.write(f"True Positives: {TP}\n")
        f.write(f"False Negatives: {FN}\n")
        f.write(f"False Positives: {FP}\n")
        f.write(f"True Negatives: {TN}\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1_score:.4f}\n")

    # Create and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'NLB'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Northern Corn Leaf Blight Detection\nConfusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"confusion_matrix_{now}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {SAVE_DIR}")
    print(f"Confusion matrix saved as: confusion_matrix_{now}.png")

if __name__ == "__main__":
    main()
