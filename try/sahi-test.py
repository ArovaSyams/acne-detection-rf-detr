import os
import numpy as np
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.predict import get_sliced_prediction
from rfdetr import RFDETRLarge


# =========================================================
# STEP 1: SAHI DETECTOR WRAPPER FOR RF-DETR
# =========================================================
class RFDETRSahi(DetectionModel):
    def load_model(self):
        print(f"Loading RF-DETR model checkpoint from: {self.model_path}...")
        self.model = RFDETRLarge(pretrain_weights=self.model_path)
        print("RF-DETR model loaded successfully.")

    def perform_inference(self, image):
        """Perform detection inference on a sliced tile (NumPy array)."""
        self._original_predictions = self.model.predict(
            image, threshold=self.confidence_threshold
        )

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int | float]] | None = [[0, 0]],
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        """Translate RF-DETR detections into SAHI ObjectPrediction instances."""
        detections = self._original_predictions

        # Handle compatibility for shift and full shape dimensions
        if shift_amount_list is not None and isinstance(shift_amount_list[0], (int, float)):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], (int, float)):
            full_shape_list = [full_shape_list]

        shift_amount = shift_amount_list[0] if shift_amount_list else [0, 0]
        full_shape = full_shape_list[0] if full_shape_list else None

        boxes = detections.xyxy
        scores = detections.confidence
        class_ids = detections.class_id

        object_prediction_list = []
        for i in range(len(boxes)):
            x_min, y_min, x_max, y_max = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            object_prediction = ObjectPrediction(
                bbox=[float(x_min), float(y_min), float(x_max), float(y_max)],
                category_id=int(class_id),
                score=float(score),
                category_name="acne",
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            object_prediction_list.append(object_prediction)

        self._object_prediction_list_per_image = [object_prediction_list]


# =========================================================
# STEP 2: EXECUTE SAHI SLICED INFERENCE
# =========================================================
def main():
    model_path = os.path.abspath("best_rfdetr_acne.pth")
    image_path = os.path.abspath("test_wajah.jpg")

    print(f"Locating model checkpoint: {model_path}")
    print(f"Target test image: {image_path}")

    detection_model = RFDETRSahi(
        model_path=model_path,
        confidence_threshold=0.25,
        device="cuda:0"
    )

    print("\nStarting Slicing Aided Hyper Inference (SAHI)...")
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_type="NMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5
    )

    total_detections = len(result.object_prediction_list)
    print(f"\n[INFO] Total acne lesions detected: {total_detections}")

    if total_detections == 0:
        print("[!] No acne detected. Try adjusting confidence threshold or check input image.")
    else:
        output_name = "sahi_acne_detection_result"
        print(f"Exporting visualization with highlighted bounding boxes to '{output_name}.jpg'...")

        result.export_visuals(
            export_dir="./",
            file_name=output_name,
            rect_th=3,
            text_size=0.8
        )
        print("Visualization export complete! Output saved as 'sahi_acne_detection_result.png'.")


if __name__ == "__main__":
    main()