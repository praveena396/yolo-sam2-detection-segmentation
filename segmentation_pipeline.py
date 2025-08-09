import cv2
import numpy as np
import requests
import base64
import os
import time
from typing import List, Tuple, Dict


class FaceHandSegmentationPipeline:
    """
    Automated pipeline for face and hand segmentation using Haar Cascade + SAM2 API
    """

    def __init__(self, replicate_api_token: str):
        self.replicate_token = replicate_api_token
        self.headers = {
            "Authorization": f"Token {replicate_api_token}",
            "Content-Type": "application/json"
        }
        self.load_detection_models()

    def load_detection_models(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("Using Haar Cascade for face detection")
        print("Hand detection will use color-based and contour detection")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [(x, y, w, h) for (x, y, w, h) in detected_faces]

    def detect_hands(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        hands = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                ar = w / h
                if 0.5 <= ar <= 2.0:
                    hands.append((x, y, w, h))
        return hands

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"

    def bbox_to_sam2_prompt(self, bbox: Tuple[int, int, int, int], label: str) -> Dict:
        x, y, w, h = bbox
        return {
            "type": "box",
            "data": [int(x), int(y), int(x + w), int(y + h)],
            "label": label
        }

    def call_sam2_api(self, image_base64: str, bounding_boxes: List[Dict]) -> Dict:
        url = "https://api.replicate.com/v1/predictions"
        data = {
            "version": "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
            "input": {
                "image": image_base64,
                "prompts": bounding_boxes
            }
        }
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 201:
            prediction_id = response.json()["id"]
            while True:
                result = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers=self.headers
                ).json()
                if result["status"] == "succeeded":
                    return result
                elif result["status"] == "failed":
                    raise Exception(f"API call failed: {result.get('error', 'Unknown error')}")
                time.sleep(1)
        else:
            print("SAM2 API call failed:", response.status_code)
            print("Response:", response.text)  # <--- Add this line
            raise Exception(f"API call failed {response.status_code}: {response.text}")

    def process_image(self, image_path: str, output_path: str = None) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        original = image.copy()
        faces = self.detect_faces(image)
        hands = self.detect_hands(image)
        print(f"Found {len(faces)} faces and {len(hands)} hands")
        prompts = [self.bbox_to_sam2_prompt(b, "face") for b in faces] \
                + [self.bbox_to_sam2_prompt(b, "hand") for b in hands]
        if not prompts:
            return original
        image_base64 = self.encode_image_to_base64(image)
        try:
            result = self.call_sam2_api(image_base64, prompts)
            masks = result["output"]["masks"]
            output = self.apply_masks(original, masks, faces, hands)
            if output_path:
                cv2.imwrite(output_path, output)
                print("Output saved:", output_path)
            return output
        except Exception as e:
            print("Error calling SAM2 API:", e)
            return self.draw_bounding_boxes(original, faces, hands)

    def apply_masks(self, image, masks, faces, hands):
        output = image.copy()
        face_color = (0, 255, 0)
        hand_color = (255, 0, 0)
        for i, mask in enumerate(masks):
            color = face_color if i < len(faces) else hand_color
            m = np.array(mask, dtype=np.uint8) * 255
            cm = np.zeros_like(image); cm[:] = color
            mask3d = np.stack([m]*3, axis=-1)/255.0
            output = output*(1-mask3d*0.5) + cm*(mask3d*0.5)
        return output.astype(np.uint8)

    def draw_bounding_boxes(self, image, faces, hands):
        out = image.copy()
        for x, y, w, h in faces:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for x, y, w, h in hands:
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(out, 'Hand', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return out
