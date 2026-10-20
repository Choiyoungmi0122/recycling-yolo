"""
방법 1 통합 추론기

    Input Frame (BGR)
    → YOLO26s 4-class (CAN / GLASS / PET / PLASTIC + bbox)
    → 주 객체 bbox 선택 (기본: 면적 최대)
    → bbox 10% padding + clamp → crop → RGB
    → 해당 재질의 EfficientNet-B0 하나만 실행
    → 최종 11-class

사용 예:

    predictor = Method1Predictor(device=1)
    result = predictor.predict(frame)   # 검출 없으면 result["detected"] == False
"""

from pathlib import Path
import re
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
from ultralytics import YOLO


# ============================================================
# 기본 경로
# ============================================================

DEFAULT_YOLO_MODEL = Path(
    r"E:\recycling_data\runs\recycling-yolo-comparison"
    r"\yolo26s_4class_common30_rerun-2\weights\best.pt"
)

DEFAULT_CNN_MODELS = {
    "can": Path(r"E:\recycling_data\runs\can-efficientnet-b0\best_can_efficientnet_b0.pth"),
    "glass": Path(r"E:\recycling_data\runs\glass-efficientnet-b0\best_glass_efficientnet_b0.pth"),
    "pet": Path(r"E:\recycling_data\runs\pet-efficientnet-b0\best_pet_efficientnet_b0.pth"),
    "plastic": Path(r"E:\recycling_data\runs\plastic-efficientnet-b0\best_plastic_efficientnet_b0.pth"),
}


# ============================================================
# 최종 11-class (클라이언트 요구 출력 체계 - 변경 금지)
# ============================================================

FINAL_CLASSES = [
    "can_normal",
    "can_dirty",
    "glass_normal",
    "glass_dirty",
    "glass_packaging",
    "pet_normal",
    "pet_dirty",
    "pet_packaging",
    "pet_dirty_packaging",
    "plastic_normal",
    "plastic_dirty",
]

FINAL_CLASS_TO_ID = {
    name: index for index, name in enumerate(FINAL_CLASSES)
}


# ============================================================
# CNN 전처리 (학습 시 val_transform 과 동일)
# ============================================================

CNN_INPUT_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# CNN dataset 생성 script(make_*_classifier_dataset.py)와 동일
CROP_PADDING_RATIO = 0.10


# ============================================================
# 주 객체 선택
# 나중에 ROI 중심 기준 등으로 바꿀 때 이 함수만 교체
# ============================================================

def select_largest_detection(detections):
    """
    detections: [{"material_id", "bbox": [x1, y1, x2, y2], "confidence"}, ...]
    반환: 면적이 가장 큰 detection 1개 또는 None
    """

    best = None
    best_area = -1.0

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        if area > best_area:
            best = det
            best_area = area

    return best


# ============================================================
# bbox → padding crop 좌표
# make_*_classifier_dataset.py 의 yolo_to_crop_box 와 동일 규칙
# (입력이 normalized cxcywh 가 아니라 pixel xyxy 인 점만 다름)
# ============================================================

def padded_crop_box(
    bbox,
    image_width,
    image_height,
    padding_ratio=CROP_PADDING_RATIO,
):

    x1, y1, x2, y2 = bbox

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    pad_x = bbox_width * padding_ratio
    pad_y = bbox_height * padding_ratio

    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(image_width, int(round(x2 + pad_x)))
    y2 = min(image_height, int(round(y2 + pad_y)))

    return x1, y1, x2, y2


# ============================================================
# checkpoint class_names → 11-class 이름
# 예: "3_pet_dirty_packaging" → "pet_dirty_packaging"
# ============================================================

_INDEX_PREFIX = re.compile(r"^\d+_")


def strip_index_prefix(class_name):
    return _INDEX_PREFIX.sub("", class_name)


# ============================================================
# EfficientNet-B0 로드
# ============================================================

def load_state_classifier(checkpoint_path, material, device):
    """
    반환: (model, final_ids, states)
        final_ids[cnn_index] = 최종 11-class id
        states[cnn_index]    = 재질 prefix 를 뺀 상태 이름 (예: "dirty_packaging")
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN checkpoint 없음 ({material}):\n{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    ckpt_material = checkpoint.get("material")

    if ckpt_material is not None and ckpt_material != material:
        raise ValueError(
            f"checkpoint material 불일치: 기대 {material}, "
            f"checkpoint {ckpt_material} ({checkpoint_path})"
        )

    class_names = list(checkpoint["class_names"])

    # ImageFolder 는 class_to_idx 순서로 output index 를 정함
    class_to_idx = checkpoint.get("class_to_idx")

    if class_to_idx is not None:
        for index, name in enumerate(class_names):
            if class_to_idx[name] != index:
                raise ValueError(
                    f"{material}: class_names 와 class_to_idx 순서 불일치 "
                    f"({name}: {class_to_idx[name]} != {index})"
                )

    final_ids = []
    states = []

    for name in class_names:

        final_name = strip_index_prefix(name)

        if final_name not in FINAL_CLASS_TO_ID:
            raise ValueError(
                f"{material}: checkpoint class '{name}' → '{final_name}' 이 "
                f"최종 11-class 에 없음"
            )

        prefix = f"{material}_"

        if not final_name.startswith(prefix):
            raise ValueError(
                f"{material}: checkpoint class '{name}' 가 다른 재질의 class 임"
            )

        final_ids.append(FINAL_CLASS_TO_ID[final_name])
        states.append(final_name[len(prefix):])

    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model, final_ids, states


# ============================================================
# 방법 1 통합 추론기
# ============================================================

class Method1Predictor:

    def __init__(
        self,
        yolo_model=DEFAULT_YOLO_MODEL,
        cnn_models=None,
        device=0,
        imgsz=640,
        conf=0.25,
        yolo_fp16=False,
        cnn_amp=True,
        select_detection=select_largest_detection,
    ):

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

        self.device = torch.device(f"cuda:{device}")
        self.device_index = device
        self.imgsz = imgsz
        self.conf = conf
        self.cnn_amp = cnn_amp
        self.select_detection = select_detection

        # Ultralytics 8.4: half 는 deprecated → quantize=16 (FP16) / None (FP32)
        self.yolo_quantize = 16 if yolo_fp16 else None

        # 첫 CNN 입력 tensor 정보를 한 번만 출력하기 위한 flag
        self.log_first_input = True

        torch.cuda.set_device(self.device)

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        yolo_model = Path(yolo_model)

        if not yolo_model.exists():
            raise FileNotFoundError(f"YOLO model 없음:\n{yolo_model}")

        self.yolo = YOLO(str(yolo_model))

        # YOLO class id → 재질 이름 (소문자 = CNN key)
        self.yolo_names = {
            int(class_id): name for class_id, name in self.yolo.names.items()
        }

        # ----------------------------------------------------
        # 재질별 CNN
        # ----------------------------------------------------

        cnn_models = cnn_models or DEFAULT_CNN_MODELS

        self.cnn = {}

        for material, path in cnn_models.items():
            self.cnn[material] = load_state_classifier(path, material, self.device)

        yolo_materials = {name.lower() for name in self.yolo_names.values()}

        if yolo_materials != set(self.cnn):
            raise ValueError(
                f"YOLO 재질 {sorted(yolo_materials)} 와 "
                f"CNN 재질 {sorted(self.cnn)} 이 일치하지 않음"
            )

        # 네 CNN 출력을 합치면 정확히 11-class 전체가 되어야 함
        covered = sorted(
            final_id
            for _, final_ids, _ in self.cnn.values()
            for final_id in final_ids
        )

        if covered != list(range(len(FINAL_CLASSES))):
            raise ValueError(
                f"CNN 출력이 최종 11-class 를 정확히 덮지 않음: {covered}"
            )

    # --------------------------------------------------------

    def _sync(self):
        torch.cuda.synchronize(self.device)

    def mapping_table(self):
        """(재질, cnn_index, state, final_id, final_class) 목록 - 검증/로그용"""

        rows = []

        for material, (_, final_ids, states) in self.cnn.items():
            for cnn_index, (final_id, state) in enumerate(zip(final_ids, states)):
                rows.append(
                    (material, cnn_index, state, final_id, FINAL_CLASSES[final_id])
                )

        return rows

    def device_report(self):
        """
        모델이 실제로 올라간 device / dtype
        YOLO 는 첫 predict 시점에 device 로 옮겨지므로 warmup() 이후에 호출
        """

        yolo_param = next(self.yolo.model.parameters())

        report = {
            "YOLO": f"{yolo_param.device} {yolo_param.dtype}",
        }

        for material, (model, _, _) in self.cnn.items():
            param = next(model.parameters())
            report[f"{material.upper()} CNN"] = (
                f"{param.device} {param.dtype} training={model.training}"
            )

        return report

    # --------------------------------------------------------

    def warmup(self, frame_shape=(1080, 1920, 3), iterations=3):
        """CUDA kernel / cudnn 초기화 비용이 latency 측정에 섞이지 않도록"""

        dummy = np.zeros(frame_shape, dtype=np.uint8)
        dummy_crop = torch.zeros(
            1, 3, CNN_INPUT_SIZE, CNN_INPUT_SIZE, device=self.device
        )

        for _ in range(iterations):

            self._detect(dummy)

            for model, _, _ in self.cnn.values():
                self._forward(model, dummy_crop)

        self._sync()

    # --------------------------------------------------------

    def _detect(self, frame):

        result = self.yolo.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device_index,
            quantize=self.yolo_quantize,
            verbose=False,
        )[0]

        detections = []

        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, class_id, confidence in zip(xyxy, classes, confidences):
            detections.append({
                "material_id": int(class_id),
                "bbox": [float(v) for v in box],
                "confidence": float(confidence),
            })

        return detections

    # --------------------------------------------------------


    def _preprocess(self, frame, bbox):
        """bbox padding crop → RGB → 224 tensor (GPU). crop 이 비면 (None, box)"""

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = padded_crop_box(bbox, width, height)

        if x2 <= x1 or y2 <= y1:
            return None, [x1, y1, x2, y2]

        crop_bgr = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # 학습과 동일하게 PIL 기반 Resize 사용
        tensor = CNN_TRANSFORM(Image.fromarray(crop_rgb))

        # pageable memory → GPU 복사는 동기식이라 별도 synchronize 불필요
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor, [x1, y1, x2, y2]

    def _forward(self, model, tensor):

        model.eval()

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=self.cnn_amp
        ):
            logits = model(tensor)

        return torch.softmax(logits.float(), dim=1)[0]

    # --------------------------------------------------------

    def predict(self, frame):
        """
        frame: OpenCV BGR (H, W, 3) uint8

        반환: 항상 dict
            detected=False → 검출 없음 (material~final_class 는 None, timing 만 기록)
            detected=True  → material / bbox / state / final_class ... + timing

        synchronize 는 latency 측정용 2곳만: YOLO 직후, CNN forward 직후
        frame 시작 시점은 직전 frame 의 마지막 sync 이후라 GPU 가 비어 있음
        """

        start = time.perf_counter()

        detections = self._detect(frame)

        self._sync()
        yolo_end = time.perf_counter()

        main_det = self.select_detection(detections)

        result = {
            "detected": main_det is not None,
            "material_id": None,
            "material": None,
            "bbox": None,
            "yolo_confidence": None,
            "num_detections": len(detections),
            "state": None,
            "cnn_confidence": None,
            "final_class_id": None,
            "final_class": None,
            "crop_box": None,
            "yolo_ms": (yolo_end - start) * 1000.0,
            "crop_ms": 0.0,
            "cnn_ms": 0.0,
            "total_ms": (yolo_end - start) * 1000.0,
        }

        if main_det is None:
            return result

        material_id = main_det["material_id"]
        material_name = self.yolo_names[material_id]

        result.update({
            "material_id": material_id,
            "material": material_name,
            "bbox": main_det["bbox"],
            "yolo_confidence": main_det["confidence"],
        })

        tensor, crop_box = self._preprocess(frame, main_det["bbox"])

        crop_end = time.perf_counter()

        result["crop_box"] = crop_box
        result["crop_ms"] = (crop_end - yolo_end) * 1000.0
        result["total_ms"] = (crop_end - start) * 1000.0

        # padding 후에도 crop 이 비는 극단적 bbox → 재질만 반환
        if tensor is None:
            return result

        if self.log_first_input:
            print(
                f"[CNN input] device={tensor.device} dtype={tensor.dtype} "
                f"shape={list(tensor.shape)} autocast_fp16={self.cnn_amp}"
            )
            self.log_first_input = False

        # 해당 재질의 CNN 하나만 실행
        model, final_ids, states = self.cnn[material_name.lower()]

        probs = self._forward(model, tensor)

        self._sync()
        end = time.perf_counter()

        cnn_index = int(torch.argmax(probs).item())
        final_class_id = final_ids[cnn_index]

        result.update({
            "state": states[cnn_index],
            "cnn_confidence": float(probs[cnn_index].item()),
            "final_class_id": final_class_id,
            "final_class": FINAL_CLASSES[final_class_id],
            "cnn_ms": (end - crop_end) * 1000.0,
            "total_ms": (end - start) * 1000.0,
        })

        return result
