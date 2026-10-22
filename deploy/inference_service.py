"""
방법 1 추론 서비스 (Docker 안에서 실행)

    카메라 / 영상 → (ROI) → Method1Predictor → 5-frame voting → HTTP 로 내보냄

HTTP (기본 0.0.0.0:8765)
    GET  /stream.mjpg  카메라 화면 (bbox 표시, MJPEG) - 키오스크 화면에서 <img src> 로 사용
    GET  /full.jpg     ROI 조정용 원본 전체 화면 (좌표 격자 + 현재 ROI 사각형)
    GET  /events       판정 이벤트 (Server-Sent Events, JSON)
    GET  /status       최근 frame / decision (JSON)
    POST /reset        voting window 초기화 (컨트롤러가 새 물체 인식을 시작할 때)
    GET  /             동작 확인용 페이지

이벤트
    {"type": "frame", ...}     매 frame raw 결과
    {"type": "decision", ...}  voting 결과가 바뀔 때만 (final_class 가 None 이면 UNDECIDED)

안내 문구 / 커버 제어 / NUCLEO 통신은 이 서비스가 아니라 호스트 컨트롤러가 담당한다.

예:
    # PC 에서 영상 파일로 테스트
    python deploy/inference_service.py --source videos/input/comparison_video_30cm.mp4 \
        --yolo <best.pt> --cnn-dir <cnn 폴더> --device 1 --realtime --loop

    # Orin CSI 카메라
    python3 deploy/inference_service.py --source csi --roi 240 150 1400 932 \
        --yolo /models/yolo/best.pt --cnn-dir /models/cnn --device 0
"""

from pathlib import Path
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import queue
import sys
import threading
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from method1_integrated_inference import (  # noqa: E402
    DEFAULT_CNN_MODELS,
    DEFAULT_YOLO_MODEL,
    FINAL_CLASSES,
    Method1Predictor,
)


MATERIAL_COLORS = {
    "CAN": (0, 165, 255),
    "GLASS": (255, 200, 0),
    "PET": (0, 220, 0),
    "PLASTIC": (220, 0, 220),
}


# ============================================================
# 입력
# ============================================================

def csi_pipeline(sensor_id, sensor_mode, width, height, fps, flip_method):

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1, format=NV12 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=1 max-buffers=1 sync=false"
    )


class LatestFrameReader:
    """카메라용: 별도 thread 에서 계속 읽고 가장 최신 frame 만 넘긴다 (지연 누적 방지)"""

    def __init__(self, cap):

        self.cap = cap
        self.frame = None
        self.ok = True
        self.cond = threading.Condition()

        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):

        while self.ok:

            ok, frame = self.cap.read()

            with self.cond:
                self.ok = ok
                self.frame = frame if ok else None
                self.cond.notify_all()

    def read(self):

        with self.cond:

            while self.ok and self.frame is None:
                self.cond.wait()

            frame, self.frame = self.frame, None

        return frame


class FileReader:
    """영상 파일용: 순서대로 읽고, --realtime 이면 원본 fps 속도로, --loop 면 반복"""

    def __init__(self, path, realtime, loop):

        self.path = path
        self.realtime = realtime
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        self.interval = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.next_time = time.perf_counter()

    def read(self):

        ok, frame = self.cap.read()

        if not ok and self.loop:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.path)
            ok, frame = self.cap.read()

        if not ok:
            return None

        if self.realtime:
            self.next_time += self.interval
            delay = self.next_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                self.next_time = time.perf_counter()

        return frame


def open_source(args):

    if args.source == "csi":
        pipeline = csi_pipeline(
            args.csi_sensor_id, args.csi_sensor_mode, args.csi_width, args.csi_height,
            args.csi_fps, args.csi_flip,
        )
        print(f"[source] CSI: {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    elif args.source.isdigit():
        print(f"[source] USB camera {args.source}")
        cap = cv2.VideoCapture(int(args.source))

    else:
        if not Path(args.source).exists():
            raise FileNotFoundError(f"영상 없음: {args.source}")
        print(f"[source] file {args.source}")
        return FileReader(args.source, args.realtime, args.loop)

    if not cap.isOpened():
        raise RuntimeError(f"카메라 열기 실패: {args.source}")

    return LatestFrameReader(cap)


# ============================================================
# Voting (기존 compare_voting_357.py 와 동일 규칙)
# 최근 N frame 중 같은 11-class 가 required 표 이상이면 확정, 아니면 UNDECIDED
# ============================================================

class FrameVoter:

    def __init__(self, window, required):

        self.window = deque(maxlen=window)
        self.required = required

    def reset(self):
        self.window.clear()

    def update(self, result):

        self.window.append(result if result["final_class"] is not None else None)

        valid = [item for item in self.window if item is not None]

        if not valid:
            return None, 0

        class_id, count = Counter(
            item["final_class_id"] for item in valid
        ).most_common(1)[0]

        if count < self.required:
            return None, count

        return class_id, count


# ============================================================
# 이벤트 / 영상 공유
# ============================================================

class Broadcaster:

    def __init__(self):

        self.lock = threading.Lock()
        self.clients = []

        self.jpeg = None
        self.jpeg_id = 0
        self.jpeg_cond = threading.Condition()
        self.stream_clients = 0

        self.last_frame = None
        self.last_decision = None
        self.last_raw = None
        self.roi = None
        self.reset_requested = threading.Event()

    # SSE

    def subscribe(self):
        q = queue.Queue(maxsize=256)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.remove(q)

    def publish(self, event):

        data = json.dumps(event, ensure_ascii=False)

        with self.lock:
            for q in self.clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass  # 느린 client 는 frame 이벤트를 건너뜀

    # MJPEG

    def set_jpeg(self, jpeg):
        with self.jpeg_cond:
            self.jpeg = jpeg
            self.jpeg_id += 1
            self.jpeg_cond.notify_all()

    def wait_jpeg(self, last_id, timeout=2.0):
        with self.jpeg_cond:
            self.jpeg_cond.wait_for(lambda: self.jpeg_id != last_id, timeout)
            return self.jpeg_id, self.jpeg


class QuietHTTPServer(ThreadingHTTPServer):
    """client 가 연결을 끊을 때 나는 traceback 은 정상 동작이므로 출력하지 않음"""

    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


TEST_PAGE ="""<!doctype html><meta charset="utf-8"><title>Method1 service</title>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif">
<img src="/stream.mjpg" style="width:100%;max-height:80vh;object-fit:contain">
<pre id="d" style="font-size:20px;padding:8px"></pre>
<script>
const es = new EventSource('/events');
es.onmessage = e => { const m = JSON.parse(e.data);
  if (m.type === 'decision') document.getElementById('d').textContent = JSON.stringify(m, null, 1); };
</script>"""


def make_handler(hub):

    class Handler(BaseHTTPRequestHandler):

        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):

            if self.path == "/":
                body = TEST_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path == "/status":
                self._json({"frame": hub.last_frame, "decision": hub.last_decision})

            elif self.path.startswith("/full.jpg"):
                self._full()

            elif self.path == "/events":
                self._events()

            elif self.path.startswith("/stream.mjpg"):
                self._stream()

            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):

            if self.path == "/reset":
                hub.reset_requested.set()
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)

        def _full(self):
            """ROI 조정용: 원본 전체 frame + 100px 좌표 격자 + 현재 ROI 사각형"""

            raw = hub.last_raw

            if raw is None:
                self._json({"error": "no frame yet"}, 503)
                return

            view = raw.copy()
            h, w = view.shape[:2]

            for x in range(0, w, 100):
                cv2.line(view, (x, 0), (x, h), (80, 80, 80), 1)
                cv2.putText(view, str(x), (x + 2, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            for y in range(0, h, 100):
                cv2.line(view, (0, y), (w, y), (80, 80, 80), 1)
                cv2.putText(view, str(y), (2, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            if hub.roi:
                x1, y1, x2, y2 = hub.roi
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(view, f"ROI {x1},{y1} - {x2},{y2}", (x1 + 5, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.putText(view, f"{w}x{h}", (w - 160, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 85])
            body = jpeg.tobytes()

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _events(self):

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = hub.subscribe()

            try:
                if hub.last_decision is not None:
                    q.put_nowait(json.dumps(hub.last_decision, ensure_ascii=False))

                while True:
                    try:
                        data = q.get(timeout=10)
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()

            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

            finally:
                hub.unsubscribe(q)

        def _stream(self):

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with hub.lock:
                hub.stream_clients += 1

            last_id = -1

            try:
                while True:
                    last_id, jpeg = hub.wait_jpeg(last_id)
                    if jpeg is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        + jpeg + b"\r\n"
                    )
                    self.wfile.flush()

            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

            finally:
                with hub.lock:
                    hub.stream_clients -= 1

    return Handler


# ============================================================
# 화면용 표시
#   기본: bbox 만 (사용자 화면 - 한글 안내 문구는 키오스크가 담당)
#   --debug-overlay: class / confidence / voting 결과 글자까지 (개발 확인용)
# ============================================================

def draw_result(frame, result, decision, debug):

    if result["detected"]:

        color = MATERIAL_COLORS.get(result["material"], (255, 255, 255))
        x1, y1, x2, y2 = (int(round(v)) for v in result["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    if not debug:
        return

    if result["detected"]:

        label = f"{result['material']} {result['yolo_confidence']:.2f}"
        if result["final_class"] is not None:
            label += f" | {result['final_class']} {result['cnn_confidence']:.2f}"

        cv2.putText(
            frame, label, (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )

    if decision is not None:
        cv2.putText(
            frame, f"DECIDED: {decision}", (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
        )


# ============================================================
# Main
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True,
                        help="'csi' | USB 카메라 번호 | 영상 파일 경로")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="원본 frame 에서 잘라 쓸 영역 (예: 240 150 1400 932)")

    parser.add_argument("--yolo", default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--cnn-dir", default=None,
                        help="best_{can,glass,pet,plastic}_efficientnet_b0.pth 가 있는 폴더 "
                             "(없으면 PC 기본 경로)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--yolo-fp16", action="store_true")
    parser.add_argument("--no-cnn-amp", action="store_true")

    parser.add_argument("--vote-window", type=int, default=5)
    parser.add_argument("--vote-required", type=int, default=3)

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--debug-overlay", action="store_true",
                        help="스트림 화면에 class / confidence / voting 결과 글자 표시")

    parser.add_argument("--realtime", action="store_true",
                        help="영상 파일을 원본 fps 속도로 재생")
    parser.add_argument("--loop", action="store_true", help="영상 파일 반복")

    parser.add_argument("--csi-sensor-id", type=int, default=0,
                        help="Argus 카메라 번호 (Jetson-IO 에서 인식된 순서, 물리 포트 번호와 다를 수 있음)")
    parser.add_argument("--csi-sensor-mode", type=int, default=3)
    parser.add_argument("--csi-width", type=int, default=1640)
    parser.add_argument("--csi-height", type=int, default=1232)
    parser.add_argument("--csi-fps", type=int, default=30)
    parser.add_argument("--csi-flip", type=int, default=6)

    return parser.parse_args()


def main():

    args = parse_args()

    cnn_models = DEFAULT_CNN_MODELS

    if args.cnn_dir:
        cnn_dir = Path(args.cnn_dir)
        cnn_models = {
            material: cnn_dir / f"best_{material}_efficientnet_b0.pth"
            for material in DEFAULT_CNN_MODELS
        }

    predictor = Method1Predictor(
        yolo_model=args.yolo,
        cnn_models=cnn_models,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        yolo_fp16=args.yolo_fp16,
        cnn_amp=not args.no_cnn_amp,
    )

    reader = open_source(args)

    first = reader.read()

    if first is None:
        raise RuntimeError("첫 frame 을 읽지 못했습니다.")

    full_h, full_w = first.shape[:2]

    if args.roi:
        x1, y1, x2, y2 = args.roi
        if not (0 <= x1 < x2 <= full_w and 0 <= y1 < y2 <= full_h):
            raise ValueError(f"ROI {args.roi} 가 frame 크기 {full_w}x{full_h} 를 벗어남")

    def apply_roi(raw):
        if not args.roi:
            return raw
        x1, y1, x2, y2 = args.roi
        return raw[y1:y2, x1:x2]

    print(f"[source] frame {full_w}x{full_h} | ROI {args.roi or 'none'}")

    predictor.warmup(frame_shape=apply_roi(first).shape)

    for name, info in predictor.device_report().items():
        print(f"[device] {name:12s}: {info}")

    voter = FrameVoter(args.vote_window, args.vote_required)
    hub = Broadcaster()
    hub.roi = args.roi

    server = QuietHTTPServer((args.host, args.port), make_handler(hub))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"[http] http://{args.host}:{args.port}/  (stream.mjpg / full.jpg / events / status / reset)")
    print(f"[voting] {args.vote_window} frame / {args.vote_required} vote")

    frame_index = 0
    decided_id = None
    raw = first
    fps_t0 = time.perf_counter()

    try:
        while raw is not None:

            frame_index += 1

            # ROI 조정용 /full.jpg 에서 원본 전체를 보여주기 위해 보관
            hub.last_raw = raw
            frame = apply_roi(raw)

            if hub.reset_requested.is_set():
                hub.reset_requested.clear()
                voter.reset()
                decided_id = None

            result = predictor.predict(frame)
            class_id, vote_count = voter.update(result)

            frame_event = {
                "type": "frame",
                "frame": frame_index,
                "time": time.time(),
                "detected": result["detected"],
                "material": result["material"],
                "yolo_confidence": result["yolo_confidence"],
                "state": result["state"],
                "cnn_confidence": result["cnn_confidence"],
                "final_class_id": result["final_class_id"],
                "final_class": result["final_class"],
                "bbox": result["bbox"],
                "total_ms": round(result["total_ms"], 2),
            }

            hub.last_frame = frame_event
            hub.publish(frame_event)

            if class_id != decided_id:

                decided_id = class_id
                name = FINAL_CLASSES[class_id] if class_id is not None else None

                decision_event = {
                    "type": "decision",
                    "frame": frame_index,
                    "time": time.time(),
                    "final_class_id": class_id,
                    "final_class": name,
                    "material": name.split("_", 1)[0].upper() if name else None,
                    "state": name.split("_", 1)[1] if name else None,
                    "vote_count": vote_count,
                    "window": args.vote_window,
                }

                hub.last_decision = decision_event
                hub.publish(decision_event)
                print(f"[decision] frame {frame_index}: {name or 'UNDECIDED'} ({vote_count} votes)")

            if hub.stream_clients > 0:
                view = frame.copy()
                draw_result(
                    view, result,
                    FINAL_CLASSES[decided_id] if decided_id is not None else None,
                    args.debug_overlay,
                )
                ok, jpeg = cv2.imencode(
                    ".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                )
                if ok:
                    hub.set_jpeg(jpeg.tobytes())

            if frame_index % 100 == 0:
                now = time.perf_counter()
                print(f"[run] frame {frame_index} | {100 / (now - fps_t0):.1f} FPS")
                fps_t0 = now

            raw = reader.read()

    except KeyboardInterrupt:
        pass

    finally:
        server.shutdown()
        print("[exit]")


if __name__ == "__main__":
    main()
