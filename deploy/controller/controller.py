"""
분리수거함 메인 컨트롤러 (Orin 호스트에서 실행, Python 표준 라이브러리만 사용)

    추론 서비스(/events) → 상태 머신 → 키오스크 화면 / 버저 / NUCLEO(커버) / 로그

상태 흐름 (system_workflow 도식 기준)
    STANDBY      대기 영상. 물체가 stabilize_seconds 이상 감지되면 인식 시작
    RECOGNIZING  카메라 화면 + "인식 중". voting 확정 결과를 기다림
      ├ normal            → ACCEPTED: OPEN,<ID> 전송 → NUCLEO 가 10초 뒤 자동으로 닫고
      │                     CLOSED,<ID> 를 보내면 대기 (closed_timeout_seconds 안에 안 오면 CLOSE 후 대기)
      ├ dirty / packaging → GUIDE: 안내 문구 (커버 미개방) → guide_seconds 후 재인식
      └ 시간 초과         → RETRY: "각도를 바꿔…" → 재인식 (max_retries 회)
                             → FAILED: "일반 쓰레기통…" → 대기
    물체가 absent_seconds 이상 사라지면 (ACCEPTED 제외) 대기로 복귀

키오스크 화면 (기본 :8080)
    GET /       키오스크 페이지 (Chromium 전체화면으로 띄움)
    GET /ui     화면 상태 (Server-Sent Events)
    GET /state  현재 상태 (JSON)
    GET /stream.mjpg  추론 서비스 카메라 스트림 중계

실행:
    python3 deploy/controller/controller.py                 # config.json 사용
    python3 deploy/controller/controller.py --config my.json
"""

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import mimetypes
import queue
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nucleo import make_nucleo  # noqa: E402


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(tag, text):
    print(f"[{now_str()}] [{tag}] {text}", flush=True)


# ============================================================
# 추론 서비스 이벤트 수신 (SSE, 끊기면 자동 재연결)
# ============================================================

class ServiceEventReader(threading.Thread):

    def __init__(self, service_url, out_queue):
        super().__init__(daemon=True)
        self.url = service_url.rstrip("/") + "/events"
        self.out = out_queue

    def run(self):

        while True:

            try:
                with urllib.request.urlopen(self.url, timeout=30) as resp:

                    self.out.put({"type": "service", "up": True})

                    for raw in resp:
                        line = raw.decode("utf-8").strip()
                        if line.startswith("data:"):
                            self.out.put(json.loads(line[5:]))

            except Exception as exc:  # noqa: BLE001 - 어떤 오류든 재연결
                self.out.put({"type": "service", "up": False, "error": str(exc)})

            time.sleep(2.0)


def post_reset(service_url):

    req = urllib.request.Request(service_url.rstrip("/") + "/reset", method="POST", data=b"")

    try:
        urllib.request.urlopen(req, timeout=2).read()
    except Exception as exc:  # noqa: BLE001
        log("service", f"reset 실패: {exc}")


# ============================================================
# 키오스크 화면 상태 공유
# ============================================================

class UiHub:

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = []
        self.state = {"screen": "standby"}

    def set(self, **state):

        self.state = state
        data = json.dumps(state, ensure_ascii=False)

        with self.lock:
            for q in self.clients:
                q.put(data)

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.clients.append(q)
        q.put(json.dumps(self.state, ensure_ascii=False))
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.remove(q)


class QuietHTTPServer(ThreadingHTTPServer):

    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def make_handler(ui, kiosk_dir, media_root, kiosk_config, service_url):

    class Handler(BaseHTTPRequestHandler):

        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, body, content_type, code=200):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):

            path = self.path.split("?", 1)[0]

            if path in ("/", "/index.html"):
                self._send((kiosk_dir / "index.html").read_bytes(), "text/html; charset=utf-8")

            elif path == "/config.json":
                self._send(json.dumps(kiosk_config).encode("utf-8"), "application/json")

            elif path == "/state":
                self._send(json.dumps(ui.state, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

            elif path == "/ui":
                self._ui_events()

            elif path == "/stream.mjpg":
                self._stream_proxy()

            elif path.startswith("/media/"):
                self._media(path[len("/media/"):])

            else:
                self._send(b"not found", "text/plain", 404)

        def _media(self, rel):

            target = (media_root / rel).resolve()

            if media_root not in target.parents or not target.is_file():
                self._send(b"not found", "text/plain", 404)
                return

            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(target.read_bytes(), ctype)

        def _stream_proxy(self):
            """추론 서비스 카메라 스트림을 같은 주소(:8080)로 중계 - 키오스크는 포트 하나만 사용"""

            try:
                upstream = urllib.request.urlopen(
                    service_url.rstrip("/") + "/stream.mjpg", timeout=5,
                )
            except Exception:  # noqa: BLE001
                self._send(b"camera stream unavailable", "text/plain", 503)
                return

            with upstream:

                self.send_response(200)
                self.send_header("Content-Type", upstream.headers["Content-Type"])
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                try:
                    while True:
                        chunk = upstream.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()

                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        def _ui_events(self):

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            q = ui.subscribe()

            try:
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
                ui.unsubscribe(q)

    return Handler


# ============================================================
# 결과 로그 (JSON Lines, 날짜별 파일)
# ============================================================

class ResultLogger:

    def __init__(self, log_dir):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, **record):
        record = {"time": now_str(), **record}
        path = self.dir / f"results_{time.strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log("result", json.dumps(record, ensure_ascii=False))


# ============================================================
# 상태 머신
# ============================================================

class Controller:

    def __init__(self, cfg, ui, nucleo, results):

        self.cfg = cfg
        self.t = cfg["timing"]
        self.ui = ui
        self.nucleo = nucleo
        self.results = results
        self.service_url = cfg["service_url"]

        self.state = None
        self.deadline = None
        self.retries = 0
        self.reset_time = 0.0
        self.open_cover = None
        self.last_class = None
        self.nucleo_up = False

        self.detect_start = None
        self.last_seen = 0.0

        self.go_standby()

    # --------------------------------------------------------

    def present(self, now):
        return now - self.last_seen < self.t["absent_seconds"]

    def set_state(self, state, deadline=None):
        if state != self.state:
            log("state", f"{self.state} → {state}")
        self.state = state
        self.deadline = deadline

    def camera_ui(self, tone, title, message):
        self.ui.set(screen="camera", tone=tone, title=title, message=message, state=self.state)

    # --------------------------------------------------------

    def go_standby(self):
        self.set_state("STANDBY")
        self.detect_start = None
        self.retries = 0
        self.ui.set(screen="standby", state=self.state)

    def start_recognition(self, now):
        self.reset_time = time.time()
        post_reset(self.service_url)
        self.set_state("RECOGNIZING", now + self.t["recognize_timeout_seconds"])
        self.camera_ui("info", "", self.cfg["messages"]["recognizing"])

    def apply_rule(self, now, event):

        final_class = event["final_class"]
        rule = self.cfg["rules"][final_class]
        material = self.cfg["materials"][event["material"]]
        self.last_class = final_class

        if rule["action"] == "open":

            self.open_cover = material["cover"]

            if not self.nucleo.open_cover(self.open_cover):
                self.door_error(now, "send_failed")
                return

            self.nucleo.buzz("ok")
            # 정상이면 NUCLEO 가 10초 뒤 자동으로 닫고 CLOSED 를 보냄 → on_event 에서 대기로
            self.set_state("ACCEPTED", now + self.t.get("closed_timeout_seconds", 20.0))
            self.camera_ui(
                "ok",
                f"{material['name']} · 재활용 가능",
                self.cfg["messages"]["open"].format(name=material["name"]),
            )

        else:

            self.nucleo.buzz("warn")
            self.results.write(result="guide", final_class=final_class, retries=self.retries)
            self.set_state("GUIDE", now + self.t["guide_seconds"])
            self.camera_ui("warn", f"{material['name']} · 추가 처리 필요", rule["message"])

    def door_error(self, now, reason):
        """문을 열지 못함 (송신 실패 / ERR / STOPPED) → 안내 후 대기"""
        self.nucleo.buzz("error")
        self.results.write(result="door_error", reason=reason, final_class=self.last_class,
                           cover=self.open_cover, retries=self.retries)
        self.open_cover = None
        self.set_state("FAILED", now + self.t["failed_seconds"])
        self.camera_ui("error", "수거함 오류", self.cfg["messages"].get(
            "door_error", "수거함을 열 수 없습니다. 잠시 후 다시 시도해주세요."))

    def on_nucleo(self, event, now):

        if self.state != "ACCEPTED" or event.get("id") != self.open_cover:
            return

        msg = event["msg"]

        if msg == "CLOSED":
            self.results.write(result="accepted", final_class=self.last_class,
                               cover=self.open_cover, retries=self.retries)
            self.open_cover = None
            self.go_standby()

        elif msg == "ERR":
            self.door_error(now, f"ERR_{event.get('code')}")

        elif msg == "STOPPED":
            self.door_error(now, "STOPPED")

    # --------------------------------------------------------

    def on_event(self, event, now):

        if event["type"] == "service":
            log("service", "연결됨" if event["up"] else f"연결 끊김: {event.get('error')}")
            return

        if event["type"] == "nucleo_link":
            self.nucleo_up = event["up"]
            return

        if event["type"] == "nucleo":
            self.on_nucleo(event, now)
            return

        if event["type"] == "frame":

            if event["detected"]:
                self.last_seen = now
                if self.detect_start is None:
                    self.detect_start = now
            else:
                self.detect_start = None

            if (
                self.state == "STANDBY"
                and self.detect_start is not None
                and now - self.detect_start >= self.t["stabilize_seconds"]
            ):
                self.retries = 0
                self.start_recognition(now)

        elif event["type"] == "decision":

            # reset 이전에 나온 결과는 무시
            if (
                self.state == "RECOGNIZING"
                and event["final_class"] is not None
                and event["time"] >= self.reset_time
            ):
                self.apply_rule(now, event)

    def on_tick(self, now):

        if self.state in ("RECOGNIZING", "GUIDE", "RETRY") and not self.present(now):
            self.results.write(result="abandoned", final_class=self.last_class, retries=self.retries)
            self.go_standby()
            return

        if self.deadline is None or now < self.deadline:
            return

        if self.state == "RECOGNIZING":

            if self.retries < self.t["max_retries"]:
                self.retries += 1
                self.nucleo.buzz("warn")
                self.set_state("RETRY", now + self.t["retry_prompt_seconds"])
                self.camera_ui("warn", f"다시 인식 ({self.retries}/{self.t['max_retries']})",
                               self.cfg["messages"]["retry"])
            else:
                self.nucleo.buzz("error")
                self.results.write(result="failed", final_class=None, retries=self.retries)
                self.set_state("FAILED", now + self.t["failed_seconds"])
                self.camera_ui("error", "판정 실패", self.cfg["messages"]["failed"])

        elif self.state in ("RETRY", "GUIDE"):
            self.start_recognition(now)

        elif self.state == "ACCEPTED":
            # CLOSED 가 제한 시간 안에 오지 않음 → 예외 상황이므로 CLOSE 로 즉시 닫기 요청
            log("nucleo", f"CLOSED,{self.open_cover} 이 오지 않음 → CLOSE 전송")
            self.nucleo.close_cover(self.open_cover)
            self.results.write(result="close_timeout", final_class=self.last_class,
                               cover=self.open_cover, retries=self.retries)
            self.open_cover = None
            self.go_standby()

        elif self.state == "FAILED":
            self.go_standby()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--service-url", default=None, help="config 의 service_url 대신 사용")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    if args.service_url:
        cfg["service_url"] = args.service_url

    base = config_path.parent
    media_root = (base / "media").resolve()
    media_root.mkdir(exist_ok=True)

    standby_video = cfg.get("standby_video", "")
    kiosk_config = {
        "standby_video": "/" + standby_video if (base / standby_video).is_file() else None,
    }

    ui = UiHub()
    events = queue.Queue()

    server = QuietHTTPServer(
        (cfg["ui_host"], cfg["ui_port"]),
        make_handler(ui, HERE / "kiosk", media_root, kiosk_config, cfg["service_url"]),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()

    ServiceEventReader(cfg["service_url"], events).start()

    controller = Controller(
        cfg, ui, make_nucleo(cfg["nucleo"], events), ResultLogger(base / cfg["log_dir"]),
    )

    log("ui", f"http://{cfg['ui_host']}:{cfg['ui_port']}/  (standby video: "
              f"{kiosk_config['standby_video'] or '없음 - 기본 대기 화면'})")
    log("service", cfg["service_url"])

    try:
        while True:
            try:
                event = events.get(timeout=0.1)
                controller.on_event(event, time.monotonic())
            except queue.Empty:
                pass
            controller.on_tick(time.monotonic())

    except KeyboardInterrupt:
        pass

    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
