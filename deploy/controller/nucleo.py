"""
Jetson ↔ NUCLEO 문 제어 (프로토콜: 개발 계획서 4장)

    Jetson → NUCLEO   OPEN,<ID> / CLOSE,<ID> / STOP,<ID> / PING
    NUCLEO → Jetson   ACK,<CMD>,<ID> / OPENED,<ID> / CLOSED,<ID> / STOPPED,<ID> / PONG / ERR,<CODE>,<ID>

정상 흐름: Jetson 은 OPEN 만 보내고, NUCLEO 가 열기 → 10초 유지 → 자동 닫기 후 CLOSED 를 보낸다.
CLOSE 는 수동/예외 상황에서 즉시 닫을 때만 쓴다.

두 구현 모두 받은 메시지를 컨트롤러 이벤트 큐에 같은 형태로 넣는다.
    {"type": "nucleo", "msg": "OPENED", "id": 3}
    {"type": "nucleo", "msg": "ERR", "code": "BUSY", "id": 3}
    {"type": "nucleo_link", "up": True/False}
"""

import socket
import threading
import time


def log(tag, text):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{tag}] {text}", flush=True)


def parse_message(line):
    """NUCLEO 한 줄 → 이벤트 dict (모르는 형식이면 None)"""

    parts = [p.strip() for p in line.strip().split(",")]

    if not parts or not parts[0]:
        return None

    msg = parts[0].upper()

    try:
        if msg == "PONG" and len(parts) == 1:
            return {"type": "nucleo", "msg": "PONG"}
        if msg == "ACK" and len(parts) == 3:
            return {"type": "nucleo", "msg": "ACK", "cmd": parts[1].upper(), "id": int(parts[2])}
        if msg in ("OPENED", "CLOSED", "STOPPED") and len(parts) == 2:
            return {"type": "nucleo", "msg": msg, "id": int(parts[1])}
        if msg == "ERR" and len(parts) == 3:
            target = int(parts[2]) if parts[2].isdigit() else None
            return {"type": "nucleo", "msg": "ERR", "code": parts[1].upper(), "id": target}
    except ValueError:
        return None

    return None


# ============================================================
# 실제 TCP 클라이언트
# ============================================================

class TcpNucleo:

    def __init__(self, host, port, events, ping_interval=1.0, pong_timeout=3.0):

        self.addr = (host, port)
        self.events = events
        self.ping_interval = ping_interval
        self.pong_timeout = pong_timeout

        self.sock = None
        self.lock = threading.Lock()
        self.last_pong = 0.0
        self.connected = False

        threading.Thread(target=self._run, daemon=True).start()

    # ---- 송신

    def _send(self, line):

        with self.lock:
            sock = self.sock

        if sock is None:
            log("nucleo", f"미연결 상태라 보내지 못함: {line}")
            return False

        try:
            sock.sendall((line + "\n").encode("ascii"))
            if line != "PING":
                log("nucleo", f"SEND {line}")
            return True
        except OSError as exc:
            log("nucleo", f"송신 실패 {line}: {exc}")
            self._drop()
            return False

    def open_cover(self, cover):
        return self._send(f"OPEN,{cover}")

    def close_cover(self, cover):
        return self._send(f"CLOSE,{cover}")

    def stop_cover(self, cover):
        return self._send(f"STOP,{cover}")

    def buzz(self, kind):
        # 버저 위치 미정 (협의 사항). NUCLEO 쪽으로 정해지면 BUZZ 명령 추가
        log("nucleo", f"BUZZ {kind} (미구현)")

    # ---- 연결 유지

    def _set_link(self, up):
        if up != self.connected:
            self.connected = up
            self.events.put({"type": "nucleo_link", "up": up})
            log("nucleo", f"{'연결됨' if up else '연결 끊김'} {self.addr[0]}:{self.addr[1]}")

    def _drop(self):
        with self.lock:
            sock, self.sock = self.sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        self._set_link(False)

    def _run(self):

        while True:

            try:
                sock = socket.create_connection(self.addr, timeout=3)
                sock.settimeout(0.2)
            except OSError:
                self._set_link(False)
                time.sleep(2.0)
                continue

            with self.lock:
                self.sock = sock

            self.last_pong = time.monotonic()
            self._set_link(True)
            self._session(sock)
            self._drop()
            time.sleep(2.0)

    def _session(self, sock):

        buf = b""
        next_ping = 0.0

        while True:

            now = time.monotonic()

            if now >= next_ping:
                if not self._send("PING"):
                    return
                next_ping = now + self.ping_interval

            if now - self.last_pong > self.pong_timeout:
                log("nucleo", f"PONG 이 {self.pong_timeout:.0f}초간 없음 → 재연결")
                return

            try:
                chunk = sock.recv(1024)
                if not chunk:
                    return
                buf += chunk
            except socket.timeout:
                continue
            except OSError:
                return

            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("ascii", "replace").strip()
                event = parse_message(line)

                if event is None:
                    if line:
                        log("nucleo", f"알 수 없는 메시지: {line!r}")
                    continue

                if event["msg"] == "PONG":
                    self.last_pong = time.monotonic()
                    continue

                log("nucleo", f"RECV {line}")
                self.events.put(event)


# ============================================================
# 가짜 (하드웨어 없이 컨트롤러 흐름 시험)
# NUCLEO 와 같은 순서로 ACK → OPENED → (hold) → CLOSED 이벤트를 만든다
# ============================================================

class MockNucleo:

    def __init__(self, events, move_seconds=1.0, hold_seconds=10.0):
        self.events = events
        self.move = move_seconds
        self.hold = hold_seconds
        self.timers = {}
        events.put({"type": "nucleo_link", "up": True})

    def _later(self, cover, delay, event):
        t = threading.Timer(delay, self.events.put, args=(event,))
        t.daemon = True
        self.timers.setdefault(cover, []).append(t)
        t.start()

    def _cancel(self, cover):
        for t in self.timers.pop(cover, []):
            t.cancel()

    def open_cover(self, cover):
        log("nucleo", f"(mock) SEND OPEN,{cover}")
        self._cancel(cover)
        self.events.put({"type": "nucleo", "msg": "ACK", "cmd": "OPEN", "id": cover})
        self._later(cover, self.move, {"type": "nucleo", "msg": "OPENED", "id": cover})
        self._later(cover, self.move * 2 + self.hold, {"type": "nucleo", "msg": "CLOSED", "id": cover})
        return True

    def close_cover(self, cover):
        log("nucleo", f"(mock) SEND CLOSE,{cover}")
        self._cancel(cover)
        self.events.put({"type": "nucleo", "msg": "ACK", "cmd": "CLOSE", "id": cover})
        self._later(cover, self.move, {"type": "nucleo", "msg": "CLOSED", "id": cover})
        return True

    def stop_cover(self, cover):
        log("nucleo", f"(mock) SEND STOP,{cover}")
        self._cancel(cover)
        self.events.put({"type": "nucleo", "msg": "STOPPED", "id": cover})
        return True

    def buzz(self, kind):
        log("nucleo", f"(mock) BUZZ {kind}")


def make_nucleo(cfg, events):

    mode = cfg.get("mode", "mock")

    if mode == "mock":
        return MockNucleo(events, hold_seconds=cfg.get("mock_hold_seconds", 10.0))

    if mode == "tcp":
        return TcpNucleo(cfg["host"], int(cfg["port"]), events)

    raise ValueError(f"알 수 없는 nucleo mode: {mode}")
