"""
가짜 NUCLEO TCP 서버 — 실제 펌웨어가 오기 전에 Jetson 컨트롤러를 시험하기 위한 것

프로토콜 (개발 계획서 4장)
    OPEN,<ID>   → ACK,OPEN,<ID> → (move) OPENED,<ID> → (hold 10초) → (move) CLOSED,<ID>
    CLOSE,<ID>  → ACK,CLOSE,<ID> → 자동 닫기 타이머 취소 → (move) CLOSED,<ID>
    STOP,<ID>   → ACK,STOP,<ID> → STOPPED,<ID> (이후 새 명령 전까지 대기)
    PING        → PONG
    오류        → ERR,INVALID_CMD,0 / ERR,INVALID_ID,<ID> / ERR,BUSY,<ID>
    PING 이 3초간 없으면 열린 문을 모두 닫음 (안전 동작)

실행 (PC 또는 Orin):
    python deploy/tools/fake_nucleo.py                  # 0.0.0.0:5000
    python deploy/tools/fake_nucleo.py --hold 3         # 빠른 시험용 3초 유지
    python deploy/tools/fake_nucleo.py --busy-rate 0.3  # 30% 확률로 BUSY 응답 (오류 처리 시험)
"""

import argparse
import random
import socket
import threading
import time

DOORS = (1, 2, 3, 4)


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


class FakeNucleo:

    def __init__(self, conn, move, hold, busy_rate):
        self.conn = conn
        self.move = move
        self.hold = hold
        self.busy_rate = busy_rate
        self.lock = threading.Lock()
        self.state = {d: "CLOSED" for d in DOORS}   # CLOSED / OPENING / OPEN / CLOSING / STOPPED
        self.generation = {d: 0 for d in DOORS}     # 새 명령이 오면 이전 타이머 무효화
        self.last_ping = time.monotonic()
        self.alive = True

    def send(self, line):
        try:
            self.conn.sendall((line + "\n").encode("ascii"))
            if line != "PONG":
                log(f"  → {line}")
        except OSError:
            self.alive = False

    # ---- 문 동작 (move 초 걸린다고 가정)

    def _after(self, delay, door, gen, fn):
        def run():
            time.sleep(delay)
            with self.lock:
                if self.generation[door] != gen or not self.alive:
                    return
                fn()
        threading.Thread(target=run, daemon=True).start()

    def _start_close(self, door):
        self.generation[door] += 1
        gen = self.generation[door]
        self.state[door] = "CLOSING"
        log(f"  [문 {door}] 닫는 중")

        def closed():
            self.state[door] = "CLOSED"
            self.send(f"CLOSED,{door}")

        self._after(self.move, door, gen, closed)

    def _open(self, door):
        self.generation[door] += 1
        gen = self.generation[door]
        self.state[door] = "OPENING"
        log(f"  [문 {door}] 여는 중")

        def opened():
            self.state[door] = "OPEN"
            self.send(f"OPENED,{door}")
            log(f"  [문 {door}] {self.hold:.0f}초 유지")
            self._after(self.hold, door, gen, lambda: self._start_close(door))

        self._after(self.move, door, gen, opened)

    # ---- 명령 처리

    def handle(self, line):

        parts = [p.strip().upper() for p in line.split(",")]
        cmd = parts[0]

        if cmd == "PING" and len(parts) == 1:
            self.last_ping = time.monotonic()
            self.send("PONG")
            return

        log(f"← {line}")

        if cmd not in ("OPEN", "CLOSE", "STOP") or len(parts) != 2:
            self.send("ERR,INVALID_CMD,0")
            return

        if not parts[1].isdigit() or int(parts[1]) not in DOORS:
            self.send(f"ERR,INVALID_ID,{parts[1]}")
            return

        door = int(parts[1])

        with self.lock:

            if cmd == "STOP":
                self.generation[door] += 1
                self.state[door] = "STOPPED"
                self.send(f"ACK,STOP,{door}")
                self.send(f"STOPPED,{door}")
                return

            if self.busy_rate and random.random() < self.busy_rate:
                self.send(f"ERR,BUSY,{door}")
                return

            self.send(f"ACK,{cmd},{door}")

            if cmd == "OPEN":
                self._open(door)
            else:
                self._start_close(door)

    # ---- PING 감시: 3초간 없으면 열린 문 닫기

    def watchdog(self):
        while self.alive:
            time.sleep(0.5)
            if time.monotonic() - self.last_ping > 3.0:
                with self.lock:
                    opened = [d for d in DOORS if self.state[d] in ("OPENING", "OPEN")]
                    for d in opened:
                        log(f"  PING 3초 없음 → 안전 동작: 문 {d} 닫기")
                        self._start_close(d)
                self.last_ping = time.monotonic()


def serve(args):

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Windows 의 SO_REUSEADDR 는 다른 프로그램이 쓰는 포트에도 bind 를 허용해 접속을 뺏길 수 있음
    # → Windows 는 단독 사용, Linux 는 재시작 직후 재사용만 허용
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    log(f"가짜 NUCLEO 대기 {args.host}:{args.port} (move {args.move}s, hold {args.hold}s)")

    while True:
        conn, addr = srv.accept()
        log(f"접속: {addr[0]}:{addr[1]}")
        dev = FakeNucleo(conn, args.move, args.hold, args.busy_rate)
        threading.Thread(target=dev.watchdog, daemon=True).start()

        buf = b""
        try:
            while dev.alive:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("ascii", "replace").strip()
                    if line:
                        dev.handle(line)
        except OSError:
            pass
        finally:
            dev.alive = False
            conn.close()
            log("연결 종료")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--move", type=float, default=1.0, help="여닫는 데 걸리는 시간(초)")
    p.add_argument("--hold", type=float, default=10.0, help="열린 상태 유지 시간(초)")
    p.add_argument("--busy-rate", type=float, default=0.0)
    serve(p.parse_args())


if __name__ == "__main__":
    main()
