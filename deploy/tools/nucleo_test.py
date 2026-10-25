"""
현장 NUCLEO 연결 점검 (Orin 에서 실행, 표준 라이브러리만 사용)

    python3 deploy/tools/nucleo_test.py                 # 연결 + PING 확인
    python3 deploy/tools/nucleo_test.py open 1          # 1번 문 OPEN → ACK / OPENED / CLOSED 까지 기다림
    python3 deploy/tools/nucleo_test.py close 1         # 1번 문 즉시 닫기
    python3 deploy/tools/nucleo_test.py stop 1          # 1번 모터 정지
    python3 deploy/tools/nucleo_test.py all             # 1~4번 문을 차례로 OPEN (문마다 CLOSED 까지)
    python3 deploy/tools/nucleo_test.py raw "OPEN,9"    # 임의 문자열 (오류 응답 확인용)

    --host / --port 로 주소 변경 (기본 192.168.0.120:5000)

키오스크가 켜져 있으면 컨트롤러가 이미 NUCLEO 에 접속해 있어 이 도구의 접속과 겹칠 수 있음
→ 점검 전에 ./deploy/kiosk.sh stop
"""

import argparse
import socket
import sys
import time


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


class Link:

    def __init__(self, host, port):
        log(f"접속 시도 {host}:{port} ...")
        t0 = time.monotonic()
        self.sock = socket.create_connection((host, port), timeout=3)
        self.sock.settimeout(0.2)
        self.buf = b""
        log(f"접속 성공 ({(time.monotonic() - t0) * 1000:.0f} ms)")

    def send(self, line):
        self.sock.sendall((line + "\n").encode("ascii"))
        log(f"SEND {line}")

    def recv_until(self, done, timeout, keepalive=True):
        """done(line) 이 True 인 줄을 받을 때까지 대기. 받은 줄 목록 반환 (keepalive: 1초마다 PING)"""
        got, end, next_ping = [], time.monotonic() + timeout, time.monotonic() + 1.0
        while time.monotonic() < end:
            if keepalive and time.monotonic() >= next_ping:
                self.sock.sendall(b"PING\n")
                next_ping += 1.0
            try:
                chunk = self.sock.recv(1024)
                if not chunk:
                    log("NUCLEO 가 연결을 끊음")
                    return got
                self.buf += chunk
            except socket.timeout:
                continue
            while b"\n" in self.buf:
                raw, self.buf = self.buf.split(b"\n", 1)
                line = raw.decode("ascii", "replace").strip()
                if not line:
                    continue
                if line != "PONG" or not keepalive:
                    log(f"RECV {line}")
                got.append(line)
                if done(line):
                    return got
        return got


def check_ping(link):
    link.send("PING")
    t0 = time.monotonic()
    got = link.recv_until(lambda l: l == "PONG", 3, keepalive=False)
    if "PONG" in got:
        log(f"PING OK ({(time.monotonic() - t0) * 1000:.0f} ms)")
        return True
    log("PING 응답 없음 (PONG 미수신)")
    return False


def check_open(link, door, close_timeout):
    link.send(f"OPEN,{door}")
    got = link.recv_until(lambda l: l == f"CLOSED,{door}" or l.startswith("ERR"), close_timeout)
    expect = [f"ACK,OPEN,{door}", f"OPENED,{door}", f"CLOSED,{door}"]
    for e in expect:
        log(f"  {'✅' if e in got else '❌'} {e}")
    return all(e in got for e in expect)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="ping", choices=["ping", "open", "close", "stop", "all", "raw"])
    p.add_argument("arg", nargs="?")
    p.add_argument("--host", default="192.168.0.120")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--close-timeout", type=float, default=20.0, help="OPEN 후 CLOSED 를 기다리는 시간(초)")
    a = p.parse_args()

    try:
        link = Link(a.host, a.port)
    except OSError as exc:
        log(f"접속 실패: {exc}")
        log("확인: 케이블 / NUCLEO 전원 / IP·포트 / 'ping " + a.host + "'")
        sys.exit(1)

    ok = check_ping(link)

    if a.cmd == "open":
        ok = check_open(link, int(a.arg or 1), a.close_timeout) and ok
    elif a.cmd == "all":
        for door in (1, 2, 3, 4):
            ok = check_open(link, door, a.close_timeout) and ok
    elif a.cmd in ("close", "stop"):
        door = int(a.arg or 1)
        link.send(f"{a.cmd.upper()},{door}")
        final = "CLOSED" if a.cmd == "close" else "STOPPED"
        link.recv_until(lambda l: l == f"{final},{door}" or l.startswith("ERR"), 10)
    elif a.cmd == "raw":
        link.send(a.arg or "")
        link.recv_until(lambda l: False, 3)

    log("결과: " + ("정상" if ok else "확인 필요"))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
