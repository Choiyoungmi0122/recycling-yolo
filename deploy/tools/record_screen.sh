#!/usr/bin/env bash
# 모니터 화면 녹화 (Orin 호스트, GStreamer · 추가 설치 불필요)
#
#   ./deploy/tools/record_screen.sh start    녹화 시작 (SSH 가 끊겨도 계속 녹화)
#   ./deploy/tools/record_screen.sh stop     녹화 끝 → ~/Videos/screen_날짜_시각.mp4
#   ./deploy/tools/record_screen.sh status   녹화 중인지 확인
#
# Orin Nano 는 하드웨어 인코더가 없어 CPU(x264, 15fps)로 인코딩

set -uo pipefail

STATE_DIR="$HOME/.local/state/recycling-kiosk"
PID_FILE="$STATE_DIR/record.pid"
OUT_FILE="$STATE_DIR/record.out"
LOG="$STATE_DIR/record.log"
OUT_DIR="$HOME/Videos"
FPS="${FPS:-15}"
BITRATE="${BITRATE:-4000}"   # kbps

mkdir -p "$STATE_DIR" "$OUT_DIR"

say() { echo "[record] $*"; }

recording() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "${1:-}" in
    start)
        if recording; then
            say "이미 녹화 중: $(cat "$OUT_FILE")"
            exit 0
        fi
        if [[ -z "${DISPLAY:-}" ]]; then
            sock="$(ls /tmp/.X11-unix/ 2>/dev/null | sort -V | tail -1)"
            [[ -n "$sock" ]] && export DISPLAY=":${sock#X}"
        fi
        [[ -z "${XAUTHORITY:-}" && -f "/run/user/$(id -u)/gdm/Xauthority" ]] \
            && export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"

        out="$OUT_DIR/screen_$(date +%Y%m%d_%H%M%S).mp4"
        # -e: 종료 신호(SIGINT)를 받으면 mp4 를 정상적으로 마무리
        setsid gst-launch-1.0 -e \
            ximagesrc display-name="$DISPLAY" use-damage=false show-pointer=false \
            ! video/x-raw,framerate="$FPS"/1 ! queue ! videoconvert ! video/x-raw,format=I420 \
            ! x264enc speed-preset=ultrafast tune=zerolatency bitrate="$BITRATE" key-int-max=$((FPS * 2)) \
            ! mp4mux ! filesink location="$out" \
            >"$LOG" 2>&1 < /dev/null &
        echo $! > "$PID_FILE"
        echo "$out" > "$OUT_FILE"
        sleep 2
        if recording; then
            say "녹화 시작 → $out (끝낼 때: $0 stop)"
        else
            say "녹화 시작 실패 — 로그: $LOG"
            tail -5 "$LOG"
            exit 1
        fi
        ;;
    stop)
        if ! recording; then
            say "녹화 중이 아님"
            exit 0
        fi
        kill -INT "$(cat "$PID_FILE")"
        for _ in $(seq 1 20); do
            recording || break
            sleep 0.5
        done
        recording && kill "$(cat "$PID_FILE")" 2>/dev/null
        out="$(cat "$OUT_FILE")"
        rm -f "$PID_FILE"
        say "녹화 끝 → $out ($(du -h "$out" 2>/dev/null | cut -f1))"
        ;;
    status)
        if recording; then say "녹화 중: $(cat "$OUT_FILE")"; else say "녹화 중이 아님"; fi
        ;;
    *)
        sed -n '2,8p' "$0"; exit 1 ;;
esac
