#!/usr/bin/env bash
# Orin 호스트에서 실행: 컨트롤러 + 키오스크 화면 (추론 서비스 컨테이너는 run_orin.sh 로 먼저 실행)
#
#   ./deploy/start_kiosk.sh            컨트롤러 실행 + 모니터에 Chromium 전체화면
#   ./deploy/start_kiosk.sh --no-ui    컨트롤러만 (원격 PC 브라우저로 http://<orin>:8080 확인)
#
# 안내 문구 / 시간 / 커버 번호: deploy/controller/config.json
# 대기 영상: deploy/controller/media/standby.mp4 (없으면 기본 대기 화면)

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python3 "$APP_DIR/deploy/controller/controller.py" &
CONTROLLER_PID=$!
trap 'kill $CONTROLLER_PID 2>/dev/null' EXIT

if [[ "${1:-}" != "--no-ui" ]]; then

    export DISPLAY="${DISPLAY:-:0}"
    sleep 2

    if BROWSER="$(command -v chromium-browser || command -v chromium)"; then
        "$BROWSER" --kiosk --noerrdialogs --disable-infobars \
            --autoplay-policy=no-user-gesture-required http://localhost:8080 &
    elif command -v firefox >/dev/null; then
        # Orin 기본 설치는 Firefox (snap)
        firefox --kiosk http://localhost:8080 &
    else
        echo "브라우저가 없습니다. 원격 브라우저로 http://<orin>:8080 을 여세요." >&2
    fi
fi

wait $CONTROLLER_PID
