#!/usr/bin/env bash
# Orin 에서 실행: 추론 서비스 컨테이너 빌드 / 실행
#
#   ./deploy/run_orin.sh build          이미지 빌드
#   ./deploy/run_orin.sh run            CSI 카메라로 실행 (이미지 안의 코드)
#   ./deploy/run_orin.sh dev            CSI 카메라 + 코드 마운트 (코드 수정 후 재빌드 불필요)
#   ./deploy/run_orin.sh video <파일>   영상 파일로 실행 (예: videos/comparison_video_30cm.mp4)
#
# 카메라 / ROI / voting 설정은 deploy/camera.env 에서 수정 (재빌드 불필요)
# 접속: http://<orin>:8765/  (stream.mjpg / full.jpg / events / status / reset)

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="recycling-method1:orin"
NAME="recycling-method1"
MODELS_DIR="${MODELS_DIR:-$APP_DIR/models}"
CONFIG="${CAMERA_CONFIG:-$APP_DIR/deploy/camera.env}"

# shellcheck source=/dev/null
source "$CONFIG"

COMMON=(
    --rm
    --name "$NAME"
    --runtime nvidia
    -e AARCH64_IGPU=1
    --network host
    -v /tmp/argus_socket:/tmp/argus_socket
    -v "$MODELS_DIR":/models:ro
)

# 터미널에서 직접 실행할 때만 -it (ssh 원격 실행 / 백그라운드에서는 TTY 없음)
if [[ -t 0 && -t 1 ]]; then
    COMMON+=(-it)
fi

DEV_MOUNTS=(
    -v "$APP_DIR/scripts":/workspace/app/scripts
    -v "$APP_DIR/deploy":/workspace/app/deploy
)

SERVICE_ARGS=(
    --yolo /models/yolo/best.pt --cnn-dir /models/cnn --device 0
    --conf "$CONF" --vote-window "$VOTE_WINDOW" --vote-required "$VOTE_REQUIRED"
)

CSI_ARGS=(
    --source csi
    --csi-sensor-mode "$CSI_SENSOR_MODE" --csi-width "$CSI_WIDTH"
    --csi-height "$CSI_HEIGHT" --csi-fps "$CSI_FPS" --csi-flip "$CSI_FLIP"
)

if [[ -n "${ROI:-}" ]]; then
    # shellcheck disable=SC2206
    CSI_ARGS+=(--roi $ROI)
fi

case "${1:-}" in

    build)
        docker build -f "$APP_DIR/deploy/Dockerfile.orin" -t "$IMAGE" "$APP_DIR"
        ;;

    run)
        docker run "${COMMON[@]}" "$IMAGE" "${CSI_ARGS[@]}" "${SERVICE_ARGS[@]}"
        ;;

    dev)
        docker run "${COMMON[@]}" "${DEV_MOUNTS[@]}" "$IMAGE" \
            "${CSI_ARGS[@]}" "${SERVICE_ARGS[@]}"
        ;;

    video)
        # 녹화 영상은 이미 ROI 가 적용됐을 수 있으므로 ROI 없이 실행
        VIDEO="${2:?영상 파일 경로를 지정하세요}"
        VIDEO_ABS="$(cd "$(dirname "$VIDEO")" && pwd)/$(basename "$VIDEO")"
        docker run "${COMMON[@]}" "${DEV_MOUNTS[@]}" \
            -v "$VIDEO_ABS":/input/video.mp4:ro \
            "$IMAGE" \
            --source /input/video.mp4 --realtime --loop --debug-overlay "${SERVICE_ARGS[@]}"
        ;;

    *)
        sed -n '2,11p' "$0"
        exit 1
        ;;
esac
