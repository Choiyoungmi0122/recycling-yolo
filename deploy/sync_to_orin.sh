#!/usr/bin/env bash
# PC (Git Bash) → Orin: 코드 / 모델 / 테스트 영상 복사
#
#   bash deploy/sync_to_orin.sh            코드만
#   bash deploy/sync_to_orin.sh --models   코드 + 모델 5개 + 30cm 테스트 영상
#
# Orin 쪽 구조 (~/recycling_app):
#   deploy/  scripts/method1_integrated_inference.py
#   models/yolo/best.pt  models/cnn/best_{can,glass,pet,plastic}_efficientnet_b0.pth
#   videos/comparison_video_30cm.mp4

set -euo pipefail

HOST="${ORIN_HOST:-orin}"
REMOTE="recycling_app"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ssh "$HOST" "mkdir -p ~/$REMOTE/deploy/controller/kiosk ~/$REMOTE/deploy/controller/media \
    ~/$REMOTE/scripts ~/$REMOTE/models/yolo ~/$REMOTE/models/cnn ~/$REMOTE/videos"

scp "$ROOT/deploy/Dockerfile.orin" \
    "$ROOT/deploy/inference_service.py" \
    "$ROOT/deploy/run_orin.sh" \
    "$ROOT/deploy/start_kiosk.sh" \
    "$ROOT/deploy/jetson-ota-public.asc" \
    "$HOST:~/$REMOTE/deploy/"

scp "$ROOT/deploy/controller/controller.py" "$HOST:~/$REMOTE/deploy/controller/"
scp "$ROOT/deploy/controller/kiosk/index.html" "$HOST:~/$REMOTE/deploy/controller/kiosk/"

# Orin 에서 직접 조정하는 설정 파일은 처음 한 번만 복사 (덮어쓰지 않음)
#   camera.env: ROI / 카메라,  config.json: 안내 문구 / 시간 / 커버 번호
copy_once() {
    if ! ssh "$HOST" "test -f ~/$REMOTE/$2"; then
        scp "$ROOT/$1" "$HOST:~/$REMOTE/$2"
    fi
}

copy_once deploy/camera.env deploy/camera.env
copy_once deploy/controller/config.json deploy/controller/config.json

scp "$ROOT/scripts/method1_integrated_inference.py" "$HOST:~/$REMOTE/scripts/"

ssh "$HOST" "chmod +x ~/$REMOTE/deploy/run_orin.sh ~/$REMOTE/deploy/start_kiosk.sh"

if [[ "${1:-}" == "--models" ]]; then

    scp "/e/recycling_data/runs/recycling-yolo-comparison/yolo26s_4class_common30_rerun-2/weights/best.pt" \
        "$HOST:~/$REMOTE/models/yolo/best.pt"

    for m in can glass pet plastic; do
        scp "/e/recycling_data/runs/$m-efficientnet-b0/best_${m}_efficientnet_b0.pth" \
            "$HOST:~/$REMOTE/models/cnn/"
    done

    scp "$ROOT/videos/input/comparison_video_30cm.mp4" "$HOST:~/$REMOTE/videos/"
fi

echo "done → $HOST:~/$REMOTE"
