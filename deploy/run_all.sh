#!/usr/bin/env bash
# 전체 실행 (Orin 터미널에서 이것 하나만)
#   대기 슬라이드 먼저 표시 → NUCLEO 연결 확인 → 추론 서비스(Docker + CSI 카메라) 시작
#   (추론 서비스가 뜨면 컨트롤러가 자동으로 연결되어, 물체 감지 시 카메라 화면으로 전환)
#
#   ./deploy/run_all.sh          전부 켜기
#   ./deploy/run_all.sh stop     전부 끄기 (감시도 멈춤. 재부팅하면 다시 자동 실행)
#   ./deploy/run_all.sh status   상태 보기
#   ./deploy/run_all.sh network  NUCLEO 네트워크만 설정·확인 (유선 직결 설정은 처음 한 번 sudo)
#   ./deploy/run_all.sh enable   전원 켜면 슬라이드 + 추론까지 자동 실행 (기본)
#   ./deploy/run_all.sh disable  자동 실행 끄기 = 점검 모드 (다음 부팅부터)
#
# 추론 서비스는 감시 프로세스가 띄우고 지켜봄
#   도커 · 카메라가 준비되면 시작 → 응답이 30초 없으면 다시 시작
#   10분 안에 5번 넘게 다시 시작하면 멈춤 (무한 재시작 방지, 원인은 supervise.log)
#
# 설정 파일
#   deploy/camera.env              카메라 / ROI
#   deploy/controller/config.json  NUCLEO 주소(nucleo.host / port), 안내 문구, 시간
# 로그
#   추론 서비스: ~/.local/state/recycling-kiosk/inference.log
#   컨트롤러   : ~/.local/state/recycling-kiosk/kiosk.log

set -uo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"

STATE_DIR="$HOME/.local/state/recycling-kiosk"
SVC_LOG="$STATE_DIR/inference.log"
SUP_LOG="$STATE_DIR/supervise.log"
SUP_PID="$STATE_DIR/supervise.pid"
SUP_GAVE_UP="$STATE_DIR/supervise.gave_up"
mkdir -p "$STATE_DIR"

MAX_RESTARTS=5          # RESTART_WINDOW 초 안에 이만큼 넘게 다시 시작하면 멈춤
RESTART_WINDOW=600
DOWN_LIMIT=30           # 이 시간(초) 동안 응답이 없으면 다시 시작
LOAD_LIMIT=180          # 시작 후 모델 로딩을 기다리는 최대 시간(초)

CONTAINER="recycling-method1"

read -r NUCLEO_MODE NUCLEO_HOST NUCLEO_PORT < <(python3 - <<'EOF'
import json
n = json.load(open("deploy/controller/config.json", encoding="utf-8")).get("nucleo", {})
print(n.get("mode", "mock"), n.get("host", "192.168.0.120"), n.get("port", 5000))
EOF
)

step() { echo; echo "=== $* ==="; }

tcp_ok() { timeout 2 bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null; }

service_up() { curl -s -m 1 http://127.0.0.1:8765/status >/dev/null 2>&1; }

# ------------------------------------------------------------
check_nucleo() {

    step "2/4 NUCLEO 연결 확인 ($NUCLEO_MODE · $NUCLEO_HOST:$NUCLEO_PORT)"

    if [[ "$NUCLEO_MODE" != "tcp" ]]; then
        echo "mock 모드 (가짜 NUCLEO) → 실제 문은 움직이지 않음"
        return
    fi

    if tcp_ok "$NUCLEO_HOST" "$NUCLEO_PORT"; then
        echo "✅ NUCLEO 접속 가능"
        return
    fi

    # NUCLEO 를 Orin 유선 포트에 직접 꽂은 경우: Wi-Fi(192.168.0.x)와 대역이 같아
    # NUCLEO 주소 하나만 유선으로 가도록 설정 (최초 1회, sudo 비밀번호 필요)
    local carrier
    carrier="$(cat /sys/class/net/eno1/carrier 2>/dev/null || echo 0)"

    if [[ "$carrier" == "1" ]] && ! nmcli -t -f NAME con show 2>/dev/null | grep -qx "nucleo-direct"; then
        echo "유선 포트(eno1)에 케이블이 연결됨 → NUCLEO 직결 설정을 만듭니다 (sudo 비밀번호 입력)"
        sudo nmcli con add type ethernet ifname eno1 con-name nucleo-direct \
            ipv4.method manual ipv4.addresses 192.168.0.200/32 \
            ipv4.routes "$NUCLEO_HOST/32" ipv4.never-default yes \
            ipv6.method disabled connection.autoconnect yes \
        && sudo nmcli con up nucleo-direct
        sleep 3
        if tcp_ok "$NUCLEO_HOST" "$NUCLEO_PORT"; then
            echo "✅ NUCLEO 접속 가능 (유선 직결)"
            return
        fi
    fi

    echo "⚠️  NUCLEO 접속 안 됨 → 화면 테스트는 그대로 진행 (normal 판정 시 '수거함 오류' 안내)"
    echo "    확인: NUCLEO 전원 / 랜선 / 'ping $NUCLEO_HOST' / NUCLEO 펌웨어가 포트 $NUCLEO_PORT 대기 중인지"
    echo "    유선 포트 상태: carrier=$carrier (1 = 케이블 연결됨)"
}

# ------------------------------------------------------------
# 추론 서비스 감시 (백그라운드)
# ------------------------------------------------------------
slog() { echo "[$(date '+%F %T')] $*" >> "$SUP_LOG"; }

supervisor_running() {
    [[ -f "$SUP_PID" ]] && kill -0 "$(cat "$SUP_PID")" 2>/dev/null
}

launch_container() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    echo "---- $(date '+%F %T') 시작 ----" >> "$SVC_LOG"
    nohup ./deploy/run_orin.sh dev >> "$SVC_LOG" 2>&1 < /dev/null &
}

# 도커 · 카메라가 준비될 때까지 대기 (부팅 직후용)
wait_ready() {
    local waited=0
    until docker info >/dev/null 2>&1 && [[ -e /dev/video0 ]] \
          && systemctl is-active --quiet nvargus-daemon; do
        (( waited % 30 == 0 )) && slog "준비 대기: docker=$(docker info >/dev/null 2>&1 && echo ok || echo no)" \
            "camera=$([[ -e /dev/video0 ]] && echo ok || echo no)" \
            "argus=$(systemctl is-active nvargus-daemon 2>/dev/null)"
        sleep 2
        waited=$((waited + 2))
    done
    slog "도커 · 카메라 준비됨 (${waited}초 대기)"
}

supervise() {
    echo $$ > "$SUP_PID"
    rm -f "$SUP_GAVE_UP"
    trap 'rm -f "$SUP_PID"; exit 0' TERM INT
    slog "감시 시작"

    wait_ready

    local restarts=() down_since="" started now
    if ! service_up; then
        launch_container
        slog "추론 서비스 시작"
    fi
    started=$(date +%s)

    while true; do
        sleep 5
        now=$(date +%s)

        if service_up; then
            down_since=""
            continue
        fi

        # 시작 직후 모델 로딩 중이면 기다림
        if (( now - started < LOAD_LIMIT )) && [[ -z "$down_since" ]] \
           && docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
            continue
        fi

        [[ -z "$down_since" ]] && down_since=$now
        (( now - down_since < DOWN_LIMIT )) && continue

        # 최근 RESTART_WINDOW 초 안의 재시작만 남김
        local kept=() t
        for t in "${restarts[@]}"; do (( now - t < RESTART_WINDOW )) && kept+=("$t"); done
        restarts=("${kept[@]}")

        if (( ${#restarts[@]} >= MAX_RESTARTS )); then
            slog "❌ ${RESTART_WINDOW}초 안에 ${MAX_RESTARTS}번 다시 시작해도 응답 없음 → 감시 멈춤 (로그: $SVC_LOG)"
            tail -n 20 "$SVC_LOG" >> "$SUP_LOG"
            touch "$SUP_GAVE_UP"
            rm -f "$SUP_PID"
            exit 1
        fi

        slog "응답 없음 ${DOWN_LIMIT}초 → 추론 서비스 다시 시작 ($(( ${#restarts[@]} + 1 ))/${MAX_RESTARTS})"
        restarts+=("$now")
        launch_container
        started=$now
        down_since=""
    done
}

start_supervisor() {
    if supervisor_running; then
        return
    fi
    setsid "$APP_DIR/deploy/run_all.sh" supervise >/dev/null 2>&1 < /dev/null &
    sleep 1
}

stop_supervisor() {
    if supervisor_running; then
        kill "$(cat "$SUP_PID")" 2>/dev/null
        echo "[all] 감시: 종료"
    fi
    rm -f "$SUP_PID"
}

# ------------------------------------------------------------
start_service() {

    step "3/4 추론 서비스 (Docker + 카메라)"

    if [[ ! -e /dev/video0 ]]; then
        echo "❌ 카메라(/dev/video0) 없음 → 카메라 연결과 Jetson-IO 설정(IMX219-C)을 확인하세요"
        return 1
    fi

    if service_up; then
        echo "이미 실행 중"
    else
        echo -n "모델 로딩 중"
    fi
    start_supervisor

    for _ in $(seq 1 $((LOAD_LIMIT / 2))); do
        service_up && break
        echo -n "."
        sleep 2
    done
    echo

    if service_up; then
        echo "✅ 추론 서비스 실행 중 ($(grep -a '\[source\] frame' "$SVC_LOG" | tail -1))"
    else
        echo "❌ 추론 서비스가 뜨지 않음 → 로그: tail -n 30 $SVC_LOG"
        return 1
    fi
}

# ------------------------------------------------------------
start_kiosk() {
    step "1/4 대기 슬라이드 표시"
    # 새 설정(NUCLEO 주소 등)을 읽도록 다시 시작. 추론 서비스는 나중에 켜져도 컨트롤러가 자동 연결
    ./deploy/kiosk.sh restart
}

# ------------------------------------------------------------
summary() {
    step "4/4 상태"
    sleep 3
    ./deploy/kiosk.sh status | head -2
    if service_up; then echo "추론 서비스: 실행 중"; else echo "추론 서비스: 꺼짐"; fi
    grep -aE "\[nucleo\] (연결됨|연결 끊김)" "$STATE_DIR/kiosk.log" | tail -1 | sed 's/^/NUCLEO 링크: /'
    echo
    echo "▶ 카메라 앞에 물체를 두면 슬라이드 → 카메라 화면으로 바뀝니다"
    echo "▶ 판정 / 문 열기 로그 보기 : ./deploy/kiosk.sh logs"
    echo "▶ 전부 끄기               : ./deploy/run_all.sh stop"
}

# ------------------------------------------------------------
case "${1:-start}" in

    start)
        start_kiosk
        check_nucleo
        start_service || { echo; echo "추론 서비스 없이 슬라이드만 표시 중입니다 (물체 감지 안 됨)"; }
        summary
        ;;

    stop)
        stop_supervisor
        ./deploy/kiosk.sh stop
        docker stop "$CONTAINER" >/dev/null 2>&1 && echo "[all] 추론 서비스: 종료"
        ;;

    supervise)
        supervise
        ;;

    # 로그인 자동 실행에서 호출 (kiosk.sh autostart): 슬라이드는 kiosk.sh 가 띄우고 여기서는 추론만
    boot)
        slog "---- 부팅 자동 실행 ----"
        start_supervisor
        ;;

    enable)
        ./deploy/kiosk.sh enable
        ;;

    disable)
        ./deploy/kiosk.sh disable
        echo "[all] 점검 모드: 다음 부팅부터 슬라이드 · 추론 모두 자동 실행 안 함 (지금 켜진 것은 './deploy/run_all.sh stop')"
        ;;

    network)
        check_nucleo
        echo
        echo "유선 포트: carrier=$(cat /sys/class/net/eno1/carrier 2>/dev/null || echo ?) (1 = 랜선 연결됨)"
        ip -4 -br addr show eno1 2>/dev/null
        ping -c 2 -W 1 "$NUCLEO_HOST" 2>&1 | tail -2
        ;;

    status)
        ./deploy/kiosk.sh status
        if service_up; then echo "추론 서비스: 실행 중"; else echo "추론 서비스: 꺼짐"; fi
        if supervisor_running; then echo "감시     : 실행 중"
        elif [[ -f "$SUP_GAVE_UP" ]]; then echo "감시     : ❌ 반복 실패로 멈춤 (tail $SUP_LOG)"
        else echo "감시     : 꺼짐"; fi
        echo "NUCLEO   : $NUCLEO_MODE $NUCLEO_HOST:$NUCLEO_PORT $(tcp_ok "$NUCLEO_HOST" "$NUCLEO_PORT" && echo '접속 가능' || echo '접속 안 됨')"
        ;;

    *)
        sed -n '2,25p' "$0"
        exit 1
        ;;
esac
