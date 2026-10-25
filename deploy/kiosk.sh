#!/usr/bin/env bash
# 키오스크 관리 (Orin 호스트): 컨트롤러 + 모니터 전체화면 브라우저
#
#   ./deploy/kiosk.sh start      켜기 (이미 켜져 있으면 그대로)
#   ./deploy/kiosk.sh stop       끄기 (수정 작업할 때)
#   ./deploy/kiosk.sh restart    코드 / 설정 수정 후 다시 시작
#   ./deploy/kiosk.sh status     실행 상태 확인
#   ./deploy/kiosk.sh enable     로그인하면 자동 실행 (기본)
#   ./deploy/kiosk.sh disable    자동 실행 끄기 (다음 부팅부터 적용, 지금 켜진 것은 stop 으로)
#   ./deploy/kiosk.sh logs       컨트롤러 로그 보기 (Ctrl+C 로 나가기)
#
# 화면에서 직접 작업: 키보드 Alt+F4 → 브라우저 닫힘, Ctrl+Alt+T → 터미널
# sudo 불필요 (사용자 자동 시작 ~/.config/autostart 사용)

set -uo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$HOME/.local/state/recycling-kiosk"
LOG="$STATE_DIR/kiosk.log"
PGID_FILE="$STATE_DIR/controller.pgid"
AUTOSTART="$HOME/.config/autostart/recycling-kiosk.desktop"
URL="http://localhost:8080/"

# snap Firefox 는 홈의 숨김 폴더에 접근할 수 없어 ~/snap/firefox/common 아래에 키오스크 전용 프로필을 둠
FF_PROFILE="$HOME/snap/firefox/common/recycling-kiosk-profile"

mkdir -p "$STATE_DIR"

say() { echo "[kiosk] $*"; }

# ------------------------------------------------------------
# 모니터 화면 찾기 (SSH 에서 실행해도 로그인된 데스크톱에 띄우기)
# ------------------------------------------------------------
find_display() {
    if [[ -z "${DISPLAY:-}" ]]; then
        local sock
        sock="$(ls /tmp/.X11-unix/ 2>/dev/null | sort -V | tail -1)"
        [[ -n "$sock" ]] && export DISPLAY=":${sock#X}"
    fi
    if [[ -z "${XAUTHORITY:-}" && -f "/run/user/$(id -u)/gdm/Xauthority" ]]; then
        export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
    fi
}

# ------------------------------------------------------------
# 키오스크용 데스크톱 설정 (GNOME)
#   왼쪽 위 모서리 / Super 키로 '활동' 화면이 열려 키오스크가 작아지는 것 방지,
#   알림 배너가 키오스크 위에 뜨는 것 방지. 원래 값은 저장해 두고 disable 때 되돌림
# ------------------------------------------------------------
DESKTOP_KEYS=(
    "org.gnome.desktop.interface enable-hot-corners false"
    "org.gnome.mutter overlay-key ''"
    "org.gnome.desktop.notifications show-banners false"
)
SAVED_SETTINGS="$STATE_DIR/desktop_settings.orig"

gs() {
    DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}" gsettings "$@"
}

apply_desktop_settings() {
    command -v gsettings >/dev/null || return 0
    local entry schema key value
    if [[ ! -f "$SAVED_SETTINGS" ]]; then
        for entry in "${DESKTOP_KEYS[@]}"; do
            read -r schema key value <<< "$entry"
            echo "$schema $key $(gs get "$schema" "$key" 2>/dev/null)" >> "$SAVED_SETTINGS"
        done
    fi
    for entry in "${DESKTOP_KEYS[@]}"; do
        read -r schema key value <<< "$entry"
        gs set "$schema" "$key" "$value" 2>/dev/null
    done
}

restore_desktop_settings() {
    [[ -f "$SAVED_SETTINGS" ]] || return 0
    local schema key value
    while read -r schema key value; do
        [[ -n "$value" ]] && gs set "$schema" "$key" "$value" 2>/dev/null
    done < "$SAVED_SETTINGS"
    rm -f "$SAVED_SETTINGS"
    say "데스크톱 설정: 원래대로 (모서리 / Super 키 / 알림)"
}

# 로그인 직후 GNOME 이 '활동' 화면을 띄우면 키오스크 창이 작은 미리보기로 보임 → 닫기
close_overview() {
    DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}" \
        gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell \
        --method org.freedesktop.DBus.Properties.Set org.gnome.Shell OverviewActive "<false>" \
        >/dev/null 2>&1
}

controller_running() {
    [[ -f "$PGID_FILE" ]] && kill -0 -- "-$(cat "$PGID_FILE")" 2>/dev/null
}

port_in_use() {
    (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null
}

browser_pids() {
    pgrep -f -- "--kiosk $URL" || true
}

# ------------------------------------------------------------
# 시작 / 정지
# ------------------------------------------------------------
start_controller() {
    if controller_running; then
        say "컨트롤러: 이미 실행 중"
        return
    fi
    if port_in_use; then
        say "포트 8080 을 다른 프로그램이 쓰고 있음 (수동 실행한 start_kiosk.sh 를 먼저 Ctrl+C 로 끄세요)"
        return 1
    fi

    # 죽으면 2초 뒤 다시 실행. setsid 로 별도 그룹을 만들어 stop 때 한꺼번에 종료
    setsid bash -c "
        cd '$APP_DIR'
        while true; do
            python3 -u deploy/controller/controller.py
            echo \"[kiosk] 컨트롤러 종료됨 → 2초 뒤 재시작\"
            sleep 2
        done
    " >>"$LOG" 2>&1 < /dev/null &
    echo $! > "$PGID_FILE"

    for _ in $(seq 1 20); do
        port_in_use && break
        sleep 0.5
    done
    say "컨트롤러: 시작 (로그 $LOG)"
}

prepare_firefox_profile() {
    mkdir -p "$FF_PROFILE"
    # 첫 실행 안내 / 세션 복원 / 업데이트 알림 등 키오스크에 방해되는 창 끄기
    cat > "$FF_PROFILE/user.js" <<'EOF'
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.sessionstore.max_resumed_crashes", 0);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("browser.aboutwelcome.enabled", false);
user_pref("datareporting.policy.dataSubmissionPolicyBypassNotification", true);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("app.update.auto", false);
user_pref("browser.tabs.warnOnClose", false);
user_pref("media.autoplay.default", 0);
EOF
}

start_browser() {
    if [[ -n "$(browser_pids)" ]]; then
        say "브라우저: 이미 실행 중"
        return
    fi

    find_display
    if [[ -z "${DISPLAY:-}" ]]; then
        say "로그인된 화면이 없음 (모니터 / 자동 로그인 확인)"
        return 1
    fi

    apply_desktop_settings

    if command -v chromium-browser >/dev/null || command -v chromium >/dev/null; then
        "$(command -v chromium-browser || command -v chromium)" --kiosk --noerrdialogs \
            --disable-infobars --autoplay-policy=no-user-gesture-required "$URL" \
            >>"$LOG" 2>&1 < /dev/null &
    else
        prepare_firefox_profile
        firefox --new-instance --profile "$FF_PROFILE" --kiosk "$URL" \
            >>"$LOG" 2>&1 < /dev/null &
    fi
    disown || true

    # 브라우저가 뜨는 동안 / 로그인 직후 '활동' 화면이 열려 있으면 20초 동안 계속 닫아 줌
    ( for _ in $(seq 1 20); do close_overview; sleep 1; done ) >/dev/null 2>&1 &
    disown || true

    say "브라우저: 시작 (DISPLAY=$DISPLAY)"
}

stop_all() {
    local pids
    pids="$(browser_pids)"
    if [[ -n "$pids" ]]; then
        kill $pids 2>/dev/null
        say "브라우저: 종료"
    fi
    if controller_running; then
        kill -- "-$(cat "$PGID_FILE")" 2>/dev/null
        say "컨트롤러: 종료"
    fi
    rm -f "$PGID_FILE"
}

status() {
    if controller_running; then echo "컨트롤러 : 실행 중"; else echo "컨트롤러 : 꺼짐"; fi
    if [[ -n "$(browser_pids)" ]]; then echo "브라우저 : 실행 중"; else echo "브라우저 : 꺼짐"; fi
    if [[ -f "$AUTOSTART" ]]; then echo "자동 실행: 켜짐"; else echo "자동 실행: 꺼짐"; fi
    echo "로그     : $LOG"
}

enable_autostart() {
    mkdir -p "$(dirname "$AUTOSTART")"
    cat > "$AUTOSTART" <<EOF
[Desktop Entry]
Type=Application
Name=Recycling Kiosk
Comment=분리수거함 대기 화면 / 안내 화면
Exec=$APP_DIR/deploy/kiosk.sh autostart
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=5
NoDisplay=true
EOF
    say "자동 실행: 켜짐 (로그인 5초 뒤 시작)"
}

case "${1:-}" in
    start)     start_controller && start_browser ;;
    autostart) echo "[kiosk] ---- 로그인 자동 실행 $(date '+%F %T') ----" >>"$LOG"
               start_controller && start_browser ;;
    stop)      stop_all ;;
    restart)   stop_all; sleep 2; start_controller && start_browser ;;
    status)    status ;;
    enable)    enable_autostart ;;
    disable)   rm -f "$AUTOSTART"; restore_desktop_settings
               say "자동 실행: 꺼짐 (지금 켜진 화면은 './deploy/kiosk.sh stop')" ;;
    logs)      tail -n 50 -f "$LOG" ;;
    *)         sed -n '2,13p' "$0"; exit 1 ;;
esac
