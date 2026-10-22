"""
대기 화면용 PDF → 슬라이드 PNG 변환 (PC 에서 한 번 실행 후 Orin 으로 동기화)

    pip install pymupdf
    python deploy/tools/pdf_to_slides.py "monitor_video/Project SPARK-Up_StandbyScreen.pdf" --crop spark

--crop spark : SPARK-UP 슬라이드의 바깥 파란 여백 + 초록 테두리를 잘라내고 안쪽 회색 패널만 남김
               (초록 테두리는 키오스크가 모니터 가장자리에 다시 그림 → 초록 틀이 화면을 꽉 채움)
--crop L,T,R,B : 페이지 비율(0~1)로 직접 지정

출력: deploy/controller/media/standby/slide_01.png, slide_02.png, ...
      (기존 slide_*.png 는 지우고 새로 만듦)
키오스크는 이 폴더의 PNG 를 이름 순서대로 standby_slide_seconds 마다 넘겨 보여준다.
"""

from pathlib import Path
import argparse

import pymupdf

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "controller" / "media" / "standby"

# 1920x1080 렌더 기준 안쪽 패널 (x 82~1838, y 86~998) — 초록 테두리 안쪽 #f9f9f9 영역 (6장 모두 동일)
CROPS = {
    "spark": (82 / 1920, 86 / 1080, 1838 / 1920, 998 / 1080),
}


def main():

    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--width", type=int, default=1920, help="원본 페이지 기준 가로 픽셀 (해상도)")
    p.add_argument("--crop", default=None, help="'spark' 또는 L,T,R,B (0~1 비율)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for old in out.glob("slide_*.png"):
        old.unlink()

    doc = pymupdf.open(args.pdf)

    crop = None
    if args.crop:
        crop = CROPS.get(args.crop) or tuple(float(v) for v in args.crop.split(","))

    for i, page in enumerate(doc, 1):
        zoom = args.width / page.rect.width
        r = page.rect
        clip = None
        if crop:
            clip = pymupdf.Rect(r.x0 + r.width * crop[0], r.y0 + r.height * crop[1],
                                r.x0 + r.width * crop[2], r.y0 + r.height * crop[3])
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip)
        path = out / f"slide_{i:02d}.png"
        pix.save(path)
        print(f"{path.name}  {pix.width}x{pix.height}")

    print(f"{len(doc)} slides → {out}")


if __name__ == "__main__":
    main()
