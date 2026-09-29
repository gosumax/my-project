import av
import cv2
import json
import numpy as np
import sqlite3
from pathlib import Path
from PIL import Image

root = Path(r"C:\ParserData\runs\smoke14_full_tiny_ssd_20260929_v3")
with sqlite3.connect(root / "journal.sqlite3") as db:
    rows = db.execute("SELECT frame_id,bbox_json,crop_path FROM observations WHERE roi_type='chat' AND frame_id IN (0,1,50,499) ORDER BY frame_id LIMIT 36").fetchall()
by_frame = {}
for fid, bbox, path in rows:
    by_frame.setdefault(fid, []).append((json.loads(bbox), path))
with av.open("video/smoke_14.mp4") as video:
    for fid, frame in enumerate(video.decode(video=0)):
        if fid > max(by_frame):
            break
        if fid not in by_frame:
            continue
        rgb = frame.to_ndarray(format="rgb24")
        for (x1,y1,x2,y2), path in by_frame[fid]:
            old = cv2.imread(str(root / path), cv2.IMREAD_GRAYSCALE)
            new = cv2.cvtColor(rgb[y1:y2,x1:x2], cv2.COLOR_RGB2GRAY)
            pil = np.asarray(frame.to_image().crop((x1,y1,x2,y2)).convert('L'))
            png_rgb = cv2.imread(str(root / path), cv2.IMREAD_COLOR)
            rgb_match = np.array_equal(png_rgb, cv2.cvtColor(rgb[y1:y2,x1:x2], cv2.COLOR_RGB2BGR))
            if fid == 0:
                print(Image.open(root / path).mode, Image.open(root / path).info)
                r,g,b = [rgb[y1:y2,x1:x2,i].astype(np.int32) for i in range(3)]
                trials = [(r*299+g*587+b*114+500)//1000,
                          (r*299+g*587+b*114)//1000,
                          (r*19595+g*38470+b*7471)//65536,
                          (r*77+g*150+b*29+128)//256,
                          (r*19595+g*38470+b*7471+32768)//65536]
                png_cvt = cv2.cvtColor(png_rgb, cv2.COLOR_BGR2GRAY)
                ok, packed=cv2.imencode('.png', cv2.cvtColor(rgb[y1:y2,x1:x2],cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION,0])
                ram_gray=cv2.imdecode(packed,cv2.IMREAD_GRAYSCALE)
                print(fid, rgb_match, np.array_equal(old,ram_gray), np.array_equal(old,pil), [np.count_nonzero(old!=t) for t in trials])
                if np.count_nonzero(old!=trials[1]):
                    yy,xx=np.where(old!=trials[1]); print([(int(r[y,x]),int(g[y,x]),int(b[y,x]),int(old[y,x]),int(trials[1][y,x])) for y,x in zip(yy[:8],xx[:8])])
