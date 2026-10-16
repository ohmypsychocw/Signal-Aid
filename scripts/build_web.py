"""Embed a replay into the web page, producing one self-contained file.

    python scripts/build_web.py --replay web/replay.json --out web/dist/index.html
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--replay", default="web/replay.json")
ap.add_argument("--template", default="web/template.html")
ap.add_argument("--out", default="web/dist/index.html")
ap.add_argument("--basemap", default="data/basemap.json")
a = ap.parse_args()

replay = json.loads(Path(a.replay).read_text(encoding="utf-8"))
if Path(a.basemap).exists():
    replay["basemap"] = json.loads(Path(a.basemap).read_text(encoding="utf-8"))
blob = json.dumps(replay, ensure_ascii=False).replace("</", "<\\/")
html = Path(a.template).read_text(encoding="utf-8").replace("/*REPLAY_JSON*/", blob)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size // 1024} KB, mock={replay['meta'].get('mock_decider')})")
