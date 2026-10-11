import json
from datetime import datetime, timezone
from pathlib import Path

def save_run(root: Path, result: dict) -> Path:
    out = root / "runs"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = result.get("mode", "run")
    # 若目标文件已存在(如同一秒内多次保存导致同名),追加递增序号以避免覆盖。
    # 保持既有命名格式 <stamp>_<mode>.json;仅冲突时变为 <stamp>_<mode>_N.json。
    path = out / f"{stamp}_{mode}.json"
    n = 1
    while path.exists():
        path = out / f"{stamp}_{mode}_{n}.json"
        n += 1
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
