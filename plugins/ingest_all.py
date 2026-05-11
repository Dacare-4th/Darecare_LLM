import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

scripts = [
    BASE_DIR / "tricare" / "tricare_claim_preprocess.py",
    BASE_DIR / "tricare" / "tricare_guide_preprocess.py",
    BASE_DIR / "uhcg" / "uhc_guide_preprocess.py",
    BASE_DIR / "uhcg" / "uhc_claim_preprocess.py",
    BASE_DIR / "nhis" / "ingest.py",
    BASE_DIR / "msh_china" / "preprocess_msh.py",
    BASE_DIR / "msh_china" / "preprocess_msh_policy_wording.py",
    BASE_DIR / "cigna" / "ingest.py",
]

for script in scripts:
    print(f"\n{'='*50}")
    print(f"실행 중: {script.name}")
    print('='*50)
    result = subprocess.run([sys.executable, str(script)])
    if result.returncode != 0:
        print(f"[ERROR] 실패: {script.name}")
        break