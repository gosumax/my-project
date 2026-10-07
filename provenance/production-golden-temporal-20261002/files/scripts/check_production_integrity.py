"""CI/local structural and fingerprint check; does not load OCR models or run video."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from parser_core.production_integrity import load_manifest,validate_integrity
def main():
    manifest=load_manifest(ROOT)
    validate_integrity(ROOT,manifest,strict_hashes=True)
    print('PRODUCTION INTEGRITY PASS: '+manifest['production_version'])
if __name__=='__main__':main()
