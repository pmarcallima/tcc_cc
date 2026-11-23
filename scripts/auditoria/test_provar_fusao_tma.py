import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COLLECTION = ROOT / "artifacts/prova_fusao_tma/20261103T182123Z"
VERIFIER = ROOT / "scripts/analise/provar_fusao_tma.py"


class ProofCoverageTests(unittest.TestCase):
    def test_requires_both_distinct_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "coleta"
            shutil.copytree(COLLECTION, root)
            manifest = root / "results.json"
            result = json.loads(manifest.read_text())
            small, persistent = result["cases"]
            for description, cases, valid in (
                ("ordem invertida", [persistent, small], True),
                ("caso repetido", [small, small], False),
                ("formato repetido com nomes distintos",
                 [small, dict(persistent, shape_mnk=small["shape_mnk"])], False),
            ):
                with self.subTest(description):
                    result["cases"] = cases
                    manifest.write_text(json.dumps(result))
                    run = subprocess.run([sys.executable, VERIFIER, root], capture_output=True)
                    audit = json.loads((root / "proof_verification.json").read_text())
                    self.assertEqual(run.returncode == 0, valid)
                    self.assertEqual(audit["all_passed"], valid)
                    self.assertEqual((root / "proof_viewer.html").exists(), valid)
