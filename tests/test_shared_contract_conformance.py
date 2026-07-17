import sys
import unittest
from pathlib import Path


CONTRACT_SRC = Path(__file__).parents[2] / "SceneExchangeContracts" / "src"
sys.path.insert(0, str(CONTRACT_SRC))


class SharedContractConformanceTests(unittest.TestCase):
    def test_uses_canonical_contract_suite(self):
        from scene_exchange_contracts.conformance import run_conformance_suite

        self.assertEqual(run_conformance_suite()["schema_count"], 19)


if __name__ == "__main__":
    unittest.main()
