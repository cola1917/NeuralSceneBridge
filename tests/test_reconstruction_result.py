import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCHEMAS = (
    ROOT.parent
    / "SceneExchangeContracts"
    / "src"
    / "scene_exchange_contracts"
    / "schemas"
    / "shared_exchange_protocol"
)


def _schema_example(name):
    return copy.deepcopy(json.loads((SCHEMAS / name).read_text(encoding="utf-8"))["examples"][0])


class ReconstructionResultTests(unittest.TestCase):
    def test_builds_schema_valid_result_from_token_request(self):
        from build_reconstruction_result import build_reconstruction_result
        from scene_exchange_contracts import validate_shared_document

        request = _schema_example("reconstruction_request.schema.json")
        references = _schema_example("reconstruction_result.schema.json")["payload"]["artifacts"]
        package = next(item for item in references if item["role"] == "reconstruction_package")
        others = [item for item in references if item["role"] != "reconstruction_package"]
        result = build_reconstruction_result(
            request,
            reconstruction_package_reference=package,
            artifact_references=others,
            started_at="2026-07-13T12:00:05Z",
            finished_at="2026-07-13T12:20:00Z",
            producer={"project": "NeuralSceneBridge", "component": "test", "version": "test"},
        )
        validate_shared_document(result)
        self.assertEqual(result["payload"]["scene_id"], request["payload"]["scene_id"])
        self.assertEqual(result["payload"]["product_status"]["reconstruction_package"], "succeeded")
        self.assertNotIn("scene_package", result["payload"]["product_status"])

    def test_rejects_non_token_converter_addressing(self):
        from build_reconstruction_result import build_reconstruction_result

        request = _schema_example("reconstruction_request.schema.json")
        request["payload"]["source"]["scene_key_kind"] = "name"
        references = _schema_example("reconstruction_result.schema.json")["payload"]["artifacts"]
        package = next(item for item in references if item["role"] == "reconstruction_package")
        with self.assertRaises(ValueError):
            build_reconstruction_result(
                request,
                reconstruction_package_reference=package,
                artifact_references=[],
                started_at="2026-07-13T12:00:05Z",
                finished_at="2026-07-13T12:20:00Z",
                producer={"project": "NeuralSceneBridge", "component": "test", "version": "test"},
            )


if __name__ == "__main__":
    unittest.main()
