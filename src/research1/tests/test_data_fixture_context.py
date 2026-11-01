"""Read-only original fixture visibility and current executed-code identity."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from research1.data_pipeline import fixture_context as fc
except ImportError:
    for parent in Path(__file__).resolve().parents:
        if (parent / "server_fix" / "src").is_dir():
            sys.path.insert(0, str(parent / "server_fix" / "src"))
            break
    specification = importlib.util.spec_from_file_location("offline_fixture_context", Path(__file__).with_name("fixture_context.py"))
    fc = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(fc)

from research1.data_pipeline import modules
from research1.data_pipeline.util import digest


class FixtureContextTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "original_campaign" / "source_snapshot" / fc.RELATIVE_SOURCE
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(Path(modules.__file__).read_bytes())
        self.protocol_path = self.root / "original_campaign" / "protocol.json"
        self.protocol = {"source_sha256": {fc.RELATIVE_SOURCE: hashlib.sha256(self.path.read_bytes()).hexdigest()},
                         "registry_sha256": digest(modules.frozen_registry())}
        self.write_protocol()

    def write_protocol(self):
        self.protocol["protocol_hash"] = digest({key: value for key, value in self.protocol.items() if key != "protocol_hash"})
        self.protocol_path.write_text(json.dumps(self.protocol, indent=2) + "\n", encoding="utf-8")

    def test_original_exact_source_and_all_relevant_functions_are_visible_read_only(self):
        original = self.path.read_bytes()
        with patch("research1.data_pipeline.modules.probe_module", wraps=modules.probe_module) as probe:
            # A spy wrapper changes executable identity and must be rejected;
            # the normal positive case below never invokes this function.
            with self.assertRaisesRegex(ValueError, "inspectable|differs"):
                fc.freeze_probe_contract(self.path)
            probe.assert_not_called()
        context = fc.freeze_probe_contract(self.path)
        self.assertEqual(fc.verify_probe_contract(context), context)
        self.assertEqual(context["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual({row["symbol"] for row in context["fragments"]}, set(fc.SYMBOLS))
        self.assertTrue(all(row["access"] == "read_only" for row in context["fragments"]))
        self.assertIn("completion_requirements_rechecked", context["text"])
        self.assertIn("completion_prompt_bounded", context["text"])
        self.assertIn("<= 8192", context["text"])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertIn("not", " ".join(context["limitations"]))

    def test_unfrozen_source_and_source_byte_changes_are_rejected(self):
        self.path.write_bytes(self.path.read_bytes() + b"\n# modified fixture\n")
        with self.assertRaisesRegex(ValueError, "protocol-frozen"):
            fc.freeze_probe_contract(self.path)

    def test_changing_source_manifest_cannot_hide_current_execution_mismatch(self):
        self.path.write_bytes(self.path.read_bytes().replace(b"<= 8192", b"<= 8193"))
        self.protocol["source_sha256"][fc.RELATIVE_SOURCE] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, "executed probe module file differs"):
            fc.freeze_probe_contract(self.path)

    def test_same_file_with_monkeypatched_loaded_function_is_rejected(self):
        with patch.object(modules, "_probe_in_process", lambda root, kind: {"checks": {}}):
            with self.assertRaisesRegex(ValueError, "Loaded executing fixture code differs"):
                fc.freeze_probe_contract(self.path)

    def test_context_text_limits_or_metadata_mutation_is_rejected(self):
        context = fc.freeze_probe_contract(self.path)
        for field in ("text", "function_sha256", "limitations"):
            with self.subTest(field=field):
                modified = copy.deepcopy(context)
                modified[field] = "changed"
                with self.assertRaisesRegex(ValueError, "context changed"):
                    fc.build_fixture_review_context(modified)

    def test_original_protocol_change_after_freezing_is_detected(self):
        context = fc.freeze_probe_contract(self.path)
        self.protocol["other_metadata"] = "changed"
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, "context changed"):
            fc.verify_probe_contract(context)

    def test_non_snapshot_current_source_path_is_not_accepted_as_original(self):
        with self.assertRaisesRegex(ValueError, "original campaign source_snapshot"):
            fc.freeze_probe_contract(Path(modules.__file__).resolve())


if __name__ == "__main__":
    unittest.main()
