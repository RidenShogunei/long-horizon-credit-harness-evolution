"""Exercise actual visibility boundaries and source identity, without model calls."""
import importlib.util
import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

try:
    from research1.data_pipeline import framework_context as fc
except ImportError:
    _path = Path(__file__).with_name("framework_context.py")
    _spec = importlib.util.spec_from_file_location("framework_context_patch", _path)
    fc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(fc)


def _write(root, path, source):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def _make_runtime(tmp_path):
    _write(tmp_path, fc._PROTOCOLS, "\n\n".join("class " + name + ":\n    pass" for name in fc._PROTOCOL_SYMBOLS))
    assignments = "\n".join("    " + name + " = object()" for name in ("state", "ctx", "agent_loop_spec", "tools_spec", "context_mgmt_spec", "agent_loop", "observation", "context_mgmt", "tools", "verification"))
    _write(tmp_path, fc._KERNEL, "async def run_task():\n" + assignments + "\n    try:\n        await tools.setup(ctx)\n        initial_obs = await observation.capture(ctx)\n        initial_prompt = tools.format_initial_prompt(instruction, ctx)\n        await agent_loop.run(ctx)\n    finally:\n        await tools.teardown(ctx)\n")
    _write(tmp_path, fc._AGENT, "class Terminus2Modular:\n    def __init__(self, parser_name='json', enable_summarize=True, proactive_summarization_threshold=8000, tmux_pane_width=160, tmux_pane_height=40, interleaved_thinking=False, api_key='DO_NOT_COPY_CREDENTIAL_DEFAULT'):\n        pass\n")
    _write(tmp_path, fc._STATIC, "DEFAULT_BUNDLE = 'baseline'\nclass StaticComposer:\n    async def choose(self):\n        return DEFAULT_BUNDLE\n")
    _write(tmp_path, fc._TMUX, "class TmuxSession:\n    async def send_keys(self):\n        return None\n    async def capture_pane(self):\n        return ''\n    async def get_incremental_output(self):\n        return ''\n")
    _write(tmp_path, fc._PARSER['json'], "class Parser:\n    pass\n")
    _write(tmp_path, fc._TEMPLATE['json'], "Visible grammar only\n")
    return tmp_path


def _protocol():
    return {"runtime": {"max_turns": 200, "wall_seconds": 900, "disk_floor_gb": 15},
            "policies": {"solver": {"temperature": .7, "reasoning_effort": "none"}},
            "api_key": "DO_NOT_COPY_KEY", "reward": {"tasks": ["DO_NOT_COPY_TASK"]},
            "future_trajectory": "DO_NOT_COPY_FUTURE", "environment": {"PASSWORD": "DO_NOT_COPY_ENV"}}


def _spec():
    return {"role": "proposal", "max_turns": 200, "wall_seconds": 900,
        "composer": {"name": "static", "scope": "all"}, "temperature": .7, "reasoning_effort": "none",
        "model_info": {"max_input_tokens": 176000, "max_output_tokens": 16000, "cost": "DO_NOT_COPY_COST"},
        "child_patch": "DO_NOT_COPY_PATCH", "score": 1, "api_key": "DO_NOT_COPY_SPEC_KEY"}


def _case_only_interface_and_allowlisted_runtime_values_are_visible(runtime, monkeypatch):
    monkeypatch.setenv("SECRET", "DO_NOT_COPY_PROCESS_ENV")
    _write(runtime, "tasks/secret/solution.py", "DO_NOT_COPY_PRIVATE_SOURCE")
    context = fc.build_framework_context(runtime, _protocol(), evaluation_spec=_spec())
    fc.validate_framework_context(context)
    text = json.dumps(context)
    assert "DO_NOT_COPY" not in text
    assert context["runtime_configuration"]["limits"] == {"max_turns": 200, "wall_seconds": 900}
    assert len(context["source_file_hashes"]) == 7
    assert all(row["access"] == "read_only" for row in context["source_fragments"])
    assert "src/harbor/agents/terminus_2_modular/protocols.py" not in context["editing_scope"]["editable_files"]


def _case_unchanged_build_is_identical_and_source_changes_are_detected(runtime):
    first = fc.build_framework_context(runtime, runtime_limits={"max_turns": 200, "wall_seconds": 900})
    assert first == fc.build_framework_context(runtime, _protocol())
    path = runtime / fc._PROTOCOLS
    path.write_text(path.read_text() + "\n# extra contract revision\n")
    changed = fc.build_framework_context(runtime, _protocol())
    assert changed["context_id"] != first["context_id"]
    assert changed["source_file_hashes"][fc._PROTOCOLS] != first["source_file_hashes"][fc._PROTOCOLS]


def _case_frozen_fragment_mutation_and_scope_expansion_are_rejected(runtime):
    context = fc.build_framework_context(runtime, _protocol())
    context["source_fragments"][0]["text"] += "mutable"
    with _raises(ValueError, match="changed"):
        fc.validate_framework_context(context)
    context = fc.build_framework_context(runtime, _protocol())
    context["editing_scope"]["editable_files"].append(fc._PROTOCOLS)
    with _raises(ValueError):
        fc.validate_framework_context(context)


def _case_reward_dev_or_mismatched_runtime_specs_cannot_supply_context(runtime):
    for role in ("reward", "dev"):
        spec = _spec()
        spec["role"] = role
        with _raises(ValueError, match="proposal"):
            fc.build_framework_context(runtime, _protocol(), evaluation_spec=spec)
    spec = _spec()
    spec["max_turns"] = 300
    with _raises(ValueError, match="differs"):
        fc.build_framework_context(runtime, _protocol(), evaluation_spec=spec)


def _case_missing_required_runtime_call_fails_closed(runtime):
    path = runtime / fc._KERNEL
    path.write_text(path.read_text().replace("await tools.teardown(ctx)", "pass"))
    with _raises(ValueError, match="lifecycle"):
        fc.build_framework_context(runtime, _protocol())


def _case_unlisted_paths_and_symlinked_runtime_source_are_rejected(runtime, tmp_path):
    with _raises(ValueError, match="allowlist"):
        fc._read(runtime, "../credential.txt")
    path = runtime / fc._PROTOCOLS
    original = path.read_text()
    path.unlink()
    outside = tmp_path / "outside_source.py"
    outside.write_text(original)
    try:
        path.symlink_to(outside)
    except OSError:
        if os.name == "nt":
            raise unittest.SkipTest("Windows account cannot create symlinks")
        raise
    with _raises(ValueError, match="symlink"):
        fc.build_framework_context(runtime, _protocol())


def _case_nonliteral_constructor_defaults_are_not_executed(runtime):
    path = runtime / fc._AGENT
    path.write_text(path.read_text().replace("parser_name='json'", "parser_name=__import__('os').getenv('SECRET')"))
    with _raises(ValueError, match="literal"):
        fc.build_framework_context(runtime, _protocol())


def _case_extra_fields_rejected_even_with_recomputed_context_identity(runtime, location):
    context = fc.build_framework_context(runtime, _protocol(), evaluation_spec=_spec())
    target = {"top": context, "limits": context["runtime_configuration"]["limits"],
        "defaults": context["runtime_configuration"]["constructor_defaults"],
        "sampling": context["runtime_configuration"]["sampling"], "fragment": context["source_fragments"][0]}[location]
    target["environment"] = {"TOKEN": "must not enter input"}
    context["context_id"] = "framework_" + fc._sha(fc._canonical({key: value for key, value in context.items() if key != "context_id"}).encode())
    with _raises(ValueError):
        fc.validate_framework_context(context)



def _raises(exception, match=None):
    assertion = unittest.TestCase()
    return assertion.assertRaises(exception) if match is None else assertion.assertRaisesRegex(exception, match)


class _EnvironmentPatch:
    def __init__(self, testcase):
        self.testcase = testcase

    def setenv(self, name, value):
        started = patch.dict(os.environ, {name: value})
        started.start()
        self.testcase.addCleanup(started.stop)


class FrameworkContextTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.runtime = _make_runtime(self.tmp_path / "runtime")

    def test_only_interface_and_allowlisted_runtime_values_are_visible(self):
        _case_only_interface_and_allowlisted_runtime_values_are_visible(self.runtime, _EnvironmentPatch(self))

    def test_unchanged_build_is_identical_and_source_changes_are_detected(self):
        _case_unchanged_build_is_identical_and_source_changes_are_detected(self.runtime)

    def test_frozen_fragment_mutation_and_scope_expansion_are_rejected(self):
        _case_frozen_fragment_mutation_and_scope_expansion_are_rejected(self.runtime)

    def test_reward_dev_or_mismatched_runtime_specs_cannot_supply_context(self):
        _case_reward_dev_or_mismatched_runtime_specs_cannot_supply_context(self.runtime)

    def test_missing_required_runtime_call_fails_closed(self):
        _case_missing_required_runtime_call_fails_closed(self.runtime)

    def test_unlisted_paths_and_symlinked_runtime_source_are_rejected(self):
        _case_unlisted_paths_and_symlinked_runtime_source_are_rejected(self.runtime, self.tmp_path)

    def test_nonliteral_constructor_defaults_are_not_executed(self):
        _case_nonliteral_constructor_defaults_are_not_executed(self.runtime)

    def test_extra_top_field_rejected_even_with_recomputed_identity(self):
        _case_extra_fields_rejected_even_with_recomputed_context_identity(self.runtime, "top")

    def test_extra_limit_field_rejected_even_with_recomputed_identity(self):
        _case_extra_fields_rejected_even_with_recomputed_context_identity(self.runtime, "limits")

    def test_extra_default_field_rejected_even_with_recomputed_identity(self):
        _case_extra_fields_rejected_even_with_recomputed_context_identity(self.runtime, "defaults")

    def test_extra_sampling_field_rejected_even_with_recomputed_identity(self):
        _case_extra_fields_rejected_even_with_recomputed_context_identity(self.runtime, "sampling")

    def test_extra_fragment_field_rejected_even_with_recomputed_identity(self):
        _case_extra_fields_rejected_even_with_recomputed_context_identity(self.runtime, "fragment")


if __name__ == "__main__":
    unittest.main()
