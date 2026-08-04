"""The agent-facing half of the interruption taxonomy: the skill, the
always-loaded prompt section, and the startup signal.

These are content guards. The defect they exist for (feedback #37) was
not a code bug — the agent had no guidance about preemption anywhere, so
it improvised "transient cloud infrastructure issue", retried under new
run names, and told the user it was "outside our control".
"""

from __future__ import annotations

import json
from pathlib import Path


class TestJobRecoverySkill:
    def test_registered_and_aliases_resolve(self) -> None:
        from lqh.skills import list_available_skills, load_skill_content

        entry = next(
            s for s in list_available_skills() if s["name"] == "job_recovery"
        )
        assert entry["command"] == "/recover"
        content = load_skill_content("job_recovery")
        assert content.startswith("# Skill: Job Recovery")
        for alias in ("recover", "/recover", "preempted", "orphaned", "preemption"):
            assert load_skill_content(alias) == content

    def test_tui_command_wired(self) -> None:
        from lqh.tui.commands import COMMANDS

        assert any(c.name == "/recover" for c in COMMANDS)

    def test_skill_covers_the_decisions_the_agent_got_wrong(self) -> None:
        from lqh.skills import load_skill_content

        content = load_skill_content("job_recovery")
        # A resubmit is not a resume.
        assert "never \"resume\"" in content or 'never "resume"' in content
        assert "cannot be reused" in content
        # Stop rather than retry forever.
        assert "## When to stop" in content
        assert "two infrastructure failures on the same job shape" in content
        # Money, honestly.
        assert "/feedback" in content
        assert "Do not promise a refund" in content
        assert "outside our control" in content  # named as the thing NOT to say


class TestSystemPromptGuidance:
    def test_prompt_names_preemption_and_the_never_dos(self) -> None:
        from lqh.agent import SYSTEM_PROMPT

        assert "preemptible" in SYSTEM_PROMPT
        assert "step 0" in SYSTEM_PROMPT
        assert "/feedback" in SYSTEM_PROMPT
        assert "job_recovery" in SYSTEM_PROMPT
        assert "transient cloud issue" in SYSTEM_PROMPT

    def test_train_skill_no_longer_tells_the_agent_to_poll(self) -> None:
        from lqh.skills import load_skill_content

        content = load_skill_content("train")
        # This line contradicted the tool description and both hard-rule
        # skills, which is why the agent polled in the incident session.
        assert "periodically check status" not in content
        assert "never poll" in content.lower()
        assert "job_recovery" in content

    def test_auto_and_subagent_cap_retries(self) -> None:
        from lqh.skills import load_skill_content

        for name in ("auto", "subagent"):
            content = load_skill_content(name)
            assert "step 0" in content, name
            assert "exit_auto_mode" in content, name


class TestInfraFailureSignal:
    def _project(self, tmp_path: Path, *, cls: str = "orphaned") -> Path:
        run_dir = tmp_path / "runs" / "sft_003"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}\n")
        (run_dir / "cloud_failure.json").write_text(
            json.dumps({"cls": cls, "infra": cls != "oom"}) + "\n"
        )
        return tmp_path

    def test_raised_for_an_unresubmitted_infra_failure(self, tmp_path: Path) -> None:
        from lqh.signals import infra_failure_signals

        signals = infra_failure_signals(self._project(tmp_path), {"sft_003": "failed"})

        assert len(signals) == 1
        assert signals[0].kind == "infra_failure"
        assert "job_recovery" in signals[0].text
        assert "billed" in signals[0].text

    def test_not_raised_for_a_config_side_failure(self, tmp_path: Path) -> None:
        # An OOM is the user's config, not our infrastructure — it does
        # not belong in a "we cost you money" signal.
        from lqh.signals import infra_failure_signals

        project = self._project(tmp_path, cls="oom")
        assert infra_failure_signals(project, {"sft_003": "failed"}) == []

    def test_suppressed_once_a_later_run_has_started(self, tmp_path: Path) -> None:
        import os
        import time

        from lqh.signals import infra_failure_signals

        project = self._project(tmp_path)
        later = project / "runs" / "sft_004"
        later.mkdir(parents=True)
        (later / "config.json").write_text("{}\n")
        now = time.time()
        os.utime(project / "runs" / "sft_003" / "config.json", (now - 600, now - 600))
        os.utime(later / "config.json", (now, now))

        signals = infra_failure_signals(
            project, {"sft_003": "failed", "sft_004": "running"},
        )

        assert signals == []

    def test_a_later_failed_retry_supersedes_the_earlier_one(
        self, tmp_path: Path,
    ) -> None:
        """Two infra failures in a row must surface as ONE signal — the
        latest. The old mtime heuristic surfaced the earlier one too,
        because a run carrying its own cloud_failure.json never counted
        as a follow-up."""
        import json as _json
        import os
        import time

        from lqh.signals import infra_failure_signals

        project = self._project(tmp_path)
        retry = project / "runs" / "sft_004"
        retry.mkdir(parents=True)
        (retry / "config.json").write_text("{}\n")
        (retry / "cloud_failure.json").write_text(
            _json.dumps({"cls": "timeout", "infra": True}) + "\n"
        )
        now = time.time()
        os.utime(project / "runs" / "sft_003" / "config.json", (now - 600, now - 600))
        os.utime(retry / "config.json", (now, now))

        signals = infra_failure_signals(
            project, {"sft_003": "failed", "sft_004": "failed"},
        )

        assert len(signals) == 1
        assert "sft_004" in signals[0].text
        assert "sft_003" not in signals[0].text

    def test_no_runs_dir_is_not_an_error(self, tmp_path: Path) -> None:
        from lqh.signals import infra_failure_signals

        assert infra_failure_signals(tmp_path, {}) == []


class TestTaxonomyIsConsistentAcrossModes:
    """The same failure must not get different verdicts depending on
    which prompt or skill happens to be loaded. These are the three
    places that independently describe the classes."""

    def _texts(self) -> dict[str, str]:
        from lqh.agent import SYSTEM_PROMPT
        from lqh.skills import load_skill_content

        return {
            "prompt": SYSTEM_PROMPT,
            "job_recovery": load_skill_content("job_recovery"),
            "train": load_skill_content("train"),
            "auto": load_skill_content("auto"),
            "subagent": load_skill_content("subagent"),
        }

    def test_no_skill_calls_a_timeout_infrastructure(self) -> None:
        """Core classification excludes timeout from INFRA_CLASSES — a
        wall-clock cap is a budget the job outgrew. auto/ and subagent/
        used to list it under "infrastructure" anyway."""
        from lqh.remote.failure import INFRA_CLASSES

        assert "timeout" not in INFRA_CLASSES
        for name, text in self._texts().items():
            for bad in (
                "(preempted / orphaned / timeout) you get",
                "timed-out run is infrastructure",
            ):
                assert bad not in text, f"{name} calls a timeout infrastructure"

    def test_orphaned_is_never_asserted_as_a_config_clearance(self) -> None:
        """An orphan is an observation. No surface may promise the user's
        configuration is exonerated by it."""
        train = self._texts()["train"]
        assert "An **orphaned** run is an *observation*" in train
        assert "check `artifacts` and `stderr.log`" in train
