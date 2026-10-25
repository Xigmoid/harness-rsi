"""The CLI, end to end, in a temporary project."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rsi.cli import main


def run(root: Path, *args: str) -> int:
    return main(["--root", str(root), *args])


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    assert run(tmp_path, "init", "--suite", "math", "--backend", "synthetic") == 0
    return tmp_path


class TestInit:
    def test_the_project_is_scaffolded(self, project):
        for name in ("rsi.yaml", "harness/gen00.yaml", "harbor/job.yaml"):
            assert (project / name).exists(), name
        assert (project / "subsets" / "aime_dev.json").exists()
        assert (project / "subsets" / "ineqmath_heldout.json").exists()

    def test_the_subsets_carry_a_hash(self, project):
        data = json.loads((project / "subsets" / "aime_dev.json").read_text())
        assert "sha256" in data
        assert data["split"] == "dev"

    def test_init_is_idempotent(self, project):
        before = (project / "harness" / "gen00.yaml").read_text()
        assert run(project, "init", "--suite", "math", "--backend", "synthetic") == 0
        assert (project / "harness" / "gen00.yaml").read_text() == before

    def test_the_job_template_only_pins_the_agent_config(self, project):
        import yaml

        job = yaml.safe_load((project / "harbor" / "job.yaml").read_text())
        assert job["agents"][0]["kwargs"]["config"]["agent"]["agent_class"] == (
            "rsi.rhi_agent.RhiAgent"
        )
        # The retry policy must be explicit: Harbor's default is no retries.
        assert job["retry"]["max_retries"] >= 1


class TestValidate:
    def test_the_generated_spec_validates(self, project):
        assert run(project, "validate", "harness/gen00.yaml") == 0

    def test_an_invalid_spec_is_rejected(self, project):
        bad = project / "bad.yaml"
        bad.write_text("version: 0\nagents: []\nworkflow: []\n")
        with pytest.raises(Exception):
            run(project, "validate", str(bad))


class TestPipeline:
    def test_the_full_loop(self, project):
        assert run(project, "gen", "evaluate", "--gen", "0", "--suite", "math") == 0
        assert run(
            project, "gen", "propose", "--objective", "math", "--optimizer", "random_search"
        ) == 0
        assert (project / "harness" / "gen01.yaml").exists()
        assert run(project, "gen", "evaluate", "--gen", "1", "--suite", "math") == 0

        ledger_dir = project / "ledger"
        records = sorted(ledger_dir.glob("gen*.yaml"))
        assert len(records) == 2
        # The lineage is stored inside the spec, so a generation is
        # self-describing.
        import yaml

        gen1 = yaml.safe_load((ledger_dir / "gen001.yaml").read_text())
        assert gen1["parent"] == 0
        assert gen1["changed_fields"], "the diff against the parent must be recorded"

    def test_the_report_is_generated(self, project):
        run(project, "gen", "evaluate", "--gen", "0", "--suite", "math")
        assert run(project, "report") == 0
        report = (project / "ledger" / "report.md").read_text()
        assert "# RSI experiment report" in report
        assert "Transfer matrix" in report

    def test_two_chains_can_branch_from_the_same_parent(self, project):
        """The specialised mode runs one chain per capability, all from the
        same baseline -- so two proposals may legitimately share a parent.

        Regression: numbering a generation by ``parent + 1`` made the second
        chain silently overwrite the first one's spec.
        """
        import yaml

        assert run(
            project,
            "gen",
            "propose",
            "--objective",
            "math",
            "--parent",
            "0",
            "--optimizer",
            "random_search",
        ) == 0
        assert run(
            project,
            "gen",
            "propose",
            "--objective",
            "reasoning",
            "--parent",
            "0",
            "--optimizer",
            "random_search",
        ) == 0

        specs = sorted(project.glob("harness/gen*.yaml"))
        names = [p.name for p in specs if p.name != "base_config.yaml"]
        assert names == ["gen00.yaml", "gen01.yaml", "gen02.yaml"]

        gen1 = yaml.safe_load((project / "harness" / "gen01.yaml").read_text())
        gen2 = yaml.safe_load((project / "harness" / "gen02.yaml").read_text())
        assert gen1["meta"]["parent"] == 0
        assert gen2["meta"]["parent"] == 0
        assert gen1["meta"]["objective"] == "math"
        assert gen2["meta"]["objective"] == "reasoning"

    def test_a_proposal_never_overwrites_an_existing_spec(self, project):
        assert run(
            project,
            "gen",
            "propose",
            "--objective",
            "math",
            "--optimizer",
            "random_search",
        ) == 0
        before = (project / "harness" / "gen01.yaml").read_text()
        # proposing again must land on a *new* generation, not on gen01
        assert run(
            project,
            "gen",
            "propose",
            "--objective",
            "math",
            "--optimizer",
            "random_search",
        ) == 0
        assert (project / "harness" / "gen02.yaml").exists()
        assert (project / "harness" / "gen01.yaml").read_text() == before

    def test_an_unknown_parent_is_rejected(self, project):
        with pytest.raises(SystemExit):
            run(project, "gen", "propose", "--objective", "math", "--parent", "99")

    def test_two_chains_with_the_same_seed_diverge(self, project):
        """The seed is per *chain*, not per run.

        With the seed used as-is, two chains that branch from the same
        baseline drew the same edit -- the "four specialised harnesses" would
        have been one harness four times.
        """
        import yaml

        for objective in ("math", "reasoning"):
            assert run(
                project,
                "gen",
                "propose",
                "--objective",
                objective,
                "--parent",
                "0",
                "--seed",
                "11",
                "--optimizer",
                "random_search",
            ) == 0

        gen1 = yaml.safe_load((project / "harness" / "gen01.yaml").read_text())
        gen2 = yaml.safe_load((project / "harness" / "gen02.yaml").read_text())
        assert gen1["meta"]["changed_fields"] != gen2["meta"]["changed_fields"] or (
            gen1.get("agents") != gen2.get("agents")
            or gen1.get("workflow") != gen2.get("workflow")
        )

    def test_a_chain_is_reproducible(self, project):
        """The same seed *and* objective must always give the same proposal."""
        import yaml

        run(project, "gen", "propose", "--objective", "math", "--parent", "0", "--seed", "5")
        first = (project / "harness" / "gen01.yaml").read_text()
        run(
            project,
            "gen",
            "propose",
            "--objective",
            "math",
            "--parent",
            "0",
            "--seed",
            "5",
            "--out",
            str(project / "harness" / "repeat.yaml"),
        )
        repeat = (project / "harness" / "repeat.yaml").read_text()
        a = yaml.safe_load(first)
        b = yaml.safe_load(repeat)
        a["meta"] = b["meta"] = None
        assert a == b

    def test_the_transfer_analysis_runs(self, project):
        run(project, "gen", "evaluate", "--gen", "0", "--suite", "math")
        assert run(project, "analyze", "transfer") == 0

    def test_the_noise_floor_runs(self, project):
        assert run(
            project, "noise-floor", "--gen", "0", "--subset", "aime_dev", "--seeds", "1", "2"
        ) == 0
        assert (project / "runs" / "noise_floor" / "gen000" / "noise_floor.json").exists()

    def test_the_spec_diff_command(self, project):
        run(project, "gen", "propose", "--objective", "math")
        assert run(project, "spec", "diff", "0", "1") == 0

    def test_a_single_subset_evaluation(self, project):
        assert run(project, "gen", "evaluate", "--gen", "0", "--subset", "aime_dev") == 0

    def test_a_tagged_evaluation_does_not_overwrite_the_home_one(self, project):
        """A transfer probe must not destroy the home measurement.

        The probe measures the same (generation, subset) pair with fewer
        seeds.  Without a tag the two share one path, and the single-seed
        result overwrites the three-seed one -- taking the noise floor,
        which is measured on the home run, with it.
        """
        import json

        home = run(project, "gen", "evaluate", "--gen", "0", "--subset", "aime_dev")
        assert home == 0
        home_path = project / "runs" / "gen000" / "aime_dev" / "result.json"
        assert home_path.is_file()
        before = json.loads(home_path.read_text(encoding="utf-8"))

        tagged = run(
            project,
            "gen",
            "evaluate",
            "--gen",
            "0",
            "--subset",
            "aime_dev",
            "--tag",
            "transfer",
        )
        assert tagged == 0
        tagged_path = project / "runs" / "gen000" / "aime_dev.transfer" / "result.json"
        assert tagged_path.is_file()
        # The home measurement is untouched.
        after = json.loads(home_path.read_text(encoding="utf-8"))
        assert after["n_runs"] == before["n_runs"]
        assert after["mean_score"] == before["mean_score"]

    def test_a_tagged_evaluation_gets_its_own_ledger_entry(self, project):
        import yaml

        run(project, "gen", "evaluate", "--gen", "0", "--subset", "aime_dev")
        run(
            project,
            "gen",
            "evaluate",
            "--gen",
            "0",
            "--subset",
            "aime_dev",
            "--tag",
            "transfer",
        )
        record = yaml.safe_load(
            (project / "ledger" / "gen000.yaml").read_text(encoding="utf-8")
        )
        keys = set(record["evaluations"])
        assert "aime_dev" in keys
        assert "aime_dev.transfer" in keys
