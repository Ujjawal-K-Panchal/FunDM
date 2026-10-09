"""CPU checks for automatic CoderForge holdout preparation; no model downloads."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as reproduction


class HoldoutPreparationTests(unittest.TestCase):
    def setUp(self):
        self.previous_cwd = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(os.chdir, self.previous_cwd)
        os.chdir(self.temporary.name)
        self.tasks = json.loads((ROOT / 'results/coderforge/tasks.json').read_text())
        source = Path('results/coderforge/tasks.json')
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps(self.tasks))
        self.holdout = Path(reproduction.EMB) / 'holdout.json'
        self.export = reproduction.STEPS['cf_holdout']
        self.commands = [[sys.executable, *cmd[1:]] for cmd in self.export.cmds]

    def test_holdout_export_matches_published_split_and_is_reused(self):
        with patch.object(self.export, 'cmds', self.commands), \
             contextlib.redirect_stdout(io.StringIO()):
            reproduction.ensure('cf_holdout')
        self.assertEqual(json.loads(self.holdout.read_text()), self.tasks['holdout'])
        self.assertFalse(set(self.tasks['holdout']) & set(self.tasks['train_pool']))
        with patch.object(reproduction.subprocess, 'run') as run:
            reproduction.ensure('cf_holdout')
        run.assert_not_called()

    def test_cached_embeddings_still_prepare_holdout_before_finetuning(self):
        for output in reproduction.STEPS['cf_embed'].out:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for seed in reproduction.SEEDS:
            self.assertIn('cf_holdout', reproduction.STEPS[f'cf_clm_ft_s{seed}'].deps)
        original = reproduction.STEPS['cf_clm_ft_s0']
        # Run a CPU stand-in at the actual fine-tuning boundary. It must find
        # the correct holdout file even though all embedding outputs existed.
        probe = (
            'import json; from pathlib import Path; '
            f'assert json.loads(Path({str(self.holdout)!r}).read_text()) == '
            "json.loads(Path('results/coderforge/tasks.json').read_text())['holdout']; "
            f'[Path(p).write_text("done") for p in {original.out!r}]'
        )
        substitute = reproduction.Step(original.out, [[sys.executable, '-c', probe]],
                                       deps=original.deps)
        with patch.object(self.export, 'cmds', self.commands), \
             patch.dict(reproduction.STEPS, {'cf_clm_ft_s0': substitute}), \
             contextlib.redirect_stdout(io.StringIO()):
            reproduction.ensure('cf_clm_ft_s0')
        self.assertTrue(all(Path(output).exists() for output in original.out))
        self.assertFalse(Path('logs/cf_embed.log').exists())
        self.assertFalse(Path('logs/cf_data.log').exists())


if __name__ == '__main__':
    unittest.main()
