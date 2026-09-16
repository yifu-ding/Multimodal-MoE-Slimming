import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import unattended_debug as d


class DebugTests(unittest.TestCase):
    def test_healthy_never_invokes_codex(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, 'BASE', Path(tmp)), \
                patch.object(d, 'sessions', return_value={'maes-todo-scores'}), \
                patch.object(d, 'command') as command:
            d.tick()
            command.assert_not_called()

    def test_three_failed_rounds_stop(self):
        state = {'progress': ['old']}
        for i in range(3):
            state['pending'] = True
            self.assertEqual(d.reconcile(state, ['old']), i == 2)
            self.assertEqual(state['failures'], i + 1)

    def test_real_new_artifact_resets(self):
        state = {'progress': ['old'], 'pending': True, 'failures': 2}
        self.assertFalse(d.reconcile(state, ['old', 'new']))
        self.assertEqual(state['failures'], 0)

    def test_no_progress_does_not_double_count(self):
        state = {'progress': [], 'failures': 1}
        d.reconcile(state, [])
        self.assertEqual(state['failures'], 1)

    def test_clean_auth_and_sandbox(self):
        with patch.dict(d.os.environ, {'OPENAI_API_KEY': 'fake', 'CODEX_API_KEY': 'fake'}):
            self.assertNotIn('OPENAI_API_KEY', d.clean_env())
            self.assertNotIn('CODEX_API_KEY', d.clean_env())
        args = d.codex_args(Path('/tmp/result'))
        self.assertIn('--approve-for-me', args)  # CLI-defined workspace-write + review
        self.assertIn('forced_login_method="chatgpt"', args)
        self.assertNotIn('--dangerously-bypass-approvals-and-sandbox', args)


if __name__ == '__main__':
    unittest.main()
