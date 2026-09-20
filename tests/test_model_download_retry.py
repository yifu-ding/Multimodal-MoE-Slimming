import ast
from pathlib import Path
import subprocess
import time
import unittest
from unittest.mock import Mock, patch

source = Path(__file__).resolve().parents[1] / 'scripts/download_model_mirror.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fetch')
namespace = {'subprocess': subprocess, 'time': time}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
fetch = namespace['fetch']


class RetryTest(unittest.TestCase):
    def result(self, code, http='000', body=''):
        return subprocess.CompletedProcess([], code, body + '\n' + http)

    def test_ssl_then_success_preserves_resume(self):
        sleep = Mock()
        with patch.object(subprocess, 'run', side_effect=[self.result(35), self.result(0, '200', 'data')]) as run:
            self.assertEqual(fetch(['-C', '-'], Mock(), sleep=sleep), 'data')
            self.assertEqual(run.call_args_list[0], run.call_args_list[1])
        sleep.assert_called_once_with(30)

    def test_terminal_and_transient(self):
        for code, status, expected in [(22, '403', 1), (60, '000', 1), (22, '429', 2), (35, '000', 2)]:
            with patch.object(subprocess, 'run', return_value=self.result(code, status)) as run:
                with self.assertRaises(RuntimeError):
                    fetch([], Mock(), attempts=2, sleep=Mock())
                self.assertEqual(run.call_count, expected)

    def test_backoff_capped(self):
        sleep = Mock()
        with patch.object(subprocess, 'run', return_value=self.result(28)):
            with self.assertRaises(RuntimeError):
                fetch([], Mock(), attempts=8, sleep=sleep)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [30, 60, 120, 240, 480, 600, 600])


if __name__ == '__main__':
    unittest.main()
