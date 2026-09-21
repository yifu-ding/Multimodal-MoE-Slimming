import ast
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class CachePolicyTest(unittest.TestCase):
    def test_runtime_override(self):
        tree = ast.parse((ROOT / 'lmms-eval/lmms_eval/models/simple/vllm.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and 'MAES_DISABLE_MM_PROCESSOR_CACHE' in ast.unparse(n.test))
        code = compile(ast.Module(body=[node], type_ignores=[]), '<policy>', 'exec')
        for enabled in ['0', '1']:
            kwargs = {}
            with patch.dict(os.environ, {'MAES_DISABLE_MM_PROCESSOR_CACHE': enabled}):
                exec(code, {'os': os, 'kwargs': kwargs, 'eval_logger': Mock()})
            self.assertEqual(kwargs, {'mm_processor_cache_gb': 0} if enabled == '1' else {})

    def test_task_scope_and_answer_cache(self):
        source = (ROOT / 'scripts/run_qwen3_vl_vllm_baseline.sh').read_text()
        function = source.split('build_command() {', 1)[1].split('\n}\n', 1)[0]
        setup = '''build_command() {''' + function + '''\n}
task_uses_native_video() { return 1; }
ENABLE_RESPONSE_CACHE=1
LOG_SAMPLES=0
PYTHON_CMD=(python)
'''
        for task, flag in [('egoschema_subset_local', '1'), ('mvbench_available_3800', '1'), ('gqa', '0'), ('videomme_qwen3_vllm', '0')]:
            command = setup + f'build_command {task} 0 /tmp/output "" 8 unchanged_args /tmp/answers /tmp/heartbeat 1 semantic-id\nprintf "%s\\n" "${{CMD[@]}}"'
            args = subprocess.check_output(['bash', '-c', command], text=True).splitlines()
            self.assertIn('MAES_DISABLE_MM_PROCESSOR_CACHE=' + flag, args)
            self.assertIn('LMMS_CACHE_FINGERPRINT_SALT=semantic-id', args)
            self.assertEqual(args[args.index('--model_args') + 1], 'unchanged_args')
            self.assertEqual(args[args.index('--use_cache') + 1], '/tmp/answers')


if __name__ == '__main__':
    unittest.main()
