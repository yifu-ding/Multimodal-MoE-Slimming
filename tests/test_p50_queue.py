"""Mock every external entrypoint: never launch GPU jobs in regression tests."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class QueueTests(unittest.TestCase):
    def test_missing_p30_and_failed_plan_do_not_block_other_models(self):
        script = r'''
source scripts/queue_ours_p50.sh
trap - ERR
record() { echo "REPORT:$*"; }
tmux() { [[ "$1" == rename-session ]]; }
nvidia-smi() { return 0; }
sleep() { echo 'UNEXPECTED_SLEEP'; return 1; }
python() {
    [[ "$2" == stage ]] && return 0
    return 1
}
ensure_plan() {
    echo "PLAN:$2"
    [[ "$2" != qwen3-vl-30b-a3b ]]
}
bash() { echo "EVAL:$OURS_MODELS"; return 1; }
main
'''
        result = subprocess.run(['bash', '-c', script], cwd=ROOT, text=True,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn('EVAL:qwen3', result.stdout)
        self.assertIn('EVAL:kimi', result.stdout)
        self.assertIn('EVAL:internvl3_5-30b-a3b', result.stdout)
        self.assertNotIn('UNEXPECTED_SLEEP', result.stdout)
        self.assertNotIn('DONE：', result.stdout)


if __name__ == '__main__':
    unittest.main()
