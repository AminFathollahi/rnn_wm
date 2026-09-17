import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainalign_wm.tasks.multitask import NEUROGYM_TASKS, NeuroGymBatchEnv


def test_batch_env_accepts_a_seed_beyond_32_bits():
    # continuation seeds are scaled by a large multiplier before reaching here
    NeuroGymBatchEnv(NEUROGYM_TASKS[0], 2, seed=10_000 * 1_000_003)
