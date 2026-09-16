"""Check the measured layer in a partial calibration artifact."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from validate_scores_artifact import tensor_values_finite
from validate_modality_scores import validate_modality_scores

root = Path(sys.argv[1])
layer = int(sys.argv[2])
memory = json.loads((root / "memory.json").read_text())
payload = torch.load(root / "scores.pt", map_location="cpu", weights_only=False)
metadata = payload["metadata"]
assert metadata["batch_size"] == 2
assert metadata["selected_num_samples"] == 8
assert metadata["score_token_budget"] == memory["score_token_budget"]
assert metadata["selection_manifest_sha256"] == hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
assert memory["backward_passes"] == 4 and memory["attention"] == "flash_attention_2"
assert set(payload["layerwise_loss"]) == {layer}
assert float(payload["layerwise_loss"][layer]) > 0
validate_modality_scores(payload, [layer])
assert memory.get("routing_mask_checks") == 4, "Missing per-batch routed-mask validation"
for values in [payload["layerwise_loss"], payload["layerwise_second_order_sum"],
               payload["ema_matrix"], payload["channel_scores"]["gateup_act"],
               payload["expert_scores"]["second_attr"]]:
    assert tensor_values_finite(values[layer])
assert all(math.isfinite(memory[k]) for k in ["peak_allocated_gib", "peak_reserved_gib", "total_gib"])
print(f"PASS L{layer}: 8 original samples, 4 backwards, finite scores; peak allocated={memory['peak_allocated_gib']:.2f} GiB, reserved={memory['peak_reserved_gib']:.2f} GiB")
