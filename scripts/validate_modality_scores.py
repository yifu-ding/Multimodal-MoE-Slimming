"""Reject missing/empty modality statistics, including finite all-zero artifacts."""
import torch


def vector(value):
    if isinstance(value, dict):
        return torch.cat([vector(value[k]) for k in sorted(value, key=lambda k: int(k))])
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def validate_modality_scores(payload, layers):
    for layer in layers:
        counts = []
        for modality in ("text", "visual"):
            try:
                count = vector(payload["expert_scores"][f"token_count_{modality}"][layer])
                activation = vector(payload["channel_scores"][f"gateup_act_{modality}"][layer])
            except KeyError as exc:
                raise ValueError(f"L{layer} missing modality scores: {exc}") from exc
            if not torch.isfinite(count).all() or (count < 0).any() or count.sum() <= 0:
                raise ValueError(f"L{layer} token_count_{modality} missing/zero/invalid")
            if not torch.isclose(count.sum(), torch.tensor(1.0), atol=1e-4):
                raise ValueError(f"L{layer} {modality} routing counts are not normalized")
            if not torch.isfinite(activation).all() or not activation.count_nonzero():
                raise ValueError(f"L{layer} gateup_act_{modality} missing/zero/invalid")
            counts.append(count)
        actual = vector(payload["ema_matrix"][layer])
        text, visual = counts
        expected = (visual - text) / (visual + text + 1e-8)
        if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4):
            raise ValueError(f"L{layer} EMA does not match modality routing counts")
    return True
