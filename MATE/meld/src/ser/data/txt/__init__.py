from __future__ import annotations

"""Text modality dataloading.

This package mirrors the existing audio/video folders:
- txt_npy.py: dataset + collate for per-utterance .npy features
- txt_dataloaders.py: DataLoader builders used by the global dispatcher

Batch keys follow the repo convention:
  x_txt, x_txt_mask, y
"""

__all__ = []
