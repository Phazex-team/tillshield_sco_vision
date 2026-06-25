"""SAM 2 client.

When the ``sam2`` python package is importable AND the repo-local
``facebook/sam2-hiera-large`` snapshot exists, this client wraps the
official SAM 2 image-predictor for candidate-item segmentation.

When either dependency is missing, ``has_capability()`` is False and
the pipeline records ``sam2_unavailable`` as a limitation rather than
silently failing the case. PRODUCTION_SPEC §10 requires the segmenter
but explicitly allows degraded operation when SAM 2 cannot be loaded;
the decision policy then refuses to upgrade to VERIFIED on perception
evidence alone.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .schemas import Detection, Mask


log = logging.getLogger(__name__)


class Sam2Client:
    def __init__(self, *, model_path: Optional[str] = None,
                 device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._predictor = None
        self._load_err: Optional[str] = None

    # ------------------------------------------------------------------
    # Capability
    # ------------------------------------------------------------------

    def has_capability(self) -> bool:
        if self._predictor is not None:
            return True
        if self._load_err is not None:
            return False
        try:
            import sam2  # noqa: F401
        except ImportError as exc:
            self._load_err = f"sam2 package not installed: {exc}"
            return False
        if not self.model_path or not os.path.isdir(self.model_path):
            self._load_err = f"sam2 weights missing: {self.model_path!r}"
            return False
        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        if not self.has_capability():
            raise RuntimeError(self._load_err or "sam2 unavailable")
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except Exception as exc:
            self._load_err = f"sam2 import failed: {exc}"
            raise
        # from_pretrained is the official entry point on recent releases, but
        # it expects a Hugging Face REPO ID — handing it a local directory
        # path raises KeyError (repo-id lookup miss) / AttributeError /
        # TypeError depending on the sam2 version. On ANY such failure, fall
        # back to building from the local checkpoint we actually have on disk,
        # using the PACKAGED hydra config NAME (Hydra resolves config names
        # from the sam2 package's config dir, not from an arbitrary file path).
        try:
            try:
                self._predictor = SAM2ImagePredictor.from_pretrained(
                    self.model_path, device=self.device)
            except TypeError:
                self._predictor = SAM2ImagePredictor.from_pretrained(
                    self.model_path)
        except Exception as exc_fp:
            from sam2.build_sam import build_sam2  # type: ignore
            ckpt = _find_ckpt(self.model_path)
            if not ckpt:
                self._load_err = (
                    "sam2 from_pretrained failed (%r) and no local .pt "
                    "checkpoint under %s" % (exc_fp, self.model_path))
                raise RuntimeError(self._load_err)
            cfg_name = _hydra_cfg_for_ckpt(ckpt)
            self._predictor = SAM2ImagePredictor(
                build_sam2(cfg_name, ckpt, device=self.device))

    def segment(self,
                image,
                detections: list[Detection]) -> list[Mask]:
        """Return one Mask per detection. The mask payload is left as
        the predictor's raw output; callers that persist masks should
        write them to disk and pass the resulting URI back into the
        Mask object."""
        if not detections:
            return []
        try:
            self._ensure_loaded()
        except Exception as exc:
            log.warning("sam2 load failed; continuing without masks: %s",
                        exc)
            return []
        try:
            import numpy as np
            self._predictor.set_image(np.asarray(image.convert("RGB")))
        except Exception:
            log.exception("sam2 set_image failed")
            return []
        masks: list[Mask] = []
        for idx, det in enumerate(detections):
            try:
                m, scores, _ = self._predictor.predict(
                    box=np.array(det.bbox_xyxy, dtype=np.float32),
                    multimask_output=False,
                )
                score = float(scores[0]) if hasattr(scores, "__len__") \
                    and len(scores) else 0.0
                masks.append(Mask(detection_idx=idx, score=score,
                                  bbox_xyxy=list(det.bbox_xyxy)))
            except Exception:
                log.exception("sam2 predict failed on detection %d", idx)
        return masks

    def unload(self) -> None:
        self._predictor = None


def _find_ckpt(path: str) -> Optional[str]:
    from pathlib import Path
    for p in sorted(Path(path).iterdir()):
        if p.suffix in (".pt", ".pth"):
            return str(p)
    return None


def _hydra_cfg_for_ckpt(ckpt_path: str) -> str:
    """Map a local SAM 2 checkpoint to the PACKAGED hydra config name that
    ``build_sam2`` resolves from the installed ``sam2`` package (e.g.
    ``configs/sam2/sam2_hiera_l.yaml``). Hydra needs the package-relative
    config NAME, not a local .yaml path — passing a filesystem path fails."""
    name = os.path.basename(ckpt_path).lower()
    ver = "sam2.1" if ("sam2.1" in name or "sam2_1" in name) else "sam2"
    if "base_plus" in name or "baseplus" in name or "_b+" in name:
        size = "b+"
    elif "large" in name or "_hiera_l" in name or name.endswith("_l.pt"):
        size = "l"
    elif "small" in name or "_s" in name:
        size = "s"
    elif "tiny" in name or "_t" in name:
        size = "t"
    else:
        size = "l"
    return f"configs/{ver}/{ver}_hiera_{size}.yaml"
