"""
JangSpecBuilder — convert a source JANG (MoE) model into a .jangspec bundle.

Strategy:
    1. Enumerate the source safetensors shards via the index file.
    2. Classify tensor names into hot_core vs streamed experts (tier.py).
    3. Copy hot_core tensors into a single new safetensors file via mmap.
    4. For each (layer, expert_id) tuple, slice the per-expert (qweight, scales,
       biases) triples out of the stacked 3D MoE tensors, pack into an
       ExpertBlob, append to a rolling experts-NNNNN.bin file.
    5. Record every blob's file/offset/nbytes in an ExpertIndexEntry.
    6. Write experts.jsidx and jangspec.json.
    7. Copy tokenizer files.

v1 does NOT populate `draft/` or `router_prior/`. Those are left empty until
later plans. The manifest records `has_draft=False`, `has_router_prior=False`.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from . import format as fmt
from ..format.aligned_safetensors import rewrite_aligned_safetensors
from .blob import ExpertTensors, pack_expert_blob
from .index import ExpertIndexEntry, write_index
from .manifest import Manifest, write_manifest
from .tier import classify_tensors, is_dense_model

# Matches the layer index in tensor names like "layers.7.switch_mlp.gate_proj.weight".
_LAYER_RE = re.compile(r"\.?layers\.(\d+)\.")


def _layer_idx(name: str) -> int:
    m = _LAYER_RE.search(name)
    if not m:
        raise ValueError(f"cannot parse layer index from {name!r}")
    return int(m.group(1))


def _infer_bits(qweight: np.ndarray, scales: np.ndarray, group_size: int) -> int:
    """Recover per-tensor bit width from the uint32 packed shape.

    packed_in = ceil(in_features * bits / 32), in_features = n_groups * group_size
    """
    packed_in = qweight.shape[-1]
    n_groups = scales.shape[-1]
    in_features = n_groups * group_size
    return (packed_in * 32) // in_features


@dataclass
class _ExpertShard:
    """State for the currently-open experts-NNNNN.bin shard file."""

    file_id: int
    path: Path
    size: int = 0

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "wb")

    def close(self) -> None:
        self.handle.close()

    def write(self, blob: bytes) -> Tuple[int, int]:
        """Write a blob, return (offset, nbytes)."""
        offset = self.size
        self.handle.write(blob)
        self.size += len(blob)
        return offset, len(blob)


class JangSpecBuilder:
    def __init__(
        self,
        source_dir: Path,
        out_dir: Path,
        write_streaming: bool = False,
    ):
        """
        Build a .jangspec bundle from a source JANG MoE directory.

        Parameters
        ----------
        source_dir : Path
            The source `JANG_xxx/` directory with safetensors shards.
        out_dir : Path
            Where to write the bundle.
        write_streaming : bool, default False
            If True, additionally emit the per-expert blob format
            (`target/experts.jsidx` + `experts-*.bin`) alongside the
            fat `experts.safetensors`. The per-blob format is intended
            for future SSD-streaming runtimes; it doubles the bundle
            size on disk (~126 GB for MiniMax-M2.7-JANG_2L). The default
            Swift loader never reads it.
        """
        self.source_dir = Path(source_dir)
        self.out_dir = Path(out_dir)
        self.write_streaming = write_streaming

    # ------------------------------------------------------------------ public

    def build(self) -> None:
        self._load_metadata()
        self._validate_moe()
        self._split_tiers()
        self._write_hot_core()
        self._write_experts_and_index()
        self._copy_tokenizer()
        self._write_manifest()

    # ---------------------------------------------------------------- metadata

    def _load_metadata(self) -> None:
        self.config = json.loads((self.source_dir / "config.json").read_text())
        try:
            self.jang_config = json.loads((self.source_dir / "jang_config.json").read_text())
        except FileNotFoundError:
            self.jang_config = {}

        index_path = self.source_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"missing {index_path}")
        self.st_index = json.loads(index_path.read_text())
        self.shard_map: Dict[str, str] = self.st_index["weight_map"]  # tensor_name -> shard file
        self.all_tensor_names: List[str] = sorted(self.shard_map.keys())

        q = self.config.get("quantization", {})
        self.group_size = int(q.get("group_size", 64))
        self.target_top_k = int(
            self.config.get("num_experts_per_tok")
            or self.config.get("moe_top_k")
            or self.config.get("num_local_experts_per_tok")
            or 2
        )

    def _validate_moe(self) -> None:
        if is_dense_model(self.all_tensor_names):
            raise ValueError(
                f"{self.source_dir} appears to be a dense JANG model. "
                "jang-spec only supports MoE targets — dense models already fit in RAM."
            )

    def _split_tiers(self) -> None:
        self.split = classify_tensors(self.all_tensor_names)

    # --------------------------------------------------------------- hot core

    def _open_shard(self, shard_filename: str):
        path = self.source_dir / shard_filename
        return safe_open(path, framework="numpy", device="cpu")

    def _write_hot_core(self) -> None:
        # Gather all hot-core tensors by source shard to minimise safe_open calls.
        by_shard: Dict[str, List[str]] = {}
        for name in self.split.hot_core:
            by_shard.setdefault(self.shard_map[name], []).append(name)

        tensors: Dict[str, np.ndarray] = {}
        hot_bytes = 0
        for shard_filename, names in by_shard.items():
            with self._open_shard(shard_filename) as f:
                for name in names:
                    arr = f.get_tensor(name)
                    tensors[name] = arr
                    hot_bytes += arr.nbytes

        out_path = self.out_dir / fmt.HOT_CORE_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out_path))
        rewrite_aligned_safetensors(out_path)
        self.hot_core_bytes = hot_bytes

    # -------------------------------------------------------------- experts

    def _write_experts_and_index(self) -> None:
        # Load the three 3D stacked expert tensors per layer, on demand.
        # Each base name has suffixes .weight (uint32), .scales (f16), .biases (f16).
        expert_bases = self.split.expert_base_names

        # Group the base names by layer.
        layers: Dict[int, Dict[str, str]] = {}  # layer_idx -> {"gate_proj": base, "up_proj": base, "down_proj": base}
        for base in expert_bases:
            lid = _layer_idx(base)
            for kind in ("gate_proj", "up_proj", "down_proj"):
                if base.endswith(f".switch_mlp.{kind}"):
                    layers.setdefault(lid, {})[kind] = base
                    break

        sorted_layers = sorted(layers.keys())
        if not sorted_layers:
            raise ValueError("no expert layers detected after tier split")

        # Peek the first layer to discover E (num experts) and validate shapes.
        first = layers[sorted_layers[0]]
        shard_of = lambda name: self.shard_map[name + ".weight"]
        with self._open_shard(shard_of(first["gate_proj"])) as f:
            gate_q = f.get_tensor(first["gate_proj"] + ".weight")
        if gate_q.ndim != 3:
            raise ValueError(
                f"expected stacked 3D expert tensor, got shape {gate_q.shape} for {first['gate_proj']}"
            )
        self.n_experts_per_layer = int(gate_q.shape[0])
        self.n_layers = len(sorted_layers)
        assert sorted_layers == list(range(self.n_layers)), (
            f"expert layer indices are not contiguous: {sorted_layers}"
        )

        # Per-blob streaming output — only written when write_streaming=True.
        shards: List[_ExpertShard] = []
        current: Optional[_ExpertShard] = None
        if self.write_streaming:
            current = _ExpertShard(
                file_id=0, path=self.out_dir / fmt.EXPERT_FILE_PATTERN.format(idx=0))
            current.open()
            shards.append(current)

        entries: List[ExpertIndexEntry] = []

        # Fat-layout output: all expert tensors in their original 3D stacked
        # form under their full original names. This lets the Swift runtime
        # `loadArraysAndMetadata`-mmap them directly with zero restack, the
        # same way the source JANG directory path works. This is the default
        # load path for the Swift bundle loader.
        fat_tensors: Dict[str, np.ndarray] = {}

        for lid in sorted_layers:
            base = layers[lid]
            arrays: Dict[str, Dict[str, np.ndarray]] = {}
            for kind in ("gate_proj", "up_proj", "down_proj"):
                name = base[kind]
                with self._open_shard(shard_of(name)) as f:
                    arrays[kind] = {
                        "qweight": f.get_tensor(name + ".weight"),
                        "scales": f.get_tensor(name + ".scales"),
                        "biases": f.get_tensor(name + ".biases"),
                    }

            bits = _infer_bits(arrays["gate_proj"]["qweight"], arrays["gate_proj"]["scales"], self.group_size)

            # Emit 3D stacked tensors into the fat layout. Same byte
            # layout the source directory has — these arrays are the
            # product of JANG's build pipeline and already have leading
            # expert dim at axis 0.
            for kind in ("gate_proj", "up_proj", "down_proj"):
                full_base = base[kind]
                fat_tensors[f"{full_base}.weight"] = np.ascontiguousarray(
                    arrays[kind]["qweight"])
                fat_tensors[f"{full_base}.scales"] = np.ascontiguousarray(
                    arrays[kind]["scales"])
                fat_tensors[f"{full_base}.biases"] = np.ascontiguousarray(
                    arrays[kind]["biases"])

            # Per-blob streaming format — optional, only when requested.
            if self.write_streaming:
                assert current is not None
                for expert_id in range(self.n_experts_per_layer):
                    expert = ExpertTensors(
                        bits=bits,
                        gate=(
                            np.ascontiguousarray(arrays["gate_proj"]["qweight"][expert_id]),
                            np.ascontiguousarray(arrays["gate_proj"]["scales"][expert_id]),
                            np.ascontiguousarray(arrays["gate_proj"]["biases"][expert_id]),
                        ),
                        up=(
                            np.ascontiguousarray(arrays["up_proj"]["qweight"][expert_id]),
                            np.ascontiguousarray(arrays["up_proj"]["scales"][expert_id]),
                            np.ascontiguousarray(arrays["up_proj"]["biases"][expert_id]),
                        ),
                        down=(
                            np.ascontiguousarray(arrays["down_proj"]["qweight"][expert_id]),
                            np.ascontiguousarray(arrays["down_proj"]["scales"][expert_id]),
                            np.ascontiguousarray(arrays["down_proj"]["biases"][expert_id]),
                        ),
                    )
                    blob = pack_expert_blob(layer_idx=lid, expert_id=expert_id, tensors=expert)

                    if current.size + len(blob) > fmt.EXPERT_FILE_BYTES_MAX and current.size > 0:
                        current.close()
                        current = _ExpertShard(
                            file_id=current.file_id + 1,
                            path=self.out_dir / fmt.EXPERT_FILE_PATTERN.format(idx=current.file_id + 1),
                        )
                        current.open()
                        shards.append(current)

                    offset, nbytes = current.write(blob)
                    entries.append(
                        ExpertIndexEntry(
                            layer_idx=lid,
                            expert_id=expert_id,
                            file_id=current.file_id,
                            offset=offset,
                            nbytes=nbytes,
                        )
                    )

            # Release numpy views early to keep RAM use bounded.
            del arrays

        if self.write_streaming:
            assert current is not None
            current.close()
            write_index(
                self.out_dir / fmt.INDEX_FILENAME,
                entries=entries,
                n_layers=self.n_layers,
                n_experts_per_layer=self.n_experts_per_layer,
            )
            self.expert_bytes = sum(e.nbytes for e in entries)
        else:
            # Fat-only: expert_bytes is the aggregate size of the 3D stacks
            # we wrote into experts.safetensors.
            self.expert_bytes = sum(a.nbytes for a in fat_tensors.values())

        # Save the fat layout. The Swift loader prefers this path because
        # it maps directly into unified memory with no restack cost.
        fat_path = self.out_dir / "experts.safetensors"
        save_file(fat_tensors, str(fat_path))
        rewrite_aligned_safetensors(fat_path)

    # ------------------------------------------------------------- tokenizer

    def _copy_tokenizer(self) -> None:
        # Copy every non-weight file from the source model directory into
        # the bundle root. This catches anything the model factory might
        # need (processor_config.json for VLMs, generation_config.json,
        # chat_template.jinja, preprocessor configs, modeling_*.py for
        # trust_remote_code models, etc.) without having to maintain a
        # hardcoded list per model architecture.
        skip_suffixes = (".safetensors",)
        skip_names = {"model.safetensors.index.json"}
        for child in self.source_dir.iterdir():
            if not child.is_file():
                continue
            if child.name in skip_names:
                continue
            if child.name.endswith(skip_suffixes):
                continue
            shutil.copy2(child, self.out_dir / child.name)
        # Also keep config.json and jang_config.json under target/ for the
        # streaming Swift runtime that reads from the bundle's target/ subdir.
        (self.out_dir / "target").mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "jang_config.json"):
            src = self.source_dir / name
            if src.exists():
                shutil.copy2(src, self.out_dir / "target" / name)

    # -------------------------------------------------------------- manifest

    def _write_manifest(self) -> None:
        tokenizer_path = self.out_dir / "tokenizer.json"
        if tokenizer_path.exists():
            tok_hash = "sha256:" + hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        else:
            tok_hash = ""

        manifest = Manifest(
            bundle_version=fmt.BUNDLE_VERSION,
            source_jang=self.source_dir.name,
            source_jang_dir=str(self.source_dir),
            target_arch=self.config.get("model_type", "unknown"),
            n_layers=self.n_layers,
            n_experts_per_layer=self.n_experts_per_layer,
            target_top_k=self.target_top_k,
            tokenizer_hash=tok_hash,
            hot_core_tensors=self.split.hot_core,
            expert_tensor_names=self.split.expert_base_names,
            n_experts_total=self.n_layers * self.n_experts_per_layer,
            hot_core_bytes=self.hot_core_bytes,
            expert_bytes=self.expert_bytes,
            has_draft=False,
            has_router_prior=False,
        )
        write_manifest(self.out_dir / fmt.MANIFEST_FILENAME, manifest)
