import json

from jang_tools.architectures import ArchType, detect_architecture
from jang_tools.capabilities import build_capabilities
from jang_tools.convert import (
    _is_vision_tensor_name,
    _muse_glimmer_chat_metadata,
    _muse_glimmer_runtime_metadata,
    _read_hf_local_revision,
    _remap_tokenizer_class_for_swift,
)


def _source_config() -> dict:
    return {
        "model_type": "muse_glimmer",
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "vision_config": {"model_type": "muse_glimmer_vision"},
        "text_config": {
            "model_type": "muse_glimmer_text",
            "num_hidden_layers": 52,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "sliding_window": 2048,
            "layer_types": [
                "full_attention" if (i + 1) % 4 == 0 else "sliding_attention"
                for i in range(52)
            ],
        },
    }


def test_architecture_is_dense_vision_language_even_with_layer_types(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_source_config()))
    arch = detect_architecture(tmp_path)
    assert arch.arch_type is ArchType.VISION_LANGUAGE
    assert arch.has_vision_encoder is True
    assert arch.has_moe_layers is False
    assert arch.has_ssm_layers is False


def test_all_muse_vision_namespaces_are_passthrough():
    for name in (
        "model.vision_tower.patch_embedder.patch_embedding.weight",
        "model.vision_tower.layers.0.attn.q_proj.weight",
        "model.vision_adapter.fc1.weight",
        "model.vision_projection.weight",
    ):
        assert _is_vision_tensor_name(name)
    assert not _is_vision_tensor_name("model.language_model.layers.0.self_attn.q_proj.weight")


def test_capabilities_preserve_native_reasoning_and_atem_tools():
    jang = {"has_vision": True, "has_video": True}
    caps = build_capabilities(jang, _source_config())
    assert caps == {
        "reasoning_parser": "muse_glimmer",
        "tool_parser": "atem",
        "think_in_template": False,
        "supports_tools": True,
        "supports_thinking": True,
        "family": "muse_glimmer",
        "modality": "multimodal",
        "modalities": {"text": True, "vision": True, "audio": False, "video": True},
        "has_vision": True,
        "has_audio": False,
        "has_video": True,
        "cache_type": "kv",
    }


def test_unknown_tokenizers_backend_is_not_mislabeled_as_qwen():
    tokenizer = {"tokenizer_class": "TokenizersBackend"}
    assert _remap_tokenizer_class_for_swift(tokenizer, "muse_glimmer") is None
    assert tokenizer["tokenizer_class"] == "TokenizersBackend"


def test_known_tokenizers_backend_alias_is_still_applied():
    tokenizer = {"tokenizer_class": "TokenizersBackend"}
    assert _remap_tokenizer_class_for_swift(tokenizer, "qwen3") == "Qwen2Tokenizer"
    assert tokenizer["tokenizer_class"] == "Qwen2Tokenizer"


def test_runtime_metadata_pins_mixed_cache_schedule():
    runtime = _muse_glimmer_runtime_metadata(_source_config())
    assert runtime["sliding_window"] == 2048
    assert runtime["full_attention_layers"] == list(range(3, 52, 4))
    assert len(runtime["sliding_attention_layers"]) == 39
    assert runtime["cache_topology"] == {
        "family": "hybrid_full_swa_kv",
        "one_cache_per_text_layer": True,
        "full_attention": "unbounded_kv",
        "sliding_attention": "rotating_kv",
        "prefix_cache_required": True,
        "partial_block_restore_required": True,
        "suffix_prefill_required": True,
    }


def test_hf_local_revision_is_read_from_download_metadata(tmp_path):
    metadata = tmp_path / ".cache/huggingface/download/config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("f84ecc3a0ea984a4c04542a84269e3d065350a6e\nblob-id\n")
    assert _read_hf_local_revision(tmp_path) == "f84ecc3a0ea984a4c04542a84269e3d065350a6e"


def test_chat_metadata_preserves_native_defaults_without_inventing_sampling(tmp_path):
    generation = {
        "bos_token_id": 200000,
        "eos_token_id": [200001, 200008],
        "pad_token_id": 200018,
        "max_length": 131072,
        "do_sample": False,
    }
    (tmp_path / "generation_config.json").write_text(json.dumps(generation))
    chat = _muse_glimmer_chat_metadata(tmp_path)
    assert chat["reasoning"]["default_mode"] == "high"
    assert chat["reasoning"]["control"] == "reasoning_strength"
    assert chat["tool_calling"] == {"supported": True, "parser": "atem", "format": "atem"}
    assert chat["sampling_defaults"] == {"do_sample": False}
    assert chat["generation_defaults"] == generation
    assert "temperature" not in chat["sampling_defaults"]
