import json


def _write_json(path, value):
    path.write_text(json.dumps(value))


def test_qwen4_exp_stamp_selects_native_qwen_tool_parser(tmp_path):
    from jang_tools.qwen4_exp.stamp import EOS_IDS, stamp

    _write_json(
        tmp_path / "config.json",
        {"text_config": {"max_position_embeddings": 262144}},
    )
    _write_json(tmp_path / "generation_config.json", {})
    _write_json(
        tmp_path / "model.safetensors.index.json",
        {"weight_map": {"mtp.fc.weight": "model-00001.safetensors"}},
    )
    (tmp_path / "chat_template.jinja").write_text(
        "reasoning_effort preserve_thinking <tool_call> </think>"
    )

    stamp(tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    generation = json.loads((tmp_path / "generation_config.json").read_text())
    assert config["capabilities"]["tool_parser"] == "qwen"
    assert config["jang_config"]["capabilities"]["tool_parser"] == "qwen"
    assert config["jang_config"]["mtp"]["mtp_mode"] == "preserved_enabled"
    assert generation["eos_token_id"] == EOS_IDS


def test_glm5_next_stamp_records_native_tool_reasoning_and_mtp(tmp_path):
    from jang_tools.glm5_next.stamp import EOS_IDS, stamp

    _write_json(
        tmp_path / "config.json",
        {
            "text_config": {
                "max_position_embeddings": 1048576,
                "index_topk": 2048,
            }
        },
    )
    _write_json(tmp_path / "generation_config.json", {})
    _write_json(
        tmp_path / "model.safetensors.index.json",
        {
            "weight_map": {
                "model.layers.45.mtp.weight": "model-00001.safetensors",
                "visual.encoder.weight": "model-00002.safetensors",
            }
        },
    )
    _write_json(
        tmp_path / "processor_config.json",
        {"image_processor": {}, "video_processor": {}},
    )
    (tmp_path / "chat_template.jinja").write_text(
        "reasoning_effort clear_thinking <tool_call> arg_key </think>"
    )

    stamp(tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    generation = json.loads((tmp_path / "generation_config.json").read_text())
    capabilities = config["jang_config"]["capabilities"]
    assert capabilities["tool_parser"] == "glm_xml_args"
    assert capabilities["reasoning_parser"] == "glm_think_block"
    assert capabilities["has_vision"] is True
    assert capabilities["has_video"] is True
    assert config["jang_config"]["mtp"]["mtp_mode"] == "preserved_enabled"
    assert generation["eos_token_id"] == EOS_IDS
