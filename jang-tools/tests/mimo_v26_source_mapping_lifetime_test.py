"""Real safetensor storage remains readable across streaming mapping release."""
import json
import numpy as np
import torch
import mlx.core as mx
from safetensors.torch import save_file
from jang_tools.mimo_v2.weight_loader import MiMoShardIndex


def test_returned_tensors_and_materialized_outputs_survive_release(tmp_path):
    mx.set_default_device(mx.cpu)
    expected=torch.arange(128,dtype=torch.float32).reshape(8,16).to(torch.bfloat16)
    save_file({'dense.weight':expected},str(tmp_path/'model-00001.safetensors'))
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'dense.weight':'model-00001.safetensors'}}))
    index=MiMoShardIndex(tmp_path)
    borrowed=index.read_passthrough('dense.weight')
    materialized=mx.array(borrowed.view(torch.int16).numpy()).view(mx.bfloat16)
    mx.eval(materialized)
    assert index._handles
    index.release_cached_handles()
    assert not index._handles
    assert torch.equal(borrowed,expected)
    np.testing.assert_array_equal(np.array(materialized.view(mx.uint16)),expected.view(torch.uint16).numpy())
    reopened=index.read_passthrough('dense.weight')
    assert torch.equal(reopened,expected)
    index.release_cached_handles()
    index.release_cached_handles()
    assert torch.equal(reopened,borrowed)
