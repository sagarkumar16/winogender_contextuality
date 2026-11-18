"""gpt-oss loaded with quantized=False must really be bf16."""

import pytest

pytest.importorskip("torch")


def test_check_dequantized_accepts_bf16_and_rejects_anything_else():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "GptOssConfig"):
        pytest.skip("transformers has no gpt-oss support")
    import torch
    from winogender_contextuality.modeling.run_local import check_dequantized, is_gpt_oss

    cfg = transformers.GptOssConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=64,
                                    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                    num_local_experts=4, num_experts_per_tok=2, vocab_size=128,
                                    layer_types=["full_attention", "sliding_attention"])
    model = transformers.GptOssForCausalLM(cfg)

    with pytest.raises(RuntimeError):
        check_dequantized(model)  # fp32
    model = model.to(torch.bfloat16)
    check_dequantized(model)

    class Mxfp4GptOssExperts(torch.nn.Module):
        pass
    model.model.layers[0].mlp.experts = Mxfp4GptOssExperts()
    with pytest.raises(RuntimeError):
        check_dequantized(model)

    assert is_gpt_oss("openai/gpt-oss-20b") and not is_gpt_oss("microsoft/phi-4")
