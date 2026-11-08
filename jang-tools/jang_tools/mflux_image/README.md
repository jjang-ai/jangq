# mflux image-model quantization

Scripts that build the [mflux](https://github.com/filipstrand/mflux) bundles for Qwen-Image-2.1,
Qwen-Image-2.1-Turbo and Ideogram 4 (MLX affine quantization, group size 64).

```sh
pip install mflux    # a version with Qwen-Image-2.1 support (the `qwen21` model family)
python -m jang_tools.mflux_image.quantize_qwen_image21 --bits 8 --source Qwen-Image-2.1-Turbo --out Qwen-Image-2.1-Turbo-mflux-8bit
python -m jang_tools.mflux_image.generate_qwen_image21 --model Qwen-Image-2.1-Turbo-mflux-8bit --out cafe.png --seed 7 \
    --prompt 'A small cafe on a rainy street with a wooden sign that clearly reads "OSAURUS CAFE"'
python -m jang_tools.mflux_image.generate_qwen_image21 --model Qwen-Image-2.1-Turbo-mflux-8bit --out edit.png --seed 7 \
    --ref cafe.png --prompt 'Change the sign to read "JANG BAKERY". Keep everything else.'
python -m jang_tools.mflux_image.quantize_ideogram4 --bits 8 --source ideogram-4 --out ideogram-4-mflux-8bit
```

## Qwen-Image-2.1 and -Turbo

- Bundles are saved through `QwenImage21Edit`, which keeps the Qwen3-VL vision tower: one bundle does text-to-image
  and reference editing. `modulation`, `time_text_embed` and `norm_out` stay bf16; the VAE stays fp32.
- Turbo has the same architecture as 2.1 with distilled weights. It samples on the fixed 8-node sigma grid stored in
  `model_index.json` (`sample_sigmas`), unshifted, with a terminal 0 appended and guidance 1.0. The grid sets the step
  count. A sampler that applies 2.1's shifted 40-step schedule to Turbo weights produces degraded images.
- Quantization does not need any mflux change; `save_model` copies `model_index.json` and `scheduler/`, and the
  script asserts the grid survived. Generation with Turbo needs `patches/mflux-03700b9-qwen21-turbo-scheduler.patch`
  (against mflux `03700b9`): it adds the `qwen21_turbo` scheduler, has `QwenImage21Edit.generate_image` pick it when
  the checkpoint pins `sample_sigmas`, and accepts `--scheduler qwen21_turbo` on the edit CLI.

## Ideogram 4

The tokenizer directory is copied into the bundle. Prompt Ideogram 4 with a JSON caption when testing; plain-text
prompts can return the model's learned "blocked by safety filter" card.
