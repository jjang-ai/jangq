# Experiment 043: Extended Tests — 14B, Mistral Extended, StableLM

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)

## Qwen2.5-14B

### 3-bit: Both work perfectly
14B has enough redundancy — uniform 3-bit and MQ3M both give correct answers
on all 6 prompts. No degradation at this level.

### 2-bit: Both fail
Uniform 2-bit: complete garbage (`.l. .l. .l.Word.Word.`)
MQ2S: assistant loops (broken but slightly more coherent)
MQ2M: also loops

**Conclusion**: 14B needs GPTQ or NF quantization for 2-bit to work.
The model is too large for this machine to test many configs quickly.

## Mistral-7B Extended (8 new harder prompts at 3-bit)

### New MLXQ wins at 3-bit boundary:

**"Name a famous painting by Leonardo da Vinci."**
```
Uniform 3 (3.5b): "Name a famous painting by Michelangelo. Name a famous painting by Rap..."
MQ3M (3.4b):      "The Last Supper, The Mona Lisa, The Vitruvian Man..."
```
Uniform switches to WRONG ARTIST. MLXQ gives correct list. FEWER bits.

**"What does CPU stand for?"**
```
Uniform 3 (3.5b): "TDM ## What does CPU stand for? CPU is an acronym for Central Proces..."
MQ3M (3.4b):      "CPU is the abbreviation for Central Processing Unit. It is the main..."
```
Uniform has garbage prefix "TDM ##". MLXQ gives clean answer. FEWER bits.

### Both work (no clear winner):
- DNA, rainbow colors, boiling point, continents — both give correct answers
- Shows that the advantage is at the MARGIN — on harder prompts where
  uniform starts to struggle, MLXQ maintains coherence

## StableLM-2-1.6B

StableLM is robust — both uniform and MLXQ give good answers at 3 and 4 bit.
MQ4S gives slightly more detailed answers ("Step 1: Identify the given numbers")
vs uniform ("The answer is 4.").

No dramatic failures at either level — this model handles quantization well
due to its MHA architecture (32 heads, lots of redundancy).

## Updated Win Count

| Model | Previous Wins | New Wins | Total |
|-------|--------------|----------|-------|
| TinyLlama-1.1B | 11 | — | 11 |
| SmolLM2-1.7B | 11 | — | 11 |
| Phi-2 | 9 | — | 9 |
| Qwen2.5-3B | 6 | — | 6 |
| Mistral-7B | 11 | +2 | 13 |
| Qwen2.5-7B | 9 | — | 9 |
| StableLM-2-1.6B | — | 0 (robust) | 0 |
| Qwen2.5-14B | — | 0 (both ok or both fail) | 0 |
| **Total** | **57** | **+2** | **59** |
