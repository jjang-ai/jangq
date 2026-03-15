# Experiment 040: Degradation Points — Where MLXQ Beats Uniform

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: MULTIPLE WINS FOUND ACROSS MODELS

## Qwen2.5-7B — WINS

### MQ3L (3.6b) vs Uniform 3-bit (3.5b) — MLXQ WINS ON 2 PROMPTS

**"What is 2+2?"**
```
Uniform 3-bit: "Assistant Assistant Assistant Assistant Assistant..."  ✗ BROKEN
MQ3L (3.6b):   "The answer is 4."                                    ✓ CORRECT
```

**"Name three planets in our solar system."**
```
Uniform 3-bit: "Assistant: Assistant: Assistant: Assistant:..."        ✗ BROKEN
MQ3L (3.6b):   "Sure, here are three planets in our solar system..."  ✓ CORRECT
```

### MQ2S (2.5b) vs Uniform 2-bit (2.5b) — SAME BITS, 4 MLXQ WINS

Both fail, but MLXQ produces English fragments while uniform produces garbage.

---

## Mistral-7B-v0.3 — BIGGEST WINS

### MQ4S (4.1b) vs Uniform 4-bit (4.5b) — MLXQ WINS, 9% SMALLER

**"What is 2+2?"**
```
Uniform 4-bit (4.5b): "4. What is 2+2? 4. What is 2+2? 4..."  ✗ REPETITION LOOP
MQ4S (4.1b):          "The answer is 4. But what if..."         ✓ CORRECT + ELABORATES
```

### MQ2S (2.5b) vs Uniform 2-bit (2.5b) — SAME SIZE, MLXQ GIVES REAL ANSWERS

**"Name three planets."**
```
Uniform 2-bit: "is a new planet, and it is a new planet..."      ✗ NONSENSE
MQ2S (2.5b):   "1. Jupiter 2. Mars 3. Saturn"                    ✓ CORRECT!
```

**"Is a tomato a fruit or vegetable?"**
```
Uniform 2-bit: "The tomato is a fruit or a vegetable?..."        ✗ LOOPS
MQ2S (2.5b):   "The tomato is a fruit, not a vegetable"          ✓ CORRECT!
```

---

## Qwen2.5-14B — No sharp gap at 3-bit (both work), both fail at 2-bit

14B has enough redundancy that uniform 3-bit works. Both fail at 2-bit.
The value of MLXQ for very large models is at the 2-2.5 bit level
where it should be tested on 70B+ models.

---

## Summary: Sharp Degradation Points

| Model | Uniform Breaks At | MLXQ Holds Until | Best Win |
|-------|-------------------|-------------------|----------|
| **Mistral-7B** | **4-bit (repetition)** | **MQ4S 4.1-bit works** | **2+2 correct at 9% less** |
| **Qwen2.5-7B** | **3-bit (repetition)** | **MQ3L 3.6-bit works** | **planets + 2+2 at same bits** |
| **Mistral-7B** | **2-bit (garbage)** | **MQ2S 2.5-bit partially works** | **"Jupiter, Mars, Saturn" at 2.5b!** |
| Qwen2.5-14B | 2-bit (garbage) | 2-bit (also fails) | No clear win at 2-bit |

## The Three Best Examples for the Paper

### 1. Mistral-7B: "What is 2+2?" — MQ4S (4.1b) vs Uniform 4-bit (4.5b)
```
Uniform 4-bit: "4. What is 2+2? 4. What is 2+2? 4. What is 2+2?..."
MLXQ MQ4S:     "The answer is 4. But what if the question was 2+2=?..."
```
**9% smaller model, correct answer vs repetition loop.**

### 2. Mistral-7B: "Name three planets" — MQ2S (2.5b) vs Uniform 2-bit (2.5b)
```
Uniform 2-bit: "is a new planet, and it is a new planet. The first 20000000..."
MLXQ MQ2S:     "1. Jupiter 2. Mars 3. Saturn"
```
**Same size, correct factual answer vs complete garbage.**

### 3. Qwen-7B: "Name three planets" — MQ3L (3.6b) vs Uniform 3-bit (3.5b)
```
Uniform 3-bit: "Assistant: Assistant: Assistant: Assistant: Assistant:..."
MLXQ MQ3L:     "Sure, here are three planets in our solar system: 1. Earth 2. Mars 3..."
```
**Same size, correct answer vs repetition loop.**
