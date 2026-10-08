# VLM finetuning framework comparison: LLaMA-Factory vs Qwen-VL-Series-Finetune

This page compares two open-source frameworks for finetuning vision-language models (VLMs):

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)

It looks at how hard three planned changes to the models would be in each:

1. adding a **Q-Former** between the vision encoder and the language model
2. replacing text output with an **MLP action head**
3. changing the **training loss**

It was written for the TrafficRuleVQA thesis project (TU Munich, Professorship of Autonomous Vehicle Systems). That project LoRA-finetunes Gemma-4-E2B, InternVL3.5-2B and InternVL3.5-4B with LLaMA-Factory, and smoke-tested Qwen3-VL-2B with Qwen-VL-Series-Finetune. Notes that apply only to that project's setup are marked as such.

The comparison comes from reading the code at the versions the project pins:

| Framework | Commit | Notes |
|---|---|---|
| LLaMA-Factory | [`d6bb97dd`](https://github.com/hiyouga/LLaMA-Factory/tree/d6bb97ddff5d752d8b05aa099a168127c7253562) (2026-08-31) | Version 0.9.6.dev0 |
| Qwen-VL-Series-Finetune | [`c8f7377`](https://github.com/2U1/Qwen-VL-Series-Finetune/tree/c8f7377b67dc0b7c34c77cc4e7e65698401b3dce) (2026-07-21) | |

None of the changes has been implemented or run. The effort ratings are rough estimates. Line numbers refer to these commits. Both projects have newer commits that were not reviewed.

Path prefixes used below:
- `LF/` = `src/llamafactory/` in LLaMA-Factory
- `QV/` = `src/` in Qwen-VL-Series-Finetune
- `TrafficRuleVQA/` = the thesis project repo (private)

## Recommendation

For TrafficRuleVQA, use **LLaMA-Factory** as the main framework. Keep the custom code in a module of the project repo instead of editing LLaMA-Factory itself (see [Where custom code should live](#where-custom-code-should-live)).

- **Fair comparison.** All three finetuned baselines (Gemma-4-E2B, InternVL3.5-2B, InternVL3.5-4B) were trained with LLaMA-Factory. A modified model trained the same way differs from its baseline only in the change being tested.
- **Q-Former.** Qwen-VL-Series-Finetune trains Qwen models only, and Qwen3-VL is the hardest of our architectures to give a Q-Former (see section 1). InternVL is the easiest, and only LLaMA-Factory trains it.
- **Loss.** LLaMA-Factory has a hook for custom losses (`compute_loss_func`) with three working examples. No fused kernel hides the logits, because Liger doesn't support our models there.

**Exception: a discrete action classifier on Qwen3-VL.** Qwen-VL-Series-Finetune already has a classification pipeline with an MLP head and focal and class-balanced losses. It is the fastest way to prototype one, once the loss bug in [Known issues](#known-issues-found-while-reading-the-code) is fixed. LLaMA-Factory has no classification or regression stage.

## Summary

| Change | LLaMA-Factory | Qwen-VL-Series-Finetune | Better choice |
|---|---|---|---|
| Q-Former | Work per architecture. InternVL: low to moderate. Gemma-4: moderate. Qwen3-VL: high | Qwen3-VL only: high | LLaMA-Factory, starting with InternVL |
| MLP action head, discrete classes | Moderate to high: no stage exists, so wrap the forward or add a stage | Low: a classification pipeline exists (it needs a bug fix and an inference loader) | Qwen-VL for a quick Qwen-only prototype. LLaMA-Factory to compare against the baselines |
| MLP action head, continuous values (waypoints) | Moderate to high: as above, plus float targets | Low to moderate: about 50–100 lines on top of the classification pipeline | As above |
| Token-level loss change | Low: one function next to DFT and EAFT, about 30 lines | Low to moderate: override `compute_loss` and work around Liger | LLaMA-Factory |
| Reward-based RL (GRPO) | Not available. Stages are pt, sft, rm, ppo, dpo, kto | Available: add a `*_reward` function | Qwen-VL |

## How the two frameworks are built

| | LLaMA-Factory | Qwen-VL-Series-Finetune |
|---|---|---|
| Models | Many families, including Gemma-4, InternVL3.5 and Qwen3-VL | Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5 (`QV/model/load_model.py:24-51`) |
| Model loading | `LF/model/loader.py`: `from_pretrained` (171), then `patch_model` (178), then the LoRA adapter (181). Training and `ChatModel` inference use the same path. | `QV/model/load_model.py` loads the stock HF model and patches its backbone forward (`QV/train/monkey_patch_forward.py`). Inference uses a separate loader (`QV/utils.py:25`). |
| Per-model knowledge | Registries: which modules are vision tower, projector and LLM (`COMPOSITE_MODELS` in `LF/model/model_utils/visual.py`); chat templates; multimodal plugins that decide the image token count (`LF/data/mm_plugin.py`) | Written for Qwen. Patches are applied on import. |
| Trainable parameters | `freeze_*` flags, `lora_target`, and `additional_target` (PEFT `modules_to_save`, trained in full) | `freeze_*` flags. After PEFT wraps the model, parameters whose names contain `visual` or `merger` are re-enabled (`QV/train/train_sft.py:184-192`) |
| Loss | The model's own loss, unless `compute_loss_func` is set (`LF/train/sft/trainer.py:108-125`) | The model's own loss. Liger fused cross-entropy is on by default (`QV/params.py:152`) |
| Size and how to change it | Large, maintained upstream, pinned submodule. Extend from outside: register components or patch functions. | About 7,200 lines, vendored. Edit in place. |

## 1. Adding a Q-Former

A Q-Former takes the vision encoder's patch features and returns a fixed number K of query tokens per image. These replace the image tokens in the language model's input. In either framework, three things have to agree:

1. **Token count in the prompt.** The data pipeline inserts one placeholder token per visual token. With a Q-Former it must insert K per image.
2. **Model forward.** The projector is replaced, and the features are grouped per image so that each image gets its own K queries.
3. **Position ids.** Models with ordinary 1-D positions need no change. Qwen3-VL computes 3-D M-RoPE positions from each image's patch grid, which no longer matches K tokens.

### What each architecture needs

| Model | Current projector | Image tokens now | What a Q-Former needs |
|---|---|---|---|
| InternVL3.5 | `model.multi_modal_projector`, an MLP on pixel-shuffled features | 256 per 448×448 tile, fixed by `processor.image_seq_length` | Replace the projector and set `image_seq_length = K`. Positions are 1-D, so nothing else changes. |
| Gemma-4-E2B | `model.embed_vision` | Up to 280, depends on aspect ratio | Change the count in the plugin. Patch `get_image_features`: the vision tower strips padding and hands `embed_vision` one flat sequence for all images, so the per-image grouping has to be rebuilt from `image_position_ids`. |
| Qwen3-VL | `visual.merger`, plus DeepStack mergers that add visual features into early LLM layers | One per 32×32 px, depends on image size | Change the count, the per-image split, the DeepStack features (reduce them to K or turn them off) and the M-RoPE grid |

### In LLaMA-Factory

- **Adding the module.** Add it in `patch_model` (`LF/model/patcher.py:440`), which runs after `from_pretrained` and before PEFT wraps the model. `ChatModel` loads through the same function, so evaluation gets the Q-Former too, provided the patch is also installed in the eval process.
- **Training it next to LoRA.**
  - List the module in `additional_target`. It becomes a PEFT `modules_to_save` module: trained in full, saved in `adapter_model.safetensors`, and restored on resume and at inference.
  - Keep `freeze_multi_modal_projector: true`. Otherwise `lora_target: all` can also put LoRA on the Q-Former's own linear layers, which conflicts with `modules_to_save`.
- **Token count.**
  - InternVL: set `processor.image_seq_length` (patchable in `patch_processor`, `LF/model/patcher.py:350`).
  - Gemma-4 and Qwen3-VL: subclass the plugin and register it with `register_mm_plugin` (`LF/data/mm_plugin.py:3274`).
- **Qwen3-VL positions.** The collator computes M-RoPE ids from `image_grid_thw` (`LF/data/collator.py:162`). It would have to pass a made-up grid with K cells, while the vision tower still gets the real grid.
- **Existing examples to copy.**
  - The only projector swap in the codebase is for Yi-VL (`configure_visual_model` in `LF/model/model_utils/visual.py`).
  - The youtu_vl patch (`LF/model/patcher.py:306`) shows how to wrap a model's forward.
- **Alignment first.** A newly initialised Q-Former probably needs an alignment stage before SFT. Full finetuning with `freeze_vision_tower: true` and `freeze_language_model: true` trains only the projector.

### In Qwen-VL-Series-Finetune

This framework supports only Qwen, so all four Qwen3-VL changes apply. Rough estimate: several hundred lines across about 8 files, plus the eval backend.

- **Forward.** `qwen3_vl_mixed_modality_forward` (`QV/train/monkey_patch_forward.py:338`) needs these changes:
  - Replace the `get_image_features` call (378-385): split the features by `image_grid_thw`, then run the Q-Former.
  - Handle DeepStack (396-424).
  - Handle the text-only path (369-376). It runs the vision tower on a dummy image so that vision parameters always get gradients, and it would have to go through the Q-Former too.
- **Token count.** `QV/dataset/sft_dataset.py:198` takes the counts from the HF processor. Each run of image placeholder tokens must be shortened to K, with `mm_token_type_ids` adjusted to match. The project's Qwen eval backend (`TrafficRuleVQA/05_Eval/01_FullEval/models/qwen3vl2b_finetuned.py`) also calls the HF processor directly and needs the same change.
- **Positions.** HF's `compute_3d_position_ids` is called unchanged (`monkey_patch_forward.py:426`). The cheapest workaround is to mark the K tokens as text (`mm_token_type_ids = 0`) so they get 1-D positions.
- **Training and saving.**
  - After `get_peft_model` (`QV/train/train_sft.py:177`), PEFT freezes everything except LoRA. The script then re-enables only names containing `visual` or `merger`.
  - Other trainable modules are saved to `non_lora_state_dict.bin`, but resuming doesn't reload that file (see [Known issues](#known-issues-found-while-reading-the-code)). Register the Q-Former through `modules_to_save` instead.
- **Inference.** `QV/utils.py:68-72` loads `non_lora_state_dict.bin` into the stock architecture with `strict=False`. Weights for a module the stock class doesn't have are dropped without a warning, so the module must be built before loading.

## 2. Replacing text output with an MLP action head

### Labels and evaluation come first

This part is about the TrafficRuleVQA data and eval harness, not the frameworks.

- **The dataset has no action labels an MLP can learn directly.**
  - `action_next` answers are free text. The validation set has 770 of these questions with 250 different answers, such as "Continue straight, maintaining the same speed.", "Continue straight, slowing down." and "Continue straight, stopping.".
  - A classification head needs these mapped to a fixed set of classes, for example lateral (straight, left, right, lane change) × longitudinal (keep speed, slow down, stop, accelerate).
- **Continuous targets.** `ego_motion` gives position, yaw and speed per frame, but only for the frames the model sees. Future waypoints would need data from after the clip, or the last frames held back as targets.
- **Evaluation.**
  - The eval harness scores text answers (`AnswerResult.raw_response` in `TrafficRuleVQA/05_Eval/01_FullEval/models/base.py`).
  - An action head needs its own metrics (accuracy and macro-F1 for classes, ADE/FDE for waypoints) and its own inference script.
  - Neither framework's inference code returns head outputs.

### In LLaMA-Factory

There is no stage for classification or regression heads. The closest is the reward-model stage (`rm`): a TRL value head that outputs one number per token, trained on pairs of answers (`LF/train/rm/trainer.py:87`). It can't do K-way classification or waypoint regression.

**Short route: stay in the SFT stage.**
1. Add the head in `patch_model` and list it in `additional_target`.
2. Wrap `forward` to read the hidden state at a marker token and skip the vocabulary head.
3. Compute the loss through `compute_loss_func`.
4. Pass the targets as a new dataset column through the converter, the supervised data processor and the collator.

The collator casts every floating-point tensor to bf16 (`LF/data/collator.py:566-568`), which would also cast regression targets. This route can also train text answers and the action head together by adding the two losses.

**Clean route: a new training stage.** About 10 files:
- the stage name (`LF/hparams/finetuning_args.py:460`)
- the dispatch (`LF/train/tuner.py:138-151`)
- a new `train/<stage>/` workflow and trainer
- a dataset processor
- the label column in the dataset parser and converters
- a collator

### In Qwen-VL-Series-Finetune

`QV/train/train_cls.py` and `QV/model/modeling_cls.py` already train `Qwen3VLForSequenceClassification` (`modeling_cls.py:411`):
- **Head.** An optional bridge MLP (`Linear`, `GELU`, `Dropout`; size set with `--mlp_head_dim`) and a linear `score` layer. It reads the hidden state of the last prompt token.
- **Losses.** Cross-entropy, focal, class-balanced cross-entropy and class-balanced focal (`QV/loss/loss_factory.py:5-21`).
- **Training details.** The head can have its own learning rate (`--head_lr`). With LoRA it is saved through `modules_to_save`. Liger is turned off automatically (`QV/train/train_cls.py:248-252`).

**To use it for actions:**
- Convert the data to `{"image": [...], "prompt": "...", "label": "<class>"}`. This format has no `conversations` field.
- Replace the hard-coded `CLASS_2_ID = {"A": 0, "B": 1}` (`QV/dataset/cls_dataset.py:23-26`) and set `--num_labels`.
- Fix the loss bug in [Known issues](#known-issues-found-while-reading-the-code). Otherwise every loss silently becomes plain cross-entropy.
- Write an inference loader. None exists, because `QV/utils.py` only builds text-generating models.

**For regression (waypoints)**, about 50–100 more lines:
- `num_labels == 1` uses a hard-coded `MSELoss`. With more than one output and float labels, the model assumes multi-label BCE, so set `problem_type = "regression"` explicitly.
- Change the label type in `cls_dataset.py` (lines 190 and 243).
- Add MSE and L1 to the loss factory.
- Replace the argmax and F1 metrics.

## 3. Changing the training loss

Both frameworks currently use cross-entropy on the answer tokens only. Prompt and image tokens are masked with -100.

### In LLaMA-Factory

- **Built-in alternatives.** Three, each turned on with one flag: `use_dft_loss`, `use_eaft_loss` and `use_asft_loss` (`LF/hparams/finetuning_args.py:526-538`). The implementations are in `LF/train/trainer_utils.py` (lines 639, 686 and 768).
  - ASFT doesn't work with images here: its reference-model forward passes only `input_ids` and `attention_mask` (`LF/train/sft/trainer.py:150-162`).
- **Adding a loss that needs only logits and labels.** Add a flag, a function in `trainer_utils.py`, and a branch in `LF/train/sft/trainer.py:108-125`. That is about 30 lines, with DFT as the template.
- **Adding a loss that needs images or extra targets.** Override `compute_loss` in a trainer subclass (`LF/train/sft/trainer.py:150`).
- **Memory.**
  - The hook receives the full logits and converts them to fp32.
  - LLaMA-Factory has no Liger support for gemma4, internvl or qwen3_vl (`LF/model/model_utils/liger_kernel.py`), so the current runs already build full logits.
  - At about 3,000 tokens per example, that is roughly 3 GB in fp32 for Gemma's 262,144-token vocabulary, which fits easily in 96 GB.
- **Other loss options.** Label smoothing (`label_smoothing_factor`), `train_on_prompt` and `mask_history`. Preference losses (DPO variants, KTO, ORPO, SimPO) run as separate stages.

### In Qwen-VL-Series-Finetune

- **No hook yet.** There is no `compute_loss` override: the loss comes from the model (`QV/trainer/sft_trainer.py`). A custom loss means adding one.
- **Liger is on by default** (`QV/params.py:152`). During training its fused kernel computes the loss straight from hidden states and returns no logits, so a custom loss can't read them. There are two options:
  - Turn Liger off. The full logits come back. At native resolution this is what ran out of memory in a TrafficRuleVQA smoke test (SLURM job 249, a 69.56 GiB allocation).
  - Compute the hidden states yourself and apply `lm_head` only at the answer positions.

  This Liger behaviour comes from the Liger 0.8.0 docs. Liger isn't installed on the project's training machine (ProArt), so it was not checked against the source.
- **Eval loss.** It comes from `prediction_step` (`QV/trainer/sft_trainer.py:181`), which must use the same loss.
- **Classification losses.** Add one entry to `QV/loss/loss_factory.py`, once the bug below is fixed.
- **GRPO rewards.** Add a function whose name ends in `_reward` to `QV/train/reward_funcs.py`. It is picked up automatically (`QV/utils.py:117`).

## Known issues found while reading the code

| Where | Issue | Effect |
|---|---|---|
| `QV/train/train_cls.py:255` | `model.loss_fn = ...` is set on the PEFT wrapper after `get_peft_model`. The inner model reads its own `self.loss_fn`, which stays `None`. | With `--lora_enable True`, focal and class-balanced losses are silently replaced by plain cross-entropy. Fix: `model.get_base_model().loss_fn = ...` |
| `QV/train/train_sft.py:224`, `QV/trainer/sft_trainer.py:157-179` | Each checkpoint writes `non_lora_state_dict.bin`, but resuming only calls HF `load_adapter` and never reads it | Any trainable module outside LoRA (merger, Q-Former) restarts from its initial weights after a SLURM restart. Use `modules_to_save` instead. |
| `QV/dataset/sft_dataset.py:281` | Truncation is commented out, and `max_seq_length` is never used | No length limit. This let samples of about 123K tokens reach the GPU in a TrafficRuleVQA smoke test (SLURM job 249). |
| `QV/utils.py:68-72` | `load_state_dict(..., strict=False)` | Unknown weights are dropped without a warning |
| `LF/train/sft/trainer.py:150-162` | The ASFT reference forward gets no `pixel_values` | ASFT is text-only |
| `~/venvs/phase4-finetune` on ProArt (TrafficRuleVQA setup, not a framework bug) | LLaMA-Factory is installed in editable mode from `~/projects/ma-vincent/04_Finetuning/01_LLaMAFactory/repo` | Code changes in the TrafficRuleVQA submodule are ignored by that venv. Before changing code, reinstall from the project repo: `uv pip install -e 04_Finetuning/01_LLaMAFactory/repo` |

## Where custom code should live

This section is about the TrafficRuleVQA repo layout.

**LLaMA-Factory** is a submodule there, pinned to `hiyouga/LLaMA-Factory`. Changes inside it can't be pushed with the project repo; they would need a fork, with the submodule pointed at that fork. It is simpler to leave the submodule unchanged and keep the changes in a module of the project repo:

1. Import `llamafactory`.
2. Register or patch what is needed: `register_mm_plugin`, `_register_composite_model`, a wrapped `patch_model`, a trainer subclass.
3. Call `run_exp()` (`LF/train/tuner.py:163`).

The existing `*_no_cudnn.py` wrappers already start training from a Python script in the same way. Two things to remember:
- The eval backend must import the same module before it builds `ChatModel`.
- For multi-GPU runs, start the wrapper with `torchrun` yourself, because `llamafactory-cli` starts its own launcher.

**Qwen-VL-Series-Finetune** is a vendored copy there, so changes go straight into the project repo. List any changed upstream files in `TrafficRuleVQA/04_Finetuning/03_QwenVLSeriesFinetune/UPSTREAM.md`, which currently records none.

**LLaMA-Factory `v1/`** (enabled with `USE_V1=1`) has a plugin system, but it is experimental at this commit and not usable for these runs:
- It can't freeze the vision tower, so LoRA `all` would also put LoRA into the vision tower.
- It has no plugins for losses or model architectures.
