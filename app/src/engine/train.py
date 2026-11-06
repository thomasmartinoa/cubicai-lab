# CubicAI Lab trainer: QLoRA with Unsloth. Run by the app as `python train.py config.json`.
# Every line it prints that starts with "CUBIC " is a JSON event for the app; everything else is
# log. Writing a file named STOP into the output folder ends training early and keeps the adapter.
import inspect
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def emit(event, **kw):
    print("CUBIC " + json.dumps({"event": event, **kw}), flush=True)


def only_known(fn, kwargs):
    """Keeps the arguments this version of a library accepts (TRL renames them now and then)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


# Chat-template markers that start a user turn and a reply, for training on replies only.
MARKERS = [
    ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),  # Llama 3
    ("<|im_start|>user\n", "<|im_start|>assistant\n"),  # Qwen / ChatML
    ("[INST]", "[/INST]"),  # Mistral
]


def main():
    cfg = json.load(open(sys.argv[1]))
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    stop_file = os.path.join(out, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    started = time.time()

    emit("status", text="Loading Unsloth")
    from unsloth import FastLanguageModel  # before transformers, so its patches apply
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        emit("error", text="PyTorch can't see an NVIDIA GPU, so it can't train here.")
        return 1

    emit("status", text="Loading " + cfg["base"] + " (the first time downloads it)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base"], max_seq_length=cfg["max_seq"], load_in_4bit=True, dtype=None
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    rows = [json.loads(l) for l in open(cfg["data"], encoding="utf-8") if l.strip()]
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    emit("status", text=f"{len(texts)} examples ready")
    data = Dataset.from_dict({"text": texts})

    class Events(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                emit("loss", step=state.global_step, total=state.max_steps, loss=float(logs["loss"]),
                     lr=float(logs.get("learning_rate", 0.0)), epoch=float(logs.get("epoch", 0.0)))

        def on_step_end(self, args, state, control, **kw):
            emit("progress", step=state.global_step, total=state.max_steps)
            if os.path.exists(stop_file):
                emit("status", text="Stopping after this step and keeping what was learned")
                control.should_training_stop = True

    args = SFTConfig(**only_known(SFTConfig.__init__, dict(
        output_dir=os.path.join(out, "checkpoints"),
        dataset_text_field="text",
        max_seq_length=cfg["max_seq"],
        max_length=cfg["max_seq"],
        per_device_train_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        warmup_steps=min(10, max(1, len(texts) // 20)),
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
    )))
    # Newer TRL calls the tokenizer `processing_class`; pass it under the one name this version knows.
    tok_name = "processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else "tokenizer"
    trainer = SFTTrainer(model=model, train_dataset=data, args=args, callbacks=[Events()], **{tok_name: tokenizer})

    if cfg.get("replies_only", True):
        marks = next(((u, a) for u, a in MARKERS if u in texts[0] and a in texts[0]), None)
        if marks:
            from unsloth.chat_templates import train_on_responses_only
            trainer = train_on_responses_only(trainer, instruction_part=marks[0], response_part=marks[1])
            emit("status", text="Learning from the replies only")
        else:
            emit("status", text="Unknown chat template: learning from whole conversations")

    emit("status", text="Training")
    trainer.train()
    adapter = os.path.join(out, "adapter")
    model.save_pretrained(adapter)
    emit("done", adapter=adapter, secs=time.time() - started)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 - everything goes back to the app as text
        text = f"{type(e).__name__}: {e}"
        if "out of memory" in text.lower():
            text = "The GPU ran out of memory. Lower the max length or the batch size, or use a smaller model. (" + text + ")"
        emit("error", text=text)
        sys.exit(1)
