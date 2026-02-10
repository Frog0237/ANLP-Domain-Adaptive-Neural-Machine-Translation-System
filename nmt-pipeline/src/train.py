"""
Training engine for encoder-decoder Transformer NMT models.

Design decisions:
- Supports any HuggingFace seq2seq model (e.g., MarianMT, T5)
- Optional ALiBi patch for Marian to use relative position bias
- FP16 mixed precision: reduces memory usage and improves throughput
- group_by_length (dynamic batching): reduces padding waste and improves throughput
- Multi-split evaluation: simultaneously tracks i.i.d. and o.o.d. generalization
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from .alibi import apply_alibi_to_marian
from .data import prepare_datasets

logger = logging.getLogger(__name__)

# 说明：训练入口按配置完成建模、数据准备、训练与保存。


def build_model(base_model: str, from_pretrained: bool = False, use_alibi: bool = False):
    """
    Initialize a seq2seq model.

    Why random init instead of from_pretrained:
    - Demonstrates full training pipeline capability
    - Tests whether the architecture can learn domain-specific patterns
    - Provides a controlled baseline for comparison with pretrained models

    Args:
        base_model: HuggingFace model identifier for config/tokenizer.
        from_pretrained: If True, load pretrained weights.
        use_alibi: If True, enable ALiBi patching for Marian models.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    config = AutoConfig.from_pretrained(base_model)
    if use_alibi:
        config.use_alibi = True
        config._attn_implementation = "eager"

    if from_pretrained:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model, config=config)
        logger.info(f"Loaded pretrained model from {base_model}")
    else:
        model = AutoModelForSeq2SeqLM.from_config(config)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Initialized model from scratch with {num_params:,} parameters "
            f"({num_params / 1e6:.1f}M)"
        )

    return model, tokenizer


def create_training_args(cfg: dict, output_dir: str) -> Seq2SeqTrainingArguments:
    """
    Build Seq2SeqTrainingArguments from config dict.

    Key hyperparameter choices:
    - lr=5e-5: Standard for Transformer fine-tuning; lower than pretraining
      but sufficient for small corpus convergence
    - weight_decay=0.01: Mild L2 regularization prevents overfitting on small data
    - auto_find_batch_size: Gracefully handles GPU memory limits
    - fp16: Trades negligible precision loss for major memory/speed gains
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=cfg.get("learning_rate", 5e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
        num_train_epochs=cfg.get("num_epochs", 5),
        per_device_train_batch_size=cfg.get("batch_size", 128),
        per_device_eval_batch_size=cfg.get("batch_size", 128),
        auto_find_batch_size=cfg.get("auto_find_batch_size", True),
        save_total_limit=cfg.get("save_total_limit", 1),
        predict_with_generate=False,
        logging_steps=cfg.get("logging_steps", 20),
        save_steps=cfg.get("save_steps", 500),
        eval_steps=cfg.get("eval_steps", 500),
        eval_strategy="steps",
        fp16=cfg.get("fp16", True) and torch.cuda.is_available(),
        fp16_full_eval=cfg.get("fp16_full_eval", True),
        group_by_length=cfg.get("group_by_length", True),
        generation_max_length=cfg.get("generation_max_length", 20),
        report_to=[],
        logging_dir=os.path.join(output_dir, "logs"),
    )


def train(config_path: str):
    """
    Execute full training pipeline from config file.

    Pipeline: load config → build model → prepare data → train → save
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"].get("seed", 42))

    # Build model
    use_alibi = cfg["model"].get("use_alibi", False)
    model, tokenizer = build_model(
        cfg["model"]["base_model"],
        from_pretrained=cfg["model"].get("from_pretrained", False),
        use_alibi=use_alibi,
    )
    if use_alibi:
        if getattr(model.config, "model_type", "") != "marian":
            raise ValueError("ALiBi patching is only supported for Marian models.")
        alibi_on_encoder = cfg["model"].get("alibi_on_encoder", True)
        alibi_on_decoder = cfg["model"].get("alibi_on_decoder", True)
        # 说明：仅 Marian 支持 ALiBi，且可按需启用编码器/解码器。
        model = apply_alibi_to_marian(
            model,
            use_on_encoder=alibi_on_encoder,
            use_on_decoder=alibi_on_decoder,
        )

    # Prepare data
    # Optional task prefix supports T5-style models.
    datasets = prepare_datasets(
        dataset_name=cfg["data"]["dataset_name"],
        tokenizer=tokenizer,
        splits=cfg["data"]["splits"],
        source_lang=cfg["data"]["source_lang"],
        target_lang=cfg["data"]["target_lang"],
        source_prefix=cfg["data"].get("source_prefix"),
    )

    # Build trainer
    output_dir = cfg["output"]["model_dir"]
    training_args = create_training_args(cfg["training"], output_dir)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    eval_datasets = {
        k: v for k, v in datasets.items() if k != "train"
    }

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=eval_datasets,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
    logger.info(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train NMT model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train(args.config)


if __name__ == "__main__":
    main()

