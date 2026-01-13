"""
Training script for end-to-end stage2 model.
Combines glim_parallel encoder with stage2 T5 decoder for text reconstruction.
"""

import os
import sys
import atexit
import argparse
from datetime import datetime
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from data.datamodule import GLIMDataModule
from model.stage2_e2e_model import Stage2E2EModel


class TeeLogger:
    """Logger that writes to both stdout/stderr and a file."""
    _instances = []

    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'w')
        TeeLogger._instances.append(self)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    def close(self):
        if self.log and not self.log.closed:
            self.log.close()

    @classmethod
    def close_all(cls):
        for instance in cls._instances:
            instance.close()


atexit.register(TeeLogger.close_all)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train end-to-end Stage 2 model with trainable encoder'
    )

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the merged dataframe with raw EEG data')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='Batch size for validation')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')

    # Checkpoint arguments
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help='Path to stage1 (glim_parallel) checkpoint')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to resume training from')

    # Encoder arguments
    parser.add_argument('--input_eeg_len', type=int, default=1280)
    parser.add_argument('--hidden_eeg_len', type=int, default=96)
    parser.add_argument('--input_text_len', type=int, default=96)
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--n_in_blocks', type=int, default=6)
    parser.add_argument('--n_out_blocks', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_channel_weights', action='store_true')
    parser.add_argument('--do_not_use_prompt', action='store_true')

    # Decoder arguments
    parser.add_argument('--text_model', type=str, default='google/flan-t5-large',
                        help='Text model to use')
    parser.add_argument('--freeze_strategy', type=str, default='full_freeze_llm',
                        choices=['full_freeze_llm', 'lora', 'full_trainable_llm'],
                        help='T5 freeze strategy')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--use_ei', action='store_true', default=True,
                        help='Use global EEG feature (ei)')
    parser.add_argument('--use_projector', action='store_true', default=True,
                        help='Use trainable projection layer')
    parser.add_argument('--use_metadata', action='store_true', default=True,
                        help='Include metadata in prompts')

    # Label arguments
    parser.add_argument('--sentiment_labels', type=str, nargs='+',
                        default=['non_neutral', 'neutral'])
    parser.add_argument('--topic_labels', type=str, nargs='+',
                        default=['Biographies and Factual Knowledge', 'Movie Reviews and Sentiment'])

    # Trainability control
    parser.add_argument('--encoder_trainable_mode', type=str, default='aligner_only',
                        choices=['all', 'encoder_aligner', 'aligner_only'],
                        help='Which encoder components to train')

    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--devices', type=int, nargs='+', default=[0],
                        help='GPU device IDs')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--strategy', type=str, default='auto',
                        help='Training strategy (auto, ddp, etc.)')
    parser.add_argument('--precision', type=str, default='bf16-mixed',
                        help='Training precision')

    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--experiment_name', type=str, default='stage2_e2e')

    return parser.parse_args()


def main():
    args = parse_args()

    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{args.experiment_name}_{timestamp}'
    run_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Setup logging
    sys.stdout = TeeLogger(os.path.join(run_dir, 'training.log'), sys.stdout)
    sys.stderr = TeeLogger(os.path.join(run_dir, 'training_error.log'), sys.stderr)

    print(f"Starting training: {run_name}")
    print(f"Arguments: {args}")

    # Initialize model
    model = Stage2E2EModel(
        # Encoder params
        input_eeg_len=args.input_eeg_len,
        hidden_eeg_len=args.hidden_eeg_len,
        input_text_len=args.input_text_len,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        n_in_blocks=args.n_in_blocks,
        n_out_blocks=args.n_out_blocks,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_channel_weights=args.use_channel_weights,
        use_prompt=not args.do_not_use_prompt,

        # Decoder params
        text_model_id=args.text_model,
        freeze_strategy=args.freeze_strategy,
        lora_rank=args.lora_rank,
        use_ei=args.use_ei,
        use_projector=args.use_projector,
        use_metadata=args.use_metadata,

        # Labels
        sentiment_labels=args.sentiment_labels,
        topic_labels=args.topic_labels,

        # Trainability
        encoder_trainable_mode=args.encoder_trainable_mode,

        # Training
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        device_id=f"cuda:{args.devices[0]}" if args.accelerator == 'gpu' else 'cpu',
    )

    # Load stage1 checkpoint
    print(f"\nLoading stage1 checkpoint from: {args.stage1_checkpoint}")
    model.load_stage1_checkpoint(args.stage1_checkpoint)

    # Initialize datamodule
    datamodule = GLIMDataModule(
        data_path=args.data_path,
        bsz_train=args.batch_size,
        bsz_val=args.val_batch_size,
        bsz_test=args.test_batch_size,
        num_workers=args.num_workers,
        classification_label_keys=['sentiment label', 'topic_label'],
        regression_label_keys=['length', 'surprisal'],
        use_weighted_sampler=False,
    )

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, 'checkpoints'),
        filename='model-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=TensorBoardLogger(run_dir, name='tensorboard'),
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    if args.resume_checkpoint:
        trainer.fit(model, datamodule, ckpt_path=args.resume_checkpoint)
    else:
        trainer.fit(model, datamodule)

    # Test
    print("\n" + "="*80)
    print("Starting testing...")
    print("="*80 + "\n")

    trainer.test(model, datamodule)

    print(f"\nTraining completed. Logs saved to: {run_dir}")


if __name__ == '__main__':
    main()
