"""
Stage2 End-to-End Model: Combines glim_parallel encoder with stage2 T5 decoder.

This model computes Zi embeddings on-the-fly from raw EEG signals using the glim_parallel
encoder, then uses these embeddings for text reconstruction with T5 decoder.
"""

import torch
import torch.nn as nn
import lightning as L
from typing import Literal, Dict, Any, Optional
from transformers import get_cosine_with_min_lr_schedule_with_warmup_lr_rate

from .modules import PromptEmbedder, EEGEncoder, Aligner
from .stage2_model import Stage2ReconstructionModel


class Stage2E2EModel(L.LightningModule):
    """
    End-to-end stage2 model that combines:
    - GLIM_PARALLEL encoder (PromptEmbedder + EEGEncoder + Aligner)
    - Stage2ReconstructionModel (T5 decoder)

    Allows joint training of encoder and decoder for text reconstruction.
    """

    SUPPORTED_TEXT_MODELS = Literal["google/flan-t5-xl", "google/flan-t5-large",
                                    "facebook/bart-large-cnn", "jbochi/madlad400-3b-mt"]

    def __init__(
        self,
        # Encoder params (from glim_parallel)
        input_eeg_len: int = 1280,
        hidden_eeg_len: int = 96,
        input_text_len: int = 96,
        input_dim: int = 128,
        hidden_dim: int = 128,
        embed_dim: int = 1024,
        prompt_nums: tuple = (3, 3, 31),
        prompt_dropout_probs: tuple = (0.0, 0.0, 0.0),
        evaluate_prompt_embed: Literal['zero', 'sum', 'mean', 'src'] = 'src',
        n_in_blocks: int = 6,
        n_out_blocks: int = 6,
        in_temporal_modulate: bool = True,
        out_is_causal: bool = True,
        use_prompt: bool = True,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        use_channel_weights: bool = False,
        commitment_loss_key: Literal['mse', 'kl_div'] = 'mse',
        use_y_mask: bool = False,

        # Decoder params (from stage2)
        text_model_id: SUPPORTED_TEXT_MODELS = "google/flan-t5-large",
        model_cache_dir: str = None,
        freeze_strategy: str = "full_freeze_llm",
        lora_rank: int = 8,
        attention_mask_type: str = "bidirectional",
        use_ei: bool = True,
        use_projector: bool = True,
        use_metadata: bool = True,

        # Classification labels
        sentiment_labels: list = None,
        topic_labels: list = None,

        # Trainability control
        encoder_trainable_mode: str = "aligner_only",

        # Training params
        lr: float = 1e-4,
        min_lr: float = 1e-6,
        warmup_epochs: int = 0,
        batch_size: int = 16,
        device_id: str = "cuda:0",
    ):
        super().__init__()

        # Store parameters
        self.input_text_len = input_text_len
        self.eval_pembed = evaluate_prompt_embed
        self.use_prompt = use_prompt
        self.embed_dim = embed_dim
        self.text_model_id = text_model_id
        self.model_cache_dir = model_cache_dir
        self.batch_size = batch_size
        self.use_metadata = use_metadata
        self.encoder_trainable_mode = encoder_trainable_mode

        # Optimizer parameters
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs

        # Set default labels
        if sentiment_labels is None:
            sentiment_labels = ['non_neutral', 'neutral']
        if topic_labels is None:
            topic_labels = ['Biographies and Factual Knowledge', 'Movie Reviews and Sentiment']
        self.sentiment_labels = sentiment_labels
        self.topic_labels = topic_labels

        # Prompt configuration
        self.prompt_keys = {
            'task': ['<UNK>'] + ['<NR>', '<TSR>'],
            'dataset': ['<UNK>'] + ['ZuCo1', 'ZuCo2'],
            'subject': ['<UNK>'] + ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN',
                                    'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH',
                                    'YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR',
                                    'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS',
                                    'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL'],
        }

        # Build encoder components
        self.p_embedder = PromptEmbedder(input_dim, prompt_nums, prompt_dropout_probs, self.prompt_keys)
        self.eeg_encoder = EEGEncoder(
            input_eeg_len, hidden_eeg_len, input_dim, hidden_dim,
            0, n_in_blocks, n_out_blocks,
            in_temporal_modulate, out_is_causal,
            num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
            use_channel_weights=use_channel_weights
        )
        self.aligner = Aligner(hidden_dim, embed_dim, num_heads, dropout, commitment_loss_key, use_y_mask)
        self.use_y_mask = use_y_mask

        # Build decoder
        self.stage2_model = Stage2ReconstructionModel(
            model_name=text_model_id,
            freeze_strategy=freeze_strategy,
            lora_rank=lora_rank,
            attention_mask_type=attention_mask_type,
            use_ei=use_ei,
            use_projector=use_projector,
            sentiment_labels=sentiment_labels,
            topic_labels=topic_labels,
            device=device_id
        )

        # Set encoder trainability
        self._set_encoder_trainability(encoder_trainable_mode)

        self.save_hyperparameters()

    def _set_encoder_trainability(self, mode: str):
        """Control which encoder components are trainable."""
        if mode == "all":
            # Train everything
            pass
        elif mode == "encoder_aligner":
            # Freeze PromptEmbedder
            for param in self.p_embedder.parameters():
                param.requires_grad = False
        elif mode == "aligner_only":
            # Freeze PromptEmbedder + EEGEncoder
            for param in self.p_embedder.parameters():
                param.requires_grad = False
            for param in self.eeg_encoder.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unknown encoder_trainable_mode: {mode}")

    def load_stage1_checkpoint(self, checkpoint_path: str):
        """Load glim_parallel weights into encoder components."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # Extract encoder weights
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('p_embedder.') or key.startswith('eeg_encoder.') or key.startswith('aligner.'):
                encoder_state_dict[key] = value

        # Load with strict=False to allow missing keys (task heads, stage2 model)
        missing_keys, unexpected_keys = self.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded stage1 checkpoint from {checkpoint_path}")
        print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    def _encode_labels(self, labels: list, label_names: list) -> torch.Tensor:
        """Convert label names to indices."""
        indices = []
        for label in labels:
            try:
                idx = label_names.index(label)
            except ValueError:
                idx = 0  # Default to first label if not found
            indices.append(idx)
        return torch.tensor(indices, device=self.device)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """End-to-end forward pass: EEG → encoder → Zi/ei → T5 → loss."""
        # Extract inputs
        eeg = batch['eeg']
        eeg_mask = batch['mask']
        prompts = batch['prompt']
        target_text = batch['input text']

        # Encode prompts
        prompt_ids = self.p_embedder.encode(prompts, device=self.device)
        prompt_embed = self.p_embedder(prompt_ids, self.eval_pembed)

        # Encode EEG
        eeg_hiddens, _ = self.eeg_encoder(eeg, eeg_mask, prompt_embed)

        # Get Zi and ei through aligner
        Zi, ei = self.aligner.embed_eeg(eeg_hiddens, None)

        # Get labels from batch
        sentiment_labels = batch['sentiment label']
        topic_labels = batch['topic_label']
        length = batch['length']
        surprisal = batch['surprisal']

        # Convert labels to indices
        sentiment_idx = self._encode_labels(sentiment_labels, self.sentiment_labels)
        topic_idx = self._encode_labels(topic_labels, self.topic_labels)

        # Build prompt dicts if using metadata
        prompt_dicts = None
        if self.use_metadata:
            prompt_dicts = [
                {
                    'task': prompts[0][i],
                    'dataset': prompts[1][i],
                    'subject': prompts[2][i]
                }
                for i in range(len(sentiment_idx))
            ]

        # Stage2 reconstruction
        loss = self.stage2_model(
            label_task1=sentiment_idx,
            label_task2=topic_idx,
            length=length,
            surprisal=surprisal,
            ei=ei,
            Zi=Zi,
            target_text=target_text,
            prompt_dicts=prompt_dicts
        )

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss = self(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss = self(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss = self(batch)
        self.log('test/loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        """Configure optimizer with separate param groups."""
        encoder_params = []
        decoder_params = []

        # Collect trainable encoder params
        if self.encoder_trainable_mode == "all":
            encoder_params.extend(list(self.p_embedder.parameters()))
            encoder_params.extend(list(self.eeg_encoder.parameters()))
            encoder_params.extend(list(self.aligner.parameters()))
        elif self.encoder_trainable_mode == "encoder_aligner":
            encoder_params.extend(list(self.eeg_encoder.parameters()))
            encoder_params.extend(list(self.aligner.parameters()))
        elif self.encoder_trainable_mode == "aligner_only":
            encoder_params.extend(list(self.aligner.parameters()))

        # Filter only trainable params
        encoder_params = [p for p in encoder_params if p.requires_grad]

        # Collect trainable decoder params
        decoder_params = [p for p in self.stage2_model.parameters() if p.requires_grad]

        # Build param groups
        param_groups = []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': self.lr})
        if decoder_params:
            param_groups.append({'params': decoder_params, 'lr': self.lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # Scheduler (cosine with warmup)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.warmup_epochs * total_steps / self.trainer.max_epochs) if self.trainer.max_epochs > 0 else 0

        scheduler = get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr=self.min_lr
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
