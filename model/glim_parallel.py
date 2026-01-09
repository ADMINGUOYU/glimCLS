"""
GLIM_PARALLEL: Multi-task GLIM model with parallel classification and regression heads.

This model extends GLIM_CLS to support multiple parallel tasks:
- 2 classification tasks (sentiment, topic)
- 2 regression tasks (length, surprisal)
All tasks share the same GLIM encoder and operate on the same eeg_emb vector.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Literal, Dict, Any
from copy import deepcopy
from torchmetrics.functional.classification import multiclass_accuracy
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_cosine_with_min_lr_schedule_with_warmup_lr_rate
from transformers.modeling_outputs import BaseModelOutput

from .modules import PromptEmbedder, EEGEncoder, Aligner


class GLIM_PARALLEL(L.LightningModule):
    """
    Multi-task GLIM model with 4 parallel heads (2 classification + 2 regression).
    """

    SUPPORTED_TEXT_MODELS = Literal["google/flan-t5-xl", "google/flan-t5-large",
                                    "facebook/bart-large-cnn", "jbochi/madlad400-3b-mt"]

    def __init__(self,
                 # EEG Encoder arguments
                 input_eeg_len: int = 1280,
                 hidden_eeg_len: int = 96,
                 input_text_len: int = 96,
                 tgt_text_len: int = 64,
                 input_dim: int = 128,
                 hidden_dim: int = 128,
                 embed_dim: int = 1024,
                 text_model_id: SUPPORTED_TEXT_MODELS = "google/flan-t5-large",
                 model_cache_dir: str = None,
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
                 clip_loss_weight: float = 0.5,
                 lm_loss_weight: float = 0.0,
                 commitment_loss_weight: float = 0.0,
                 commitment_loss_key: Literal['mse', 'kl_div'] = 'mse',
                 use_y_mask: bool = False,
                 batch_size: int = 24,

                 # Multi-task arguments
                 classification_tasks: dict = None,
                 regression_tasks: list = None,
                 mlp_hidden_dims: list = None,
                 mlp_dropout: float = 0.3,
                 freeze_encoder: bool = False,

                 # Loss weights
                 sentiment_loss_weight: float = 0.25,
                 topic_loss_weight: float = 0.25,
                 length_loss_weight: float = 0.25,
                 surprisal_loss_weight: float = 0.25,
                 use_class_weights: bool = False,

                 # Optimizer arguments
                 lr: float = 1e-4,
                 min_lr: float = 1e-6,
                 warmup_epochs: int = 0,
                 ):
        super().__init__()

        self.strict_loading = False

        # Set defaults
        if classification_tasks is None:
            classification_tasks = {
                'sentiment': ['non_neutral', 'neutral'],
                'topic': ['Biographies and Factual Knowledge', 'Movie Reviews and Sentiment']
            }
        if regression_tasks is None:
            regression_tasks = ['length', 'surprisal']
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [512, 256]

        # Store parameters
        self.input_text_len = input_text_len
        self.tgt_text_len = tgt_text_len
        self.eval_pembed = evaluate_prompt_embed
        self.use_prompt = use_prompt
        self.embed_dim = embed_dim
        self.text_model_id = text_model_id
        self.model_cache_dir = model_cache_dir
        self.batch_size = batch_size

        # Loss weights
        self.clip_loss_weight = clip_loss_weight
        self.lm_loss_weight = lm_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.sentiment_loss_weight = sentiment_loss_weight
        self.topic_loss_weight = topic_loss_weight
        self.length_loss_weight = length_loss_weight
        self.surprisal_loss_weight = surprisal_loss_weight

        # Optimizer parameters
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs

        # Multi-task parameters
        self.classification_tasks = classification_tasks
        self.regression_tasks = regression_tasks
        self.freeze_encoder = freeze_encoder
        self.use_class_weights = use_class_weights

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
        self.eeg_encoder = EEGEncoder(input_eeg_len, hidden_eeg_len, input_dim, hidden_dim,
                                      0, n_in_blocks, n_out_blocks,
                                      in_temporal_modulate, out_is_causal,
                                      num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                                      use_channel_weights=use_channel_weights)
        self.aligner = Aligner(hidden_dim, embed_dim, num_heads, dropout, commitment_loss_key, use_y_mask)
        self.use_y_mask = use_y_mask

        # Build 4 MLP heads
        self.sentiment_classifier = self._build_mlp_head(
            embed_dim, len(classification_tasks['sentiment']), mlp_hidden_dims, mlp_dropout
        )
        self.topic_classifier = self._build_mlp_head(
            embed_dim, len(classification_tasks['topic']), mlp_hidden_dims, mlp_dropout
        )
        self.length_regressor = self._build_mlp_head(
            embed_dim, 1, mlp_hidden_dims, mlp_dropout
        )
        self.surprisal_regressor = self._build_mlp_head(
            embed_dim, 1, mlp_hidden_dims, mlp_dropout
        )

        # Test outputs for confusion matrices
        self.test_step_outputs = {'sentiment': [], 'topic': []}
        self.val_step_outputs = {'sentiment': [], 'topic': []}

        self.save_hyperparameters(logger=True)

    def _build_mlp_head(self, input_dim, output_dim, hidden_dims, dropout):
        """Build a single MLP head with shared architecture."""
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def _compute_class_weights(self, label_counts: dict, num_classes: int) -> torch.Tensor:
        """Compute class weights using inverse frequency formula.

        Args:
            label_counts: Dict mapping class_id -> count
            num_classes: Total number of classes

        Returns:
            Tensor of shape (num_classes,) with weights for each class
        """
        total_samples = sum(label_counts.values())
        weights = torch.zeros(num_classes)
        for class_id, count in label_counts.items():
            weights[class_id] = total_samples / (num_classes * count)
        return weights

    def set_class_weights(self, sentiment_counts: dict, topic_counts: dict):
        """Set class weights for both classification tasks.

        Args:
            sentiment_counts: Dict mapping sentiment class_id -> count
            topic_counts: Dict mapping topic class_id -> count
        """
        if self.use_class_weights:
            sentiment_weights = self._compute_class_weights(
                sentiment_counts, len(self.classification_tasks['sentiment'])
            )
            topic_weights = self._compute_class_weights(
                topic_counts, len(self.classification_tasks['topic'])
            )
            self.register_buffer('sentiment_class_weights', sentiment_weights)
            self.register_buffer('topic_class_weights', topic_weights)
        else:
            self.register_buffer('sentiment_class_weights', None)
            self.register_buffer('topic_class_weights', None)

    def set_regression_stats(self, length_mean: float, length_std: float,
                            surprisal_mean: float, surprisal_std: float):
        """Set normalization statistics for regression tasks.

        Args:
            length_mean: Mean of length values in training data
            length_std: Std of length values in training data
            surprisal_mean: Mean of surprisal values in training data
            surprisal_std: Std of surprisal values in training data
        """
        self.register_buffer('length_mean', torch.tensor(length_mean, dtype=torch.float32))
        self.register_buffer('length_std', torch.tensor(length_std, dtype=torch.float32))
        self.register_buffer('surprisal_mean', torch.tensor(surprisal_mean, dtype=torch.float32))
        self.register_buffer('surprisal_std', torch.tensor(surprisal_std, dtype=torch.float32))

    def setup(self, stage):
        """Setup the text model (T5) using bfloat16 by default."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Print cache directory info
        if self.model_cache_dir:
            print(f"Loading model from cache directory: {self.model_cache_dir}")
            # Check if cache exists to use local_files_only
            import pathlib
            cache_path = pathlib.Path(self.model_cache_dir)
            model_cache_exists = (cache_path / f"models--{self.text_model_id.replace('/', '--')}").exists()
            use_local_only = model_cache_exists
            if use_local_only:
                print("Using cached files only (no network access)")
        else:
            print("Loading model from default cache: ~/.cache/huggingface")
            use_local_only = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.text_model_id,
            cache_dir=self.model_cache_dir,
            local_files_only=use_local_only
        )
        self.text_model = T5ForConditionalGeneration.from_pretrained(
            self.text_model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            cache_dir=self.model_cache_dir,
            local_files_only=use_local_only
        ).requires_grad_(False)
        assert self.embed_dim == self.text_model.config.d_model

        if self.freeze_encoder:
            for param in self.p_embedder.parameters():
                param.requires_grad = False
            for param in self.eeg_encoder.parameters():
                param.requires_grad = False
            for param in self.aligner.parameters():
                param.requires_grad = False

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Remove text model weights from checkpoint to save space."""
        for key in deepcopy(list(checkpoint['state_dict'].keys())):
            if 'text_model' in key:
                checkpoint['state_dict'].pop(key)

    def configure_optimizers(self):
        """Configure optimizer with Cosine LR scheduler with warmup."""
        encoder_params = []
        classifier_params = []
        classifier_params.extend(list(self.sentiment_classifier.parameters()))
        classifier_params.extend(list(self.topic_classifier.parameters()))
        classifier_params.extend(list(self.length_regressor.parameters()))
        classifier_params.extend(list(self.surprisal_regressor.parameters()))

        if not self.freeze_encoder:
            encoder_params.extend(list(self.p_embedder.parameters()))
            encoder_params.extend(list(self.eeg_encoder.parameters()))
            encoder_params.extend(list(self.aligner.parameters()))

        param_groups = []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': self.lr})
        param_groups.append({'params': classifier_params, 'lr': self.lr})

        optimizer = torch.optim.Adam(param_groups)

        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps / self.trainer.max_epochs
        warmup_steps = int(self.warmup_epochs * steps_per_epoch)

        print(f"Scheduler config: {warmup_steps} warmup steps ({self.warmup_epochs} epochs × {steps_per_epoch:.1f} steps/epoch)")

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

    def tokenize(self, texts: list, max_length: int):
        """Tokenize text inputs."""
        inputs = self.tokenizer(texts, max_length=max_length, padding='max_length',
                                truncation=True, return_tensors="pt")
        ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)
        return ids, mask

    def encode_classification_labels(self, labels: list, valid_labels: list, ignore_idx: int = -1):
        """Encode string labels to integer IDs for a specific classification task."""
        label_ids = []
        for label in labels:
            if label in valid_labels:
                label_id = valid_labels.index(label)
            else:
                label_id = ignore_idx
            label_ids.append(label_id)
        return torch.tensor(label_ids, dtype=torch.int64, device=self.device)

    def encode_regression_labels(self, labels: list, task: str):
        """Convert list of values to normalized float tensor.

        Args:
            labels: List of raw label values
            task: Task name ('length' or 'surprisal') for normalization
        """
        values = [float(label) for label in labels]
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        # Normalize using task-specific statistics
        if task == 'length' and hasattr(self, 'length_mean'):
            values_tensor = (values_tensor - self.length_mean) / self.length_std
        elif task == 'surprisal' and hasattr(self, 'surprisal_mean'):
            values_tensor = (values_tensor - self.surprisal_mean) / self.surprisal_std

        return values_tensor

    def denormalize_predictions(self, preds: torch.Tensor, task: str):
        """Denormalize predictions back to original scale.

        Args:
            preds: Normalized predictions
            task: Task name ('length' or 'surprisal')
        """
        if task == 'length' and hasattr(self, 'length_mean'):
            return preds * self.length_std + self.length_mean
        elif task == 'surprisal' and hasattr(self, 'surprisal_mean'):
            return preds * self.surprisal_std + self.surprisal_mean
        return preds

    def encode_text(self, src_ids, src_mask):
        """Encode text using the T5 encoder."""
        text_encoder = self.text_model.get_encoder()
        with torch.no_grad():
            outputs = text_encoder(input_ids=src_ids,
                                   attention_mask=src_mask,
                                   return_dict=True)
        hidden_states = outputs['last_hidden_state']
        return hidden_states, src_mask

    def text_decoder_forward(self, src_embeds, src_mask, tgt_ids):
        """Compute language model loss using the T5 decoder."""
        labels = tgt_ids.detach().clone()
        labels.masked_fill_(labels == self.text_model.config.pad_token_id, -100)
        mask = src_mask if (self.use_y_mask and self.training) else None
        outputs = self.text_model(encoder_outputs=BaseModelOutput(src_embeds),
                                  attention_mask=mask,
                                  labels=labels)
        loss = outputs['loss']
        return loss

    def shared_forward(self, batch):
        """Shared forward pass for all tasks."""
        eeg = batch['eeg']
        eeg_mask = batch['mask']
        prompts = batch['prompt']
        input_text = batch['input text']
        target_text = batch['target text']

        # Handle prompts
        if self.use_prompt is False:
            batch_size = eeg.shape[0]
            prompts = [['<UNK>'] * batch_size, ['<UNK>'] * batch_size, ['<UNK>'] * batch_size]

        # Encode prompts and EEG
        prompt_ids = self.p_embedder.encode(prompts, device=self.device)
        prompt_embed = self.p_embedder(prompt_ids, self.eval_pembed)

        eeg_hiddens, _ = self.eeg_encoder(eeg, eeg_mask, prompt_embed)

        # Encode text for alignment loss
        input_ids, input_mask = self.tokenize(input_text, self.input_text_len)
        input_text_embeds, hidden_text_mask = self.encode_text(input_ids, input_mask)

        # Alignment losses
        (loss_clip, logits_clip, loss_commitment,
         eeg_embeds, eeg_emb, input_text_emb) = self.aligner.forward(eeg_hiddens, input_text_embeds, hidden_text_mask)

        # Language model loss
        if self.lm_loss_weight > 0:
            tgt_ids, _ = self.tokenize(target_text, self.tgt_text_len)
            loss_lm = self.text_decoder_forward(eeg_embeds, hidden_text_mask, tgt_ids)
        else:
            loss_lm = torch.tensor(0.0, device=self.device)

        # === CLASSIFICATION TASKS ===
        # Sentiment classification
        sentiment_labels = batch['sentiment label']
        sentiment_ids = self.encode_classification_labels(
            sentiment_labels, self.classification_tasks['sentiment']
        )
        sentiment_logits = self.sentiment_classifier(eeg_emb)
        loss_sentiment = F.cross_entropy(
            sentiment_logits, sentiment_ids,
            weight=self.sentiment_class_weights if self.use_class_weights else None,
            ignore_index=-1
        )
        acc_sentiment = multiclass_accuracy(
            sentiment_logits, sentiment_ids,
            num_classes=len(self.classification_tasks['sentiment']),
            ignore_index=-1
        )

        # Topic classification
        topic_labels = batch['topic_label']
        topic_ids = self.encode_classification_labels(
            topic_labels, self.classification_tasks['topic']
        )
        topic_logits = self.topic_classifier(eeg_emb)
        loss_topic = F.cross_entropy(
            topic_logits, topic_ids,
            weight=self.topic_class_weights if self.use_class_weights else None,
            ignore_index=-1
        )
        acc_topic = multiclass_accuracy(
            topic_logits, topic_ids,
            num_classes=len(self.classification_tasks['topic']),
            ignore_index=-1
        )

        # === REGRESSION TASKS ===
        # Length regression
        length_labels = batch['length']
        length_values = self.encode_regression_labels(length_labels, 'length')
        length_preds = self.length_regressor(eeg_emb).squeeze(-1)
        loss_length = F.mse_loss(length_preds, length_values)
        # Denormalize for MAE in original scale
        length_preds_denorm = self.denormalize_predictions(length_preds, 'length')
        length_values_denorm = self.denormalize_predictions(length_values, 'length')
        mae_length = F.l1_loss(length_preds_denorm, length_values_denorm)

        # Surprisal regression
        surprisal_labels = batch['surprisal']
        surprisal_values = self.encode_regression_labels(surprisal_labels, 'surprisal')
        surprisal_preds = self.surprisal_regressor(eeg_emb).squeeze(-1)
        loss_surprisal = F.mse_loss(surprisal_preds, surprisal_values)
        # Denormalize for MAE in original scale
        surprisal_preds_denorm = self.denormalize_predictions(surprisal_preds, 'surprisal')
        surprisal_values_denorm = self.denormalize_predictions(surprisal_values, 'surprisal')
        mae_surprisal = F.l1_loss(surprisal_preds_denorm, surprisal_values_denorm)

        # === TOTAL LOSS ===
        loss = (loss_clip * self.clip_loss_weight +
                loss_lm * self.lm_loss_weight +
                loss_commitment * self.commitment_loss_weight +
                loss_sentiment * self.sentiment_loss_weight +
                loss_topic * self.topic_loss_weight +
                loss_length * self.length_loss_weight +
                loss_surprisal * self.surprisal_loss_weight)

        return {
            'total_loss': loss,
            'loss_clip': loss_clip,
            'loss_lm': loss_lm,
            'loss_commitment': loss_commitment,
            'loss_sentiment': loss_sentiment,
            'loss_topic': loss_topic,
            'loss_length': loss_length,
            'loss_surprisal': loss_surprisal,
            'acc_sentiment': acc_sentiment,
            'acc_topic': acc_topic,
            'mae_length': mae_length,
            'mae_surprisal': mae_surprisal,
            'sentiment_logits': sentiment_logits,
            'sentiment_ids': sentiment_ids,
            'topic_logits': topic_logits,
            'topic_ids': topic_ids,
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        output = self.shared_forward(batch)

        self.log('train/loss', output['total_loss'], prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/acc_sentiment', output['acc_sentiment'], prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/acc_topic', output['acc_topic'], prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/mae_length', output['mae_length'], sync_dist=True, batch_size=self.batch_size)
        self.log('train/mae_surprisal', output['mae_surprisal'], sync_dist=True, batch_size=self.batch_size)

        self.log('train/loss_clip', output['loss_clip'], sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_lm', output['loss_lm'], sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_commitment', output['loss_commitment'], sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_sentiment', output['loss_sentiment'], sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_topic', output['loss_topic'], sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_length', output['loss_length'], sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_surprisal', output['loss_surprisal'], sync_dist=True, batch_size=self.batch_size)

        return output['total_loss']

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        with torch.no_grad():
            output = self.shared_forward(batch)

        # Store predictions for epoch-end accuracy computation
        sentiment_preds = torch.argmax(output['sentiment_logits'], dim=1)
        topic_preds = torch.argmax(output['topic_logits'], dim=1)

        self.val_step_outputs['sentiment'].append({
            'predictions': sentiment_preds.detach().cpu(),
            'targets': output['sentiment_ids'].detach().cpu()
        })
        self.val_step_outputs['topic'].append({
            'predictions': topic_preds.detach().cpu(),
            'targets': output['topic_ids'].detach().cpu()
        })

        self.log('val/loss', output['total_loss'], prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/acc_sentiment', output['acc_sentiment'], prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/acc_topic', output['acc_topic'], prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/mae_length', output['mae_length'], sync_dist=True, batch_size=self.batch_size)
        self.log('val/mae_surprisal', output['mae_surprisal'], sync_dist=True, batch_size=self.batch_size)

        self.log('val/loss_clip', output['loss_clip'], sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_lm', output['loss_lm'], sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_commitment', output['loss_commitment'], sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_sentiment', output['loss_sentiment'], sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_topic', output['loss_topic'], sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_length', output['loss_length'], sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_surprisal', output['loss_surprisal'], sync_dist=True, batch_size=self.batch_size)

        return output['total_loss']

    def on_validation_epoch_end(self):
        """Compute correct accuracy from collected predictions."""
        for task_name in ['sentiment', 'topic']:
            if len(self.val_step_outputs[task_name]) == 0:
                continue

            all_preds = torch.cat([x['predictions'] for x in self.val_step_outputs[task_name]])
            all_targets = torch.cat([x['targets'] for x in self.val_step_outputs[task_name]])

            # Compute correct accuracy from collected predictions
            correct_acc = (all_preds == all_targets).float().mean().item()
            self.log(f'val/acc_{task_name}_correct', correct_acc, prog_bar=True, sync_dist=False)

        self.val_step_outputs = {'sentiment': [], 'topic': []}

    def test_step(self, batch, batch_idx):
        """Test step."""
        with torch.no_grad():
            output = self.shared_forward(batch)

        sentiment_preds = torch.argmax(output['sentiment_logits'], dim=1)
        topic_preds = torch.argmax(output['topic_logits'], dim=1)

        self.test_step_outputs['sentiment'].append({
            'predictions': sentiment_preds.detach().cpu(),
            'targets': output['sentiment_ids'].detach().cpu()
        })
        self.test_step_outputs['topic'].append({
            'predictions': topic_preds.detach().cpu(),
            'targets': output['topic_ids'].detach().cpu()
        })

        self.log('test/loss', output['total_loss'], sync_dist=True, batch_size=self.batch_size)
        self.log('test/acc_sentiment', output['acc_sentiment'], sync_dist=True, batch_size=self.batch_size)
        self.log('test/acc_topic', output['acc_topic'], sync_dist=True, batch_size=self.batch_size)

        return output

    def on_test_epoch_end(self):
        """Compute confusion matrices and correct accuracy for both classification tasks."""
        for task_name in ['sentiment', 'topic']:
            if len(self.test_step_outputs[task_name]) == 0:
                continue

            all_preds = torch.cat([x['predictions'] for x in self.test_step_outputs[task_name]])
            all_targets = torch.cat([x['targets'] for x in self.test_step_outputs[task_name]])

            # Compute confusion matrix
            labels = list(range(len(self.classification_tasks[task_name])))
            cm = confusion_matrix(all_targets.numpy(), all_preds.numpy(), labels=labels)
            setattr(self, f'confusion_matrix_{task_name}', cm)

            # Compute and log correct accuracy from collected predictions
            correct_acc = (all_preds == all_targets).float().mean().item()
            self.log(f'test/acc_{task_name}_correct', correct_acc, sync_dist=False)
            print(f"\n[Correct Accuracy] {task_name}: {correct_acc:.4f} ({correct_acc*100:.2f}%)")

        self.test_step_outputs = {'sentiment': [], 'topic': []}

    def predict_step(self, batch, batch_idx):
        """
        Prediction step that generates all 4 predictions from EEG data. 
        
        Args:
            batch:  Dictionary containing: 
                - 'eeg': EEG tensor of shape i.e. (batch_size, 1280, 128)
                - 'mask': Mask tensor of shape i.e. (batch_size, 1280)
                - 'prompt': List of prompt lists [['task' ... ], ['dataset' ... ], ['subject' ... ]]
                
        Returns:
            Dictionary containing: 
                - 'sentiment_pred': Predicted sentiment class indices
                - 'sentiment_prob': Sentiment class probabilities
                - 'topic_pred':  Predicted topic class indices
                - 'topic_prob': Topic class probabilities
                - 'length_pred': Predicted length values
                - 'surprisal_pred':  Predicted surprisal values
                - 'eeg_emb': EEG embeddings (for optional downstream use)
        """
        eeg = batch['eeg']
        if batch['mask'] is None:
            eeg_mask = torch.ones(eeg.shape[0], eeg.shape[1])
        else:
            eeg_mask = batch['mask']
        prompts = batch['prompt']
        
        # Handle prompts
        if self.use_prompt is False or batch['prompt'] is None:
            batch_size = eeg.shape[0]
            prompts = [['<UNK>'] * batch_size, ['<UNK>'] * batch_size, ['<UNK>'] * batch_size]
        
        # Encode prompts and EEG
        prompt_ids = self.p_embedder.encode(prompts, device=self.device)
        prompt_embed = self.p_embedder(prompt_ids, self.eval_pembed)
        
        eeg_hiddens, _ = self.eeg_encoder(eeg, eeg_mask, prompt_embed)

        # Get EEG embedding through aligner
        Zi, eeg_emb = self.aligner.embed_eeg(eeg_hiddens, None)  # no masks are needed (this is the input mask of embedded text)
        # Zi: (batch_size, l, e)
        # eeg_emb: (batch_size, e)

        # Generate all predictions
        with torch.no_grad():
            # Sentiment classification
            sentiment_logits = self.sentiment_classifier(eeg_emb)
            sentiment_prob = F.softmax(sentiment_logits, dim=-1)
            sentiment_pred = torch.argmax(sentiment_logits, dim=1)
            
            # Topic classification
            topic_logits = self.topic_classifier(eeg_emb)
            topic_prob = F.softmax(topic_logits, dim=-1)
            topic_pred = torch.argmax(topic_logits, dim=1)
            
            # Length regression
            length_pred = self.length_regressor(eeg_emb).squeeze(-1)
            
            # Surprisal regression
            surprisal_pred = self.surprisal_regressor(eeg_emb).squeeze(-1)
        
        return {
            'sentiment_pred': sentiment_pred,
            'sentiment_prob': sentiment_prob,
            'sentiment_label': [self.classification_tasks['sentiment'][idx.item()] for idx in sentiment_pred],
            'topic_pred': topic_pred,
            'topic_prob': topic_prob,
            'topic_label': [self.classification_tasks['topic'][idx.item()] for idx in topic_pred],
            'length_pred': length_pred,
            'surprisal_pred': surprisal_pred,
            'eeg_emb': eeg_emb,
            'Zi': Zi
        }