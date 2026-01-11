import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional


class Stage2ReconstructionModel(nn.Module):
    """
    Stage 2 model for text reconstruction from EEG features.
    Uses Flan-T5 with embedding injection for multi-modal fusion.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        freeze_strategy: str = "lora",
        lora_rank: int = 8,
        label_embed_init: Optional[Dict] = None,
        device: str = "cuda:0"
    ):
        """
        Args:
            model_name: Hugging Face model name
            freeze_strategy: 'lora' or 'full_freeze_llm'
            lora_rank: LoRA rank
            label_embed_init: Optional pre-trained label embeddings
            device: Device to use
        """
        super().__init__()
        self.device = torch.device(device)
        self.freeze_strategy = freeze_strategy

        # Determine dtype based on hardware support
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.model_dtype = torch.bfloat16
        else:
            self.model_dtype = torch.float32

        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.model_dtype
        )

        # Add special tokens and resize BEFORE applying LoRA
        special_tokens = ['<SENT_VAL>', '<TOPIC_VAL>', '<EEG_GLOBAL>', '<EEG_SEQ>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Store special token IDs
        self.sent_val_id = self.tokenizer.convert_tokens_to_ids('<SENT_VAL>')
        self.topic_val_id = self.tokenizer.convert_tokens_to_ids('<TOPIC_VAL>')
        self.eeg_global_id = self.tokenizer.convert_tokens_to_ids('<EEG_GLOBAL>')
        self.eeg_seq_id = self.tokenizer.convert_tokens_to_ids('<EEG_SEQ>')

        # Label embeddings (2 classes each, 1024 dim)
        self.label_embed_task1 = nn.Embedding(2, 1024)
        self.label_embed_task2 = nn.Embedding(2, 1024)
        nn.init.normal_(self.label_embed_task1.weight, std=0.02)
        nn.init.normal_(self.label_embed_task2.weight, std=0.02)

        # Load pre-trained label embeddings if provided
        if label_embed_init is not None:
            self.load_label_embeddings(label_embed_init)

        # Apply freeze strategy AFTER resize_token_embeddings
        if freeze_strategy == "lora":
            # Freeze base model, add LoRA
            for param in self.model.parameters():
                param.requires_grad = False

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.gradient_checkpointing_enable()

            # Unfreeze embedding layers for new tokens
            for name, param in self.model.named_parameters():
                if "embed_tokens" in name or "shared" in name:
                    param.requires_grad = True

        elif freeze_strategy == "full_freeze_llm":
            # Freeze everything in the model
            for param in self.model.parameters():
                param.requires_grad = False

        # Move to device
        self.to(self.device)

    def build_prompt(self) -> str:
        """Build prompt template with special tokens."""
        eeg_seq_tokens = " ".join(["<EEG_SEQ>"] * 96)
        prompt = (
            "System: Based on the following EEG signals, reconstruct the text. "
            "Attributes: [ <SENT_VAL>, <TOPIC_VAL> ] "
            "Global Context: <EEG_GLOBAL> "
            f"Sequence: {eeg_seq_tokens} "
            "Target:"
        )
        return prompt

    def forward(
        self,
        label_task1: torch.Tensor,
        label_task2: torch.Tensor,
        length: torch.Tensor,
        surprisal: torch.Tensor,
        ei: torch.Tensor,
        Zi: torch.Tensor,
        target_text: list
    ) -> torch.Tensor:
        """
        Forward pass with vectorized embedding injection.

        Args:
            label_task1: (batch_size,) sentiment labels
            label_task2: (batch_size,) topic labels
            length: (batch_size,) length predictions
            surprisal: (batch_size,) surprisal prediction
            ei: (batch_size, 1024) global EEG vectors
            Zi: (batch_size, 96, 1024) EEG sequences
            target_text: List of target text strings

        Returns:
            loss: Cross-entropy loss
        """
        batch_size = label_task1.shape[0]

        # Build prompt
        prompt = self.build_prompt()
        prompts = [prompt] * batch_size

        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        input_ids = prompt_encoding.input_ids

        # Get initial embeddings
        embeds = self.model.shared(input_ids)

        # Cast inputs to model dtype
        sent_embs = self.label_embed_task1(label_task1).to(embeds.dtype)
        topic_embs = self.label_embed_task2(label_task2).to(embeds.dtype)
        ei = ei.to(embeds.dtype)
        Zi = Zi.to(embeds.dtype)

        # Find token positions using masks
        sent_mask = (input_ids == self.sent_val_id)
        topic_mask = (input_ids == self.topic_val_id)
        global_mask = (input_ids == self.eeg_global_id)
        seq_mask = (input_ids == self.eeg_seq_id)

        # Direct assignment for single-token attributes
        batch_indices = torch.arange(batch_size, device=self.device)
        sent_positions = sent_mask.long().argmax(dim=1)
        topic_positions = topic_mask.long().argmax(dim=1)
        global_positions = global_mask.long().argmax(dim=1)

        embeds[batch_indices, sent_positions] = sent_embs
        embeds[batch_indices, topic_positions] = topic_embs
        embeds[batch_indices, global_positions] = ei

        # Flatten Zi for sequence replacement
        Zi_flat = Zi.view(-1, 1024)
        assert seq_mask.sum() == Zi.numel() // 1024, f"Mask count {seq_mask.sum()} != Zi elements {Zi.numel() // 1024}"
        embeds[seq_mask] = Zi_flat

        # Tokenize target text
        target_encoding = self.tokenizer(
            target_text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Forward pass
        outputs = self.model(
            inputs_embeds=embeds,
            labels=labels
        )

        return outputs.loss

    def generate(
        self,
        label_task1: torch.Tensor,
        label_task2: torch.Tensor,
        length: torch.Tensor,
        surprisal: torch.Tensor,
        ei: torch.Tensor,
        Zi: torch.Tensor,
        max_length: int = 50
    ) -> list:
        """
        Generate text from EEG features.

        Args:
            label_task1: (batch_size,) sentiment labels
            label_task2: (batch_size,) topic labels
            length: (batch_size,) length predictions
            surprisal: (batch_size,) surprisal prediction
            ei: (batch_size, 1024) global EEG vectors
            Zi: (batch_size, 96, 1024) EEG sequences
            max_length: Maximum generation length

        Returns:
            List of generated text strings
        """
        batch_size = label_task1.shape[0]

        # Build prompt
        prompt = self.build_prompt()
        prompts = [prompt] * batch_size

        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        input_ids = prompt_encoding.input_ids

        # Get initial embeddings
        embeds = self.model.shared(input_ids)

        # Cast inputs to model dtype
        sent_embs = self.label_embed_task1(label_task1).to(embeds.dtype)
        topic_embs = self.label_embed_task2(label_task2).to(embeds.dtype)
        ei = ei.to(embeds.dtype)
        Zi = Zi.to(embeds.dtype)

        # Find token positions
        sent_mask = (input_ids == self.sent_val_id)
        topic_mask = (input_ids == self.topic_val_id)
        global_mask = (input_ids == self.eeg_global_id)
        seq_mask = (input_ids == self.eeg_seq_id)

        # Direct assignment
        batch_indices = torch.arange(batch_size, device=self.device)
        embeds[batch_indices, sent_mask.long().argmax(dim=1)] = sent_embs
        embeds[batch_indices, topic_mask.long().argmax(dim=1)] = topic_embs
        embeds[batch_indices, global_mask.long().argmax(dim=1)] = ei

        Zi_flat = Zi.view(-1, 1024)
        embeds[seq_mask] = Zi_flat

        # Generate
        outputs = self.model.generate(
            inputs_embeds=embeds,
            max_length=max_length,
            num_beams=1,
            do_sample=False
        )

        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts

    def load_label_embeddings(self, state_dict: Dict):
        """Load pre-trained label embeddings."""
        if 'label_embed_task1' in state_dict:
            self.label_embed_task1.load_state_dict(state_dict['label_embed_task1'])
        if 'label_embed_task2' in state_dict:
            self.label_embed_task2.load_state_dict(state_dict['label_embed_task2'])
