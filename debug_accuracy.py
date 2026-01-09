"""
Debug script to understand the accuracy discrepancy.
"""
import torch
from model.glim_parallel import GLIM_PARALLEL
from data.datamodule import GLIMDataModule
from torchmetrics.functional.classification import multiclass_accuracy
from sklearn.metrics import confusion_matrix
import numpy as np

# Load checkpoint
ckpt_path = "/mnt/afs/250010218/glimCLS/logs/glim_parallel_20260108_165301/checkpoints/model-epoch43-acc_sentiment0.5704.ckpt"
model = GLIM_PARALLEL.load_from_checkpoint(ckpt_path)
model.eval()
model = model.cuda()

# Load data
datamodule = GLIMDataModule(
    data_path="data/ZUCO1-2_FOR_GLIMCLS/zuco_merged_with_variants.df",
    eval_noise_input=False,
    bsz_train=72,
    bsz_val=24,
    bsz_test=24,
    num_workers=4,
    use_weighted_sampler=False,
    classification_label_keys=['sentiment label', 'topic_label'],
    regression_label_keys=['length', 'surprisal'],
    use_zuco1_only=True
)
datamodule.setup('test')
test_loader = datamodule.test_dataloader()

print(f"Model batch_size attribute: {model.batch_size}")
print(f"Test loader batch size: {test_loader.batch_size}")
print(f"Number of test batches: {len(test_loader)}")

# Collect predictions and compute metrics
all_sentiment_preds = []
all_sentiment_targets = []
all_topic_preds = []
all_topic_targets = []
batch_sentiment_accs = []
batch_topic_accs = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        # Move batch to GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        # Forward pass
        output = model.shared_forward(batch)

        # Get predictions
        sentiment_preds = torch.argmax(output['sentiment_logits'], dim=1)
        topic_preds = torch.argmax(output['topic_logits'], dim=1)

        # Store for confusion matrix
        all_sentiment_preds.append(sentiment_preds.cpu())
        all_sentiment_targets.append(output['sentiment_ids'].cpu())
        all_topic_preds.append(topic_preds.cpu())
        all_topic_targets.append(output['topic_ids'].cpu())

        # Store per-batch accuracy (as computed in the model)
        batch_sentiment_accs.append(output['acc_sentiment'].item())
        batch_topic_accs.append(output['acc_topic'].item())

        if batch_idx < 3:
            print(f"\nBatch {batch_idx}:")
            print(f"  Batch size: {len(sentiment_preds)}")
            print(f"  Sentiment acc (from model): {output['acc_sentiment']:.4f}")
            print(f"  Topic acc (from model): {output['acc_topic']:.4f}")
            print(f"  Sentiment targets: {output['sentiment_ids'][:5].tolist()}")
            print(f"  Sentiment preds: {sentiment_preds[:5].tolist()}")

# Compute overall metrics
all_sentiment_preds = torch.cat(all_sentiment_preds)
all_sentiment_targets = torch.cat(all_sentiment_targets)
all_topic_preds = torch.cat(all_topic_preds)
all_topic_targets = torch.cat(all_topic_targets)

# Method 1: Average of per-batch accuracies
avg_sentiment_acc = np.mean(batch_sentiment_accs)
avg_topic_acc = np.mean(batch_topic_accs)

# Method 2: Confusion matrix accuracy
cm_sentiment = confusion_matrix(all_sentiment_targets.numpy(), all_sentiment_preds.numpy(), labels=[0, 1])
cm_topic = confusion_matrix(all_topic_targets.numpy(), all_topic_preds.numpy(), labels=[0, 1])
cm_sentiment_acc = (cm_sentiment[0, 0] + cm_sentiment[1, 1]) / cm_sentiment.sum()
cm_topic_acc = (cm_topic[0, 0] + cm_topic[1, 1]) / cm_topic.sum()

# Method 3: Overall multiclass_accuracy
overall_sentiment_acc = multiclass_accuracy(
    torch.cat([torch.zeros(len(all_sentiment_preds), 2).scatter_(1, all_sentiment_preds.unsqueeze(1), 1)]),
    all_sentiment_targets,
    num_classes=2,
    ignore_index=-1
)

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"\nSentiment Accuracy:")
print(f"  Average of per-batch accs: {avg_sentiment_acc:.4f} ({avg_sentiment_acc*100:.2f}%)")
print(f"  Confusion matrix acc: {cm_sentiment_acc:.4f} ({cm_sentiment_acc*100:.2f}%)")
print(f"\nTopic Accuracy:")
print(f"  Average of per-batch accs: {avg_topic_acc:.4f} ({avg_topic_acc*100:.2f}%)")
print(f"  Confusion matrix acc: {cm_topic_acc:.4f} ({cm_topic_acc*100:.2f}%)")

print(f"\nConfusion Matrix - Sentiment:")
print(cm_sentiment)
print(f"\nConfusion Matrix - Topic:")
print(cm_topic)

print(f"\nTotal samples: {len(all_sentiment_targets)}")
print(f"Samples with -1 sentiment label: {(all_sentiment_targets == -1).sum().item()}")
print(f"Samples with -1 topic label: {(all_topic_targets == -1).sum().item()}")
