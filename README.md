# BirdCLEF23-Supervised-Contrastive-Loss-Training-explanation
it's about theoretical explanation for BirdCLEF23 Supervised Contrastive Loss Training

# BirdCLEF23 Training with Supervised Contrastive Learning

This repository implements a supervised contrastive learning approach for the BirdCLEF 2023 competition, focusing on bird species classification from audio recordings. Below is a detailed breakdown of the training pipeline.

---

## 1. Data Preprocessing

### Audio to Mel-spectrograms
- Convert raw audio files into **Mel-spectrograms** using the following steps:
  - **Frame splitting & windowing**: Apply Short-Time Fourier Transform (STFT) to extract time-frequency representations.
  - **Mel-scale compression**: Reduce frequency dimensions using a Mel filter bank.
  - **Log scaling**: Enhance low-amplitude components with log-Mel transformation.
- Example parameters: `n_mels=128`, `hop_length=512`, `n_fft=2048`.

### Data Augmentation
- Apply random augmentations to spectrograms for robustness:
  - **Time masking**: Randomly mask segments along the time axis.
  - **Frequency masking**: Mask frequency bands.
  - **Noise injection**: Add background noise or Gaussian noise.
  - **Time stretching**: Perturb audio speed slightly.

### Dataset Splitting
- Stratified split of training and validation sets to preserve class distributions (critical for imbalanced bird species data).

---

## 2. Model Architecture

### Backbone Network
- Use a pre-trained CNN (e.g., `EfficientNet-B4`, `ResNet-50`) to extract high-level features from Mel-spectrograms.
- Freeze early layers and fine-tune deeper layers during training.

### Projection Head
- Map backbone features to a **contrastive embedding space** using an MLP:

```python
ProjectionHead(nn.Linear(backbone_feature_dim, 512),nn.ReLU(),nn.Linear(512, 128) #Embedding dimension = 128)
```

### Classification Head (Optional)
- Add a linear layer + Softmax for final species prediction:

```python
Classification(nn.Linear(128,num_classes))
```


---

## 3. Supervised Contrastive Loss (SCL)

### Key Idea
- Leverage label information to pull embeddings of the same class closer and push different classes apart.

### Loss Formula
\[
\mathcal{L}_{\text{SCL}} = \sum_{i=1}^N \frac{-1}{N_{y_i}-1} \sum_{\substack{j=1 \\ j \neq i}}^{N} \mathbb{I}_{y_i=y_j} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum_{k=1}^N \mathbb{I}_{k \neq i} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}
\]
- \( \mathbf{z}_i, \mathbf{z}_j \): Normalized embeddings of samples \(i\) and \(j\).
- \( \tau \): Temperature hyperparameter (default: 0.07).
- \( N_{y_i} \): Number of samples in the batch with class \(y_i\).

### Implementation
- Compute embeddings for a batch of size \(N\), generate positive/negative pairs using labels.

---

## 4. Training Pipeline

### Framework
- Built with **PyTorch Lightning** for distributed training and logging.

### Optimization
- **Optimizer**: AdamW with weight decay (`lr=3e-4`, `weight_decay=1e-5`).
- **Learning Rate Scheduler**: Cosine annealing with warmup.
- **Mixed Precision**: Enabled via `torch.cuda.amp`.

### Regularization
- **Label smoothing**: Mitigate overconfidence in predictions.
- **Gradient clipping**: Prevent exploding gradients.

---

## 5. Data Augmentation Strategies

### Spectrogram Augmentations
- **SpecAugment**:
- Time warping, frequency masking (max 10% bands), time masking (max 20% length).
- **Random gain**: ±6 dB amplitude variation.

### Environmental Robustness
- Simulate field recording conditions by mixing background noise (e.g., wind, rain).

---

## 6. Evaluation & Inference

### Metrics
- **Primary metric**: Macro F1-score (official competition metric).
- **Secondary metrics**: Accuracy, precision/recall per class.

### Test-Time Augmentation (TTA)
- Average predictions over multiple augmented views of the same audio.

### Model Ensemble
- Combine predictions from models with different backbones (e.g., EfficientNet + ResNet).

---

## 7. Key Advantages

- **Contrastive Learning**:
- Improved generalization on rare species (long-tail distribution).
- Robustness to acoustic variations (e.g., background noise).
- **Transfer Learning**:
- Leverage pre-trained CNNs for efficient feature extraction.

---

## Code Implementation Notes

### Data Loader
- Use `torchaudio` or `librosa` for Mel-spectrogram extraction.
- Example preprocessing pipeline:


```python
transform = Compose([LoadAudio(), ToMelSpectrogram(n_mels=128), ApplyAugmentations(), Normalize()])
```

### Loss Implementation
- Custom Pytorch module for SCL:

```python
class SupervisedContrastiveLoss(nn.Module):
    def init(self, tau = 0.07):
      super().init()
      self.tau = tau
    def forward(self, embeddings,labels):
    # Compute pairwise cosine similarity
    # Generate mask for positive pairs
    # Calculate loss
    return loss
```


### Hyperparameters
- Temperature (\( \tau \)): Critical for convergence (tune between 0.05–0.2).
- Batch size: Use large batches (≥ 512) for effective contrastive learning.

---

For full implementation details, refer to the [Kaggle Notebook](https://www.kaggle.com/code/vijayravichander/birdclef23-supervised-contrastive-loss-training).  
**Upvote the notebook if you find it helpful!** 👍

