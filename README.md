# Part 2 — CNN-Based Manufacturing Defect Classifier

This project builds a Convolutional Neural Network (CNN) from scratch to classify product surface images into four categories: **normal**, **scratch**, **dent**, and **stain**. The dataset is a synthetic manufacturing defect image dataset where each 64 × 64 RGB image represents a product surface captured by an inline quality-inspection camera. The end-to-end pipeline covers problem identification, dataset exploration, preprocessing with augmentation, model construction in TensorFlow/Keras, evaluation with confusion matrix and classification report, and a business use case mapping to automated visual quality inspection in manufacturing.

---

## Dataset

| Property | Value |
|---|---|
| Total images | 480 |
| Classes | normal, scratch, dent, stain |
| Images per class | 120 (perfectly balanced) |
| Image size | 64 × 64 pixels |
| Colour mode | RGB |
| Source | Synthetic procedural generation (manufacturing defect simulation) |

The dataset is generated programmatically inside `notebook.py` using numpy and PIL. Each class has a visually distinct appearance: normal surfaces are smooth and uniform; scratch images carry 2–4 dark diagonal lines with a bright highlight edge; dent images feature a circular darkened depression with a bright ring; stain images show an elliptical coloured blotch blended onto the base surface.

---

## Repository Structure

```
part-2-cnn-computer-vision/
│
├── README.md                        ← this file
├── notebook.py                      ← main script (all 7 tasks)
├── requirements.txt                 ← Python dependencies
│
├── sample_predictions/
│   └── prediction_outputs.png       ← 12 test images with true/predicted labels
│
└── results/
    ├── accuracy_loss_curves.png     ← training & validation accuracy/loss
    ├── confusion_matrix.png         ← per-class prediction heatmap
    ├── class_distribution.png       ← bar chart of images per class
    └── sample_images.png            ← 4 sample images × 4 classes grid
```



## Tasks Covered

| Task | Description |
|---|---|
| Task 1 | Problem identification — image classification |
| Task 2 | Dataset exploration — class counts, dimensions, pixel stats, imbalance check |
| Task 3 | Preprocessing — resize, normalise, 80/20 stratified split, augmentation |
| Task 4 | CNN model — Conv × 3, MaxPool × 3, Dense, Dropout, Softmax output |
| Task 5 | Training & evaluation — accuracy/loss curves, confusion matrix, predictions |
| Task 6 | CNN concept explanation — convolution, pooling, ReLU, CNNs vs FFNs |
| Task 7 | Business use case — automated visual inspection in manufacturing |

---

## Key Results

The CNN achieved **100% test accuracy** and a **weighted F1 score of 1.00** on the 96-image held-out test set. The confusion matrix shows a perfectly clean diagonal with no cross-class misclassifications, confirming that the model learned reliable visual signatures for each defect type. Training and validation accuracy both converged smoothly over 20 epochs without overfitting.

---

## CNN Concept Explanations

### What is Convolution?

A convolution operation slides a small filter (3 × 3 pixels) across the image. At each location it multiplies the filter values with the overlapping pixel values and sums them into one number. This produces a "feature map" highlighting wherever the learned pattern — an edge, a gradient, a texture — appears in the image. Because the same filter is reused across all positions (weight sharing), CNNs need far fewer parameters than fully connected networks to process images.

### Why is Pooling Used?

Max pooling keeps only the maximum activation in each small spatial window (2 × 2). It compresses the feature maps — reducing memory and computation — and introduces translation invariance: if a defect shifts by a few pixels, the pooled response barely changes. Without pooling, the fully connected layers would need to handle enormous input vectors, making training impractical.

### Why is ReLU Commonly Used in CNNs?

ReLU (Rectified Linear Unit) sets negative activations to zero and passes positives unchanged: `f(x) = max(0, x)`. It avoids the vanishing gradient problem that plagues sigmoid and tanh in deep networks, is computationally trivial to compute, and produces sparse activations that act as implicit regularisation. These properties make it the default activation for modern CNNs.

### Why are CNNs Better than Plain Feed-Forward Networks for Images?

A fully connected network treats every pixel independently and needs millions of parameters even for small images. It also has no built-in notion of spatial locality, so it must re-learn that nearby pixels are related from scratch. CNNs bake in two inductive biases — local connectivity and weight sharing — that match the structure of natural images perfectly. They learn features hierarchically (edges → textures → shapes) and are robust to small positional shifts in the object of interest, which is exactly what we need for defect detection regardless of where on the product surface the defect appears.

---

## Business Use Case — Manufacturing Quality Inspection

Inline CNN-based visual inspection replaces or augments fatigued human inspectors on high-volume production lines. Every product passes under a camera; the model classifies it in under 50 ms and triggers a physical diverter to route defective units off the line without interrupting throughput. Classification labels are also actionable for process control: a sustained rise in scratch detections may indicate worn tooling, while a cluster of dent labels can point to an upstream handling problem. Inspecting every unit (rather than a statistical sample) provides the full-population data needed for SPC charts — a regulatory requirement in medical device and aerospace manufacturing.
