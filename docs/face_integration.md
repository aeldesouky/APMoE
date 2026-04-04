# Face Age Prediction Model

## Integration Report

---

## 1. Introduction

This report describes the integration of a deep learning model used for predicting human age from facial images. The model is implemented using TensorFlow/Keras and outputs a numerical age value based on an input image.

---

## 2. Model Overview

* **Task:** Age prediction from facial images
* **Model Type:** Deep Learning Regression Model
* **Output:** Single numeric value (predicted age)
* **Model Format:** `.keras`

---

## 3. Input Requirements

### Input Shape

```
(1, 200, 200, 3)
```

### Details

* Image must be resized to **200 × 200**
* Must be **RGB (3 channels)**
* Pixel values normalized to **[0, 1]**
* Must include **batch dimension**

---

## 4. Preprocessing Pipeline

```python
img = img.resize((200, 200))
img_array = np.array(img)

if img_array.ndim == 2:
    img_array = np.stack((img_array,)*3, axis=-1)
elif img_array.shape[-1] == 4:
    img_array = img_array[:, :, :3]

img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)
```

---

## 5. Handling Edge Cases

* Convert **grayscale → RGB**
* Remove **alpha channel (RGBA → RGB)**

---

## 6. Model Inference

```python
prediction = model.predict(img_array)
```

---

## 7. Output Handling

```python
predicted_age = int(round(prediction[0][0]))
```

* Output is **float**
* Must be **rounded to integer**

### Confidence in APMoE

The face model is a **regressor**: it returns a single age value, not class probabilities. In the APMoE framework, `FaceAgeExpert` sets per-expert `confidence` to **`-1.0`**, which means **confidence is not reported** (not a score in `[0, 1]`). Aggregators treat `-1` as “no self-reported confidence” when combining experts; the final prediction `confidence` stays in `[0.0, 1.0]`. Use explicit `aggregation.weights` in config if you need to control how the face expert blends with others.

---

## 8. System Integration

### Pipeline

```
Image Input → Preprocessing → Model → Age Output
```

---

### Integration with Keystrokes Model

```
User Input
   ├── Image → Age Model
   └── Keystrokes → Behavior Model

Final Decision Layer → Output
```

---

## 9. Implementation Function

```python
def predict_age(image):
    img = image.resize((200, 200))
    img_array = np.array(img)

    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return int(round(prediction[0][0]))
```

---

## 10. Environment Setup

```bash
pip install tensorflow numpy pillow matplotlib
```

---

## 11. Common Issues & Fixes

### Model Not Loading

* Check file path
* Ensure `.keras` file exists

---

### Incorrect Predictions

* Missing normalization
* Wrong image size

---

### Input Errors

* Handle grayscale / RGBA correctly

---

## 12. Performance Notes

* Load model **once at startup**
* Avoid reloading per request

---

## 13. Conclusion

This model can be integrated as a standalone component within a larger system. Proper preprocessing and output handling are critical to ensure accurate predictions. When combined with other models such as keystroke analysis, it enhances system robustness and decision-making.

---
