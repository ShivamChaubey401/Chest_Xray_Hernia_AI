import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Chest X-Ray Hernia Screening",
    page_icon="🩻",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🩻 Generative AI-Enhanced Explainable Chest X-Ray Screening")

st.caption(
    "Hernia-focused research prototype using DenseNet121, "
    "GAN-based augmentation and Grad-CAM"
)

st.divider()


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_trained_model():

    model = models.densenet121(weights=None)

    num_ftrs = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )

    model_path = os.path.join(
        "models",
        "densenet121_mixed_best.pth"
    )

    if not os.path.exists(model_path):
        st.error(
            "Model file not found: "
            "models/densenet121_mixed_best.pth"
        )
        st.stop()

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


model = load_trained_model()

classes = ["Hernia", "Normal"]


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# OOD / NON-X-RAY VALIDATION FILTER
# =========================================================

def validate_chest_xray(img_pil):
    """
    Validates if the input image possesses characteristic grayscale properties 
    and contrast distribution of a genuine chest radiograph.
    Rejects natural color photos and invalid non-medical images.
    """
    img_np = np.array(img_pil)
    
    # 1. Color Variance Check (X-rays are grayscale with R ≈ G ≈ B)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        color_variance = np.mean(
            np.abs(r.astype(float) - g.astype(float)) + 
            np.abs(g.astype(float) - b.astype(float))
        )
        if color_variance > 12.0:
            return False, f"Color photo detected (Color variance: {color_variance:.1f}). X-rays must be grayscale."

    # 2. Illumination & Contrast Distribution Check
    gray = img_pil.convert('L')
    gray_np = np.array(gray)
    std_dev = np.std(gray_np)
    mean_val = np.mean(gray_np)

    if std_dev < 18.0 or mean_val < 15.0 or mean_val > 240.0:
        return False, "Low contrast or invalid illumination distribution. Please upload a standard radiograph."

    return True, "Valid X-Ray"


# =========================================================
# GRAD-CAM
# =========================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):

        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📥 Upload X-Ray")

uploaded_file = st.sidebar.file_uploader(
    "Choose a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

st.sidebar.caption(
    "Supported formats: PNG, JPG, JPEG"
)


# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is None:

    st.info(
        "Upload a chest X-ray image from the sidebar "
        "to start the screening analysis."
    )

    st.markdown("### About this prototype")

    st.write(
        "This research prototype uses a DenseNet121 model "
        "trained with real and selected synthetic Hernia "
        "images. Grad-CAM is used to visualize regions "
        "that contributed to the model prediction."
    )

    st.warning(
        "Research prototype only. This system is not a "
        "medical diagnosis."
    )


else:

    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    try:

        raw_image = Image.open(uploaded_file).convert("RGB")

    except Exception as e:

        st.error(f"Unable to read image: {e}")
        st.stop()


    # -----------------------------------------------------
    # BASIC RESOLUTION CHECK
    # -----------------------------------------------------

    width, height = raw_image.size

    if width < 100 or height < 100:

        st.error(
            "Image resolution is too small. "
            "Please upload a clear X-ray image."
        )

        st.stop()


    # -----------------------------------------------------
    # OOD / NON-MEDICAL IMAGE VALIDATION
    # -----------------------------------------------------

    is_valid, validation_reason = validate_chest_xray(raw_image)

    if not is_valid:

        st.error("❌ **INVALID / NON-MEDICAL IMAGE DETECTED**")

        st.warning(f"**Reason:** {validation_reason}")

        st.image(
            raw_image,
            caption="Uploaded Image (Rejected)",
            width=350
        )

        st.info(
            "ℹ️ The system rejected this input prior to inference "
            "to prevent false positive classifications. Please upload a genuine grayscale Chest X-Ray."
        )

        st.stop()


    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    input_tensor = transform(raw_image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)


    # =====================================================
    # PREDICTION & UNCERTAINTY HANDLING
    # =====================================================

    with torch.no_grad():

        output = model(input_tensor)

        probs = F.softmax(output, dim=1)[0]

        conf, predicted = torch.max(probs, 0)

    pred_class = classes[predicted.item()]

    confidence = conf.item() * 100

    hernia_prob = probs[0].item() * 100
    normal_prob = probs[1].item() * 100

    # Uncertainty Threshold check for non-Hernia diseases/borderline cases
    is_borderline = confidence < 78.0


    # =====================================================
    # RESULT SECTION
    # =====================================================

    st.subheader("Screening Result")

    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:

        st.image(
            raw_image,
            caption="Uploaded Chest X-Ray",
            use_container_width=True
        )


    with result_col2:

        if is_borderline:
            verdict_text = "AMBIGUOUS / BORDERLINE CASE"
            st.warning(f"### ⚠️ {verdict_text}")
            st.info(
                f"Model prediction confidence is moderate ({confidence:.2f}%). "
                "Other thoracic pathologies (e.g. Pneumonia, Effusion) or non-standard X-rays "
                "can trigger moderate predictions in binary models. Clinical review required."
            )
        elif pred_class == "Hernia":
            verdict_text = "HERNIA DETECTED"
            st.error(f"### {verdict_text}")

        else:
            verdict_text = "NORMAL"
            st.success(f"### {verdict_text}")

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.caption(
            "Confidence represents the model's predicted probability, "
            "not medical certainty."
        )

        # -------------------------------------------------
        # REPORT DOWNLOAD BUTTON
        # -------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_content = f"""==================================================
CHEST X-RAY HERNIA SCREENING REPORT
==================================================
Generated On       : {timestamp}
Uploaded File Name : {uploaded_file.name}
Image Resolution   : {width} x {height}

--------------------------------------------------
SCREENING ANALYSIS
--------------------------------------------------
Verdict            : {verdict_text}
Primary Prediction : {pred_class}
Model Confidence   : {confidence:.2f}%

PROBABILITY DISTRIBUTION:
- Hernia Probability : {hernia_prob:.2f}%
- Normal Probability : {normal_prob:.2f}%

--------------------------------------------------
TECHNICAL & CLINICAL DETAILS
--------------------------------------------------
Model Architecture : DenseNet121 (Transfer Learning)
Data Augmentation  : GAN-Augmented Synthetic Data Integrated
Status              : {"Borderline/Ambiguous (Requires Expert Review)" if is_borderline else "High Confidence Inference"}

--------------------------------------------------
DISCLAIMER
--------------------------------------------------
This report is generated by an AI research prototype 
for academic and research evaluation only. It does not 
replace professional clinical diagnosis or radiologist 
consultation.
==================================================
"""
        st.write("---")
        st.download_button(
            label="📄 Download Screening Report (.txt)",
            data=report_content,
            file_name=f"Hernia_Screening_Report_{uploaded_file.name.split('.')[0]}.txt",
            mime="text/plain"
        )


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.divider()

    st.subheader("Prediction Probabilities")

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:

        st.write("**Hernia**")

        st.progress(
            int(round(hernia_prob))
        )

        st.caption(
            f"{hernia_prob:.2f}%"
        )


    with prob_col2:

        st.write("**Normal**")

        st.progress(
            int(round(normal_prob))
        )

        st.caption(
            f"{normal_prob:.2f}%"
        )


    # =====================================================
    # GRAD-CAM (PROMINENTLY FOR HERNIA ONLY)
    # =====================================================

    st.divider()

    st.subheader("Explainable AI — Grad-CAM")

    if pred_class == "Hernia" and not is_borderline:
        st.write(
            "Grad-CAM highlights specific anatomical regions "
            "that contributed most strongly to the **Hernia** prediction."
        )

        target_layer = (
            model.features
            .denseblock4
            .denselayer16
            .conv2
        )

        grad_cam = GradCAM(
            model,
            target_layer
        )

        # Forward + backward pass for Hernia class
        model.zero_grad()
        output = model(input_tensor)
        target_score = output[0, predicted.item()]
        target_score.backward()

        gradients = grad_cam.gradients.cpu().numpy()[0]
        activations = grad_cam.activations.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        cam = np.maximum(cam, 0)

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_image = Image.fromarray(
            np.uint8(cam * 255)
        ).resize((224, 224))

        cam = np.array(cam_image) / 255.0

        # Display Heatmap
        cam_col1, cam_col2 = st.columns([1, 1])

        with cam_col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(raw_image.resize((224, 224)))
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.axis("off")
            ax.set_title("Grad-CAM Heatmap (Hernia Focus)")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        with cam_col2:
            st.markdown("#### Interpretation")
            st.write(
                "🔴 **Warm/Red regions** highlight the specific areas "
                "(e.g., diaphragm / lower thoracic cavity) where the model detected "
                "pathological features corresponding to Hernia."
            )
            st.info(
                "Grad-CAM provides visual explainability to assist radiologist review."
            )

        grad_cam.remove_hooks()

    else:
        st.success("✅ **No Pathological Anomaly Detected**")
        st.info(
            "Grad-CAM visual explanation is automatically enabled when pathological "
            "features (Hernia) are detected. For Normal or Borderline X-Rays, "
            "heatmap generation is skipped to maintain visual clarity."
        )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.divider()

    st.caption(
        "Research prototype for academic purposes only. "
        "Not intended for medical diagnosis or clinical decision-making."
    )