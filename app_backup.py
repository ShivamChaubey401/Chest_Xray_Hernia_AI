import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Generative AI-Enhanced Chest X-Ray Screening",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Generative AI-Enhanced Explainable Chest X-Ray Screening")

st.markdown(
    "##### **Focus Area:** Rare Hernia Class Screening "
    "via DenseNet121 & GAN-Augmented Training"
)

st.write("---")


# =========================================================
# 2. DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 3. MODEL
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

    checkpoint_path = os.path.join(
        "models",
        "densenet121_mixed_best.pth"
    )

    if not os.path.exists(checkpoint_path):
        st.error(
            f"❌ Model file not found:\n{checkpoint_path}"
        )
        st.stop()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


model = load_trained_model()

classes = ["Hernia", "Normal"]


# =========================================================
# 4. IMAGE PREPROCESSING
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
# 5. GRAD-CAM
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
# 6. FILE UPLOAD
# =========================================================

st.sidebar.header("📥 Upload X-Ray Input")

uploaded_file = st.sidebar.file_uploader(
    "Choose a Chest X-Ray image",
    type=["png", "jpg", "jpeg"]
)


# =========================================================
# 7. NO IMAGE
# =========================================================

if uploaded_file is None:

    st.info(
        "👈 Please upload a Chest X-Ray image from the sidebar "
        "to run the screening analysis."
    )

    st.markdown("""
    ### 🔬 Research Pipeline

    **Upload X-ray → DenseNet121 → Hernia/Normal → Confidence → Grad-CAM**

    This application uses a DenseNet121 model trained with real
    and selected GAN-generated Hernia images.
    """)


# =========================================================
# 8. IMAGE PROCESSING
# =========================================================

else:

    try:

        raw_image = Image.open(uploaded_file).convert("RGB")

    except Exception as e:

        st.error(f"❌ Unable to read image: {e}")
        st.stop()


    # -----------------------------------------------------
    # Basic image validation
    # -----------------------------------------------------

    image_array = np.array(raw_image)

    # Check extremely small images
    width, height = raw_image.size

    if width < 100 or height < 100:

        st.error(
            "⚠️ Image resolution is too small. "
            "Please upload a clear X-ray image."
        )

        st.stop()


    # -----------------------------------------------------
    # Display uploaded image
    # -----------------------------------------------------

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("🖼️ Uploaded Chest X-Ray")

        st.image(
            raw_image,
            caption="Uploaded Image",
            use_container_width=True
        )


    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    input_tensor = transform(raw_image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)


    # =====================================================
    # 9. PREDICTION
    # =====================================================

    with torch.enable_grad():

        model.zero_grad()

        output = model(input_tensor)

        probs = F.softmax(output, dim=1)[0]

        conf, predicted = torch.max(probs, 0)

        pred_class = classes[predicted.item()]

        confidence = conf.item() * 100


    # =====================================================
    # 10. RESULT
    # =====================================================

    with col2:

        st.subheader("🎯 Screening Result")

        if pred_class == "Hernia":

            st.error(
                f"### {pred_class.upper()} DETECTED"
            )

        else:

            st.success(
                f"### {pred_class.upper()}"
            )


        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        st.write("---")

        st.subheader("📊 Probability Breakdown")


        hernia_prob = probs[0].item() * 100
        normal_prob = probs[1].item() * 100


        st.progress(
            int(round(hernia_prob)),
            text=f"Hernia: {hernia_prob:.2f}%"
        )

        st.progress(
            int(round(normal_prob)),
            text=f"Normal: {normal_prob:.2f}%"
        )


    # =====================================================
    # 11. GRAD-CAM
    # =====================================================

    st.write("---")

    st.subheader("🔍 Explainable AI — Grad-CAM")


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


    # Forward pass again for Grad-CAM

    model.zero_grad()

    output = model(input_tensor)

    target_score = output[0, predicted.item()]

    target_score.backward()


    # -----------------------------------------------------
    # Get gradients & activations
    # -----------------------------------------------------

    gradients = grad_cam.gradients.cpu().numpy()[0]

    activations = grad_cam.activations.cpu().numpy()[0]


    # -----------------------------------------------------
    # Global average pooling of gradients
    # -----------------------------------------------------

    weights = np.mean(
        gradients,
        axis=(1, 2)
    )


    cam = np.zeros(
        activations.shape[1:],
        dtype=np.float32
    )


    for i, weight in enumerate(weights):

        cam += weight * activations[i]


    # -----------------------------------------------------
    # ReLU
    # -----------------------------------------------------

    cam = np.maximum(cam, 0)


    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    if cam.max() > 0:

        cam = cam / cam.max()


    # -----------------------------------------------------
    # Resize
    # -----------------------------------------------------

    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(
        (224, 224)
    )


    cam = np.array(cam_image) / 255.0


    # =====================================================
    # 12. DISPLAY GRAD-CAM
    # =====================================================

    cam_col1, cam_col2 = st.columns([1, 1])


    with cam_col1:

        fig, ax = plt.subplots(
            figsize=(6, 6)
        )

        ax.imshow(
            raw_image.resize((224, 224)),
            cmap="gray"
        )

        ax.imshow(
            cam,
            cmap="jet",
            alpha=0.45
        )

        ax.axis("off")

        ax.set_title(
            "Grad-CAM Heatmap Overlay"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


    with cam_col2:

        st.markdown("### 🧠 What does Grad-CAM show?")

        st.write(
            """
            Grad-CAM highlights image regions that contributed
            more strongly to the model's prediction.

            **Warm/red regions** indicate areas with relatively
            higher contribution to the selected prediction.

            This visualization helps inspect what regions the
            DenseNet121 model considered important.
            """
        )

        st.warning(
            "⚠️ Grad-CAM is an explainability visualization. "
            "It does not prove that a highlighted region represents "
            "a clinically confirmed pathology."
        )


    # =====================================================
    # 13. RESEARCH DISCLAIMER
    # =====================================================

    st.write("---")

    st.warning(
        "⚠️ Research Prototype: This system is intended for "
        "academic/research purposes only and must not be used "
        "as a medical diagnosis or substitute for professional "
        "radiological evaluation."
    )


    # Remove Grad-CAM hooks
    grad_cam.remove_hooks()