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
from google import genai


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

st.title(
    "🩻 Generative AI-Enhanced Explainable Chest X-Ray Screening"
)

st.caption(
    "Hernia-focused research prototype using DenseNet121, "
    "GAN-based augmentation, Grad-CAM and Generative AI"
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

    model = models.densenet121(
        weights=None
    )

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
            "❌ Model file not found:\n"
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

classes = [
    "Hernia",
    "Normal"
]


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# OOD / NON-X-RAY VALIDATION FILTER
# =========================================================

def validate_chest_xray(img_pil):

    """
    Validates whether the uploaded image has
    grayscale and contrast characteristics
    expected from a chest radiograph.

    This is a lightweight input validation filter,
    not a clinical image-quality assessment.
    """

    img_np = np.array(
        img_pil
    )

    # -----------------------------------------------------
    # 1. COLOR VARIANCE CHECK
    # -----------------------------------------------------

    if (
        len(img_np.shape) == 3
        and img_np.shape[2] == 3
    ):

        r = img_np[:, :, 0]

        g = img_np[:, :, 1]

        b = img_np[:, :, 2]

        color_variance = np.mean(

            np.abs(
                r.astype(float)
                - g.astype(float)
            )

            +

            np.abs(
                g.astype(float)
                - b.astype(float)
            )
        )

        if color_variance > 12.0:

            return (
                False,

                f"Color photo detected "
                f"(Color variance: "
                f"{color_variance:.1f}). "
                f"X-rays should be grayscale."
            )


    # -----------------------------------------------------
    # 2. CONTRAST / ILLUMINATION CHECK
    # -----------------------------------------------------

    gray = img_pil.convert(
        "L"
    )

    gray_np = np.array(
        gray
    )

    std_dev = np.std(
        gray_np
    )

    mean_val = np.mean(
        gray_np
    )

    if (
        std_dev < 18.0
        or mean_val < 15.0
        or mean_val > 240.0
    ):

        return (
            False,

            "Low contrast or invalid "
            "illumination distribution. "
            "Please upload a standard radiograph."
        )


    return (
        True,
        "Valid X-Ray"
    )


# =========================================================
# GRAD-CAM
# =========================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None


        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )


        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = (
            output.detach()
        )


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0].detach()
        )


    def remove_hooks(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# =========================================================
# GEMINI REAL-TIME GENERATIVE AI EXPLANATION
# =========================================================

def generate_gemini_explanation(
    pred_class,
    confidence,
    hernia_prob,
    normal_prob,
    is_borderline,
    image_width,
    image_height,
    gradcam_status
):

    try:

        # -------------------------------------------------
        # GET API KEY
        # -------------------------------------------------

        try:

            api_key = st.secrets[
                "GEMINI_API_KEY"
            ]

        except Exception:

            api_key = os.getenv(
                "GEMINI_API_KEY"
            )


        if not api_key:

            st.warning(
                "⚠️ Gemini API key is not configured."
            )

            return None


        # -------------------------------------------------
        # GEMINI CLIENT
        # -------------------------------------------------

        client = genai.Client(
            api_key=api_key
        )


        # -------------------------------------------------
        # CLASSIFICATION STATUS
        # -------------------------------------------------

        classification_status = (

            "AMBIGUOUS / BORDERLINE"

            if is_borderline

            else

            "HIGH-CONFIDENCE MODEL OUTPUT"
        )

        # Grad-CAM is generated separately by the application.
        # Gemini receives only its status, not the X-ray or heatmap.
        gradcam_information = gradcam_status


        # -------------------------------------------------
        # TECHNICAL PROMPT
        # -------------------------------------------------

        prompt = f"""

You are an AI explanation assistant for an
academic chest X-ray screening research prototype.

The DenseNet121 model has ALREADY performed
the classification.

Your ONLY task is to explain the supplied
machine-learning output.

Do NOT perform a new prediction.

IMPORTANT RULES:

- Do NOT diagnose the patient.
- Do NOT make a new medical prediction.
- Do NOT analyze the X-ray image.
- Do NOT invent radiological findings.
- Do NOT claim that the result is clinically confirmed.
- Do NOT provide medical advice.
- Do NOT mention findings that are not supplied.
- Model probability is NOT the same as clinical certainty.
- Use ONLY the supplied values.
- Do not claim that Grad-CAM proves the presence or location of disease.
- If Grad-CAM is generated, describe it only as a model explainability visualization.
- If Grad-CAM is not generated, do not imply that a heatmap was analyzed.
- Keep the explanation objective and technical.

MODEL OUTPUT:

Prediction:
{pred_class}

Model Confidence:
{confidence:.2f}%

Hernia Probability:
{hernia_prob:.2f}%

Normal Probability:
{normal_prob:.2f}%

Classification Status:
{classification_status}

Input Image Resolution:
{image_width} x {image_height}

Grad-CAM Status:
{gradcam_information}

Return the explanation in EXACTLY these four sections.
Use the section headings exactly as written below.
Write 1 concise technical sentence under each heading.
Do not add any other headings, introduction, conclusion, bullets, or extra text.

### 🧠 Model Interpretation
Explain what the supplied DenseNet121 prediction means from a machine-learning perspective.

### 📊 Probability Analysis
Interpret the supplied Hernia and Normal probabilities and state whether the model output is relatively confident or borderline.

### 🔥 Explainability
Use only the supplied Grad-CAM status. If generated, explain that Grad-CAM is a visualization of image regions that contributed relatively more strongly to the model prediction. Do not claim it proves disease or that you analyzed the heatmap. If not generated, state that Grad-CAM was not generated for this case.

### ⚠️ Research Limitation
Clearly state that this is an academic research-model output, that model probability is not clinical certainty, and that it is NOT a clinical diagnosis.

Use only the supplied values. Keep the tone objective, concise, and technical."""


        # -------------------------------------------------
        # GEMINI MODEL FALLBACK CHAIN
        # -------------------------------------------------

        models_to_try = [

            "gemini-3.8-flash",

            "gemini-3.7-flash",

            "gemini-3.6-flash"
        ]


        # -------------------------------------------------
        # TRY AVAILABLE MODELS
        # -------------------------------------------------

        for model_name in models_to_try:

            try:

                response = (
                    client.models.generate_content(

                        model=model_name,

                        contents=prompt
                    )
                )


                if response.text:

                    return (
                        response.text.strip()
                    )


            except Exception as e:

                error_text = str(
                    e
                ).upper()

                error_code = getattr(
                    e,
                    "code",
                    None
                )


                # -----------------------------------------
                # TEMPORARY SERVER / SSL / CONNECTION ERROR
                # -----------------------------------------

                is_temporary_error = (

                    error_code == 503

                    or

                    "503" in error_text

                    or

                    "UNAVAILABLE" in error_text

                    or

                    "UNEXPECTED_EOF_WHILE_READING"
                    in error_text

                    or

                    "SSL" in error_text

                    or

                    "CONNECTION" in error_text

                    or

                    "CONNECTION RESET"
                    in error_text

                    or

                    "TIMEOUT" in error_text
                )


                if is_temporary_error:

                    # Try next Gemini model
                    continue


                # -----------------------------------------
                # QUOTA / RATE LIMIT
                # -----------------------------------------

                is_quota_error = (

                    error_code == 429

                    or

                    "429" in error_text

                    or

                    "RESOURCE_EXHAUSTED"
                    in error_text
                )


                if is_quota_error:

                    st.warning(
                        "⚠️ Gemini API rate or quota "
                        "limit has been reached. "
                        "Please try again later."
                    )

                    return None


                # -----------------------------------------
                # OTHER API ERROR
                # -----------------------------------------

                st.error(
                    "Gemini API error: "
                    f"{str(e)}"
                )

                return None


        # -------------------------------------------------
        # ALL MODELS FAILED
        # -------------------------------------------------

        st.warning(
            "⚠️ Gemini is temporarily unavailable. "
            "The DenseNet121 screening result remains "
            "available and was not affected."
        )

        return None


    except Exception as e:

        st.error(
            "Gemini integration error: "
            f"{str(e)}"
        )

        return None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "📥 Upload X-Ray"
)


uploaded_file = (
    st.sidebar.file_uploader(

        "Choose a chest X-ray image",

        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )
)


st.sidebar.caption(
    "Supported formats: PNG, JPG, JPEG"
)


# =========================================================
# NO IMAGE
# =========================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a chest X-ray image "
        "from the sidebar to start "
        "the screening analysis."
    )


    st.markdown(
        "### About this prototype"
    )


    st.write(
        "This academic research prototype "
        "uses a DenseNet121 binary classifier "
        "trained with real and selected synthetic "
        "Hernia images. Grad-CAM provides visual "
        "explainability, while an OOD filter helps "
        "reject obvious non-radiograph inputs."
    )


    st.warning(
        "⚠️ Research prototype only. "
        "This system is not a medical diagnosis."
    )


# =========================================================
# IMAGE PROCESSING
# =========================================================

else:

    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    try:

        raw_image = (
            Image.open(
                uploaded_file
            ).convert("RGB")
        )

    except Exception as e:

        st.error(
            f"Unable to read image: {e}"
        )

        st.stop()


    # -----------------------------------------------------
    # IMAGE RESOLUTION
    # -----------------------------------------------------

    width, height = (
        raw_image.size
    )


    if (
        width < 100
        or height < 100
    ):

        st.error(
            "Image resolution is too small. "
            "Please upload a clear X-ray image."
        )

        st.stop()


    # -----------------------------------------------------
    # OOD VALIDATION
    # -----------------------------------------------------

    is_valid, validation_reason = (
        validate_chest_xray(
            raw_image
        )
    )


    if not is_valid:

        st.error(
            "❌ **INVALID / NON-MEDICAL "
            "IMAGE DETECTED**"
        )


        st.warning(
            f"**Reason:** "
            f"{validation_reason}"
        )


        st.image(
            raw_image,
            caption="Uploaded Image (Rejected)",
            width=350
        )


        st.info(
            "ℹ️ The system rejected this "
            "input before model inference "
            "to reduce invalid classifications. "
            "Please upload a genuine grayscale "
            "Chest X-Ray."
        )


        st.stop()


    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    input_tensor = transform(
        raw_image
    )


    input_tensor = (
        input_tensor.unsqueeze(0)
    )


    input_tensor = (
        input_tensor.to(device)
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    with torch.no_grad():

        output = model(
            input_tensor
        )


        probs = F.softmax(
            output,
            dim=1
        )[0]


        conf, predicted = (
            torch.max(
                probs,
                0
            )
        )


    # -----------------------------------------------------
    # MODEL VALUES
    # -----------------------------------------------------

    pred_class = (
        classes[
            predicted.item()
        ]
    )


    confidence = (
        conf.item() * 100
    )


    hernia_prob = (
        probs[0].item() * 100
    )


    normal_prob = (
        probs[1].item() * 100
    )


    # -----------------------------------------------------
    # BORDERLINE CHECK
    # -----------------------------------------------------

    is_borderline = (
        confidence < 78.0
    )


    # =====================================================
    # SCREENING RESULT
    # =====================================================

    st.subheader(
        "Screening Result"
    )


    result_col1, result_col2 = (
        st.columns([1, 1])
    )


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with result_col1:

        st.image(
            raw_image,
            caption="Uploaded Chest X-Ray",
            use_container_width=True
        )


    # -----------------------------------------------------
    # RESULT DETAILS
    # -----------------------------------------------------

    with result_col2:

        if is_borderline:

            verdict_text = (
                "AMBIGUOUS / BORDERLINE CASE"
            )


            st.warning(
                f"### ⚠️ {verdict_text}"
            )


            st.info(
                f"Model prediction confidence "
                f"is moderate ({confidence:.2f}%). "
                "This binary research model may "
                "produce uncertain outputs for "
                "non-standard inputs or conditions "
                "outside its training scope."
            )


        elif pred_class == "Hernia":

            verdict_text = (
                "HERNIA DETECTED"
            )


            st.error(
                f"### {verdict_text}"
            )


        else:

            verdict_text = (
                "NORMAL"
            )


            st.success(
                f"### {verdict_text}"
            )


        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        st.caption(
            "Confidence represents the model's "
            "predicted probability, not medical certainty."
        )


        # =================================================
        # REPORT DOWNLOAD
        # =================================================

        timestamp = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


        report_content = f"""
==================================================
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
TECHNICAL DETAILS
--------------------------------------------------

Model Architecture : DenseNet121
Classification      : Binary (Hernia / Normal)
Data Augmentation   : GAN-Augmented Training
Confidence Threshold: 78%

Status              :
{"Borderline / Ambiguous" if is_borderline else "High-Confidence Model Output"}

--------------------------------------------------
DISCLAIMER
--------------------------------------------------

This report is generated by an AI research
prototype for academic and research purposes.

The output does not constitute a clinical
diagnosis and must not replace professional
radiological evaluation.

==================================================
"""


        st.write("---")


        st.download_button(

            label=(
                "📄 Download Screening Report (.txt)"
            ),

            data=report_content,

            file_name=(
                f"Hernia_Screening_Report_"
                f"{uploaded_file.name.split('.')[0]}.txt"
            ),

            mime="text/plain"
        )


# =========================================================
# PROBABILITY SECTION
# =========================================================

    st.divider()


    st.subheader(
        "Prediction Probabilities"
    )


    prob_col1, prob_col2 = (
        st.columns(2)
    )


    # -----------------------------------------------------
    # HERNIA PROBABILITY
    # -----------------------------------------------------

    with prob_col1:

        st.write(
            "**Hernia**"
        )


        st.progress(
            int(
                round(
                    hernia_prob
                )
            )
        )


        st.caption(
            f"{hernia_prob:.2f}%"
        )


    # -----------------------------------------------------
    # NORMAL PROBABILITY
    # -----------------------------------------------------

    with prob_col2:

        st.write(
            "**Normal**"
        )


        st.progress(
            int(
                round(
                    normal_prob
                )
            )
        )


        st.caption(
            f"{normal_prob:.2f}%"
        )


# =========================================================
# GEMINI GENERATIVE AI EXPLANATION
# =========================================================

    st.divider()


    st.subheader(
        "🤖 Generative AI Explanation"
    )


    st.caption(
        "Generate a technical explanation of "
        "the DenseNet121 screening result using "
        "Google Gemini."
    )


    if st.button(
        "🤖 Explain This Result",
        use_container_width=True
    ):

        with st.spinner(
            "🤖 Generating AI explanation..."
        ):

            gemini_explanation = (
                generate_gemini_explanation(

                    pred_class=pred_class,

                    confidence=confidence,

                    hernia_prob=hernia_prob,

                    normal_prob=normal_prob,

                    is_borderline=is_borderline,

                    image_width=width,

                    image_height=height,

                    gradcam_status=(
                        "Generated for high-confidence Hernia prediction"
                        if pred_class == "Hernia" and not is_borderline
                        else "Not generated because prediction is Normal or Borderline"
                    )
                )
            )


        if gemini_explanation:

            st.success(
                "✅ AI explanation generated"
            )


            with st.container(
                border=True
            ):

                st.markdown(
                    "### 🧠 AI Explanation"
                )


                st.markdown(
                    gemini_explanation
                )


                st.caption(
                    "Generated by Gemini from the current "
                    "DenseNet121 model output and Grad-CAM status. "
                    "Gemini does not perform independent "
                    "medical diagnosis in this application."
                )


        else:

            st.warning(
                "⚠️ Gemini explanation is currently "
                "unavailable. The DenseNet121 result "
                "remains available."
            )


# =========================================================
# GRAD-CAM
# =========================================================

    st.divider()


    st.subheader(
        "Explainable AI — Grad-CAM"
    )


    # -----------------------------------------------------
    # HIGH-CONFIDENCE HERNIA ONLY
    # -----------------------------------------------------

    if (
        pred_class == "Hernia"
        and not is_borderline
    ):


        st.write(
            "Grad-CAM highlights image regions "
            "that contributed more strongly to "
            "the DenseNet121 prediction."
        )


        # ---------------------------------------------
        # TARGET LAYER
        # ---------------------------------------------

        target_layer = (
            model
            .features
            .denseblock4
            .denselayer16
            .conv2
        )


        grad_cam = GradCAM(
            model,
            target_layer
        )


        # ---------------------------------------------
        # FORWARD + BACKWARD
        # ---------------------------------------------

        model.zero_grad()


        output = model(
            input_tensor
        )


        target_score = (
            output[
                0,
                predicted.item()
            ]
        )


        target_score.backward()


        # ---------------------------------------------
        # GET GRADIENTS
        # ---------------------------------------------

        gradients = (
            grad_cam
            .gradients
            .cpu()
            .numpy()[0]
        )


        activations = (
            grad_cam
            .activations
            .cpu()
            .numpy()[0]
        )


        # ---------------------------------------------
        # CHANNEL WEIGHTS
        # ---------------------------------------------

        weights = np.mean(
            gradients,
            axis=(1, 2)
        )


        cam = np.zeros(
            activations.shape[1:],
            dtype=np.float32
        )


        # ---------------------------------------------
        # BUILD CAM
        # ---------------------------------------------

        for i, weight in enumerate(
            weights
        ):

            cam += (
                weight
                * activations[i]
            )


        # ---------------------------------------------
        # RELU
        # ---------------------------------------------

        cam = np.maximum(
            cam,
            0
        )


        # ---------------------------------------------
        # NORMALIZE
        # ---------------------------------------------

        if cam.max() > 0:

            cam = (
                cam
                / cam.max()
            )


        # ---------------------------------------------
        # RESIZE CAM
        # ---------------------------------------------

        cam_image = (
            Image.fromarray(
                np.uint8(
                    cam * 255
                )
            )
            .resize(
                (224, 224)
            )
        )


        cam = (
            np.array(
                cam_image
            )
            / 255.0
        )


        # =================================================
        # DISPLAY GRAD-CAM
        # =================================================

        cam_col1, cam_col2 = (
            st.columns([1, 1])
        )


        # -------------------------------------------------
        # HEATMAP
        # -------------------------------------------------

        with cam_col1:

            fig, ax = plt.subplots(
                figsize=(5, 5)
            )


            ax.imshow(
                raw_image.resize(
                    (224, 224)
                )
            )


            ax.imshow(
                cam,
                cmap="jet",
                alpha=0.45
            )


            ax.axis(
                "off"
            )


            ax.set_title(
                "Grad-CAM Heatmap (Hernia Focus)"
            )


            st.pyplot(
                fig,
                clear_figure=True
            )


            plt.close(
                fig
            )


        # -------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------

        with cam_col2:

            st.markdown(
                "#### Interpretation"
            )


            st.write(
                "🔴 **Warm/Red regions** indicate "
                "areas that contributed relatively "
                "more strongly to the selected model "
                "prediction."
            )


            st.info(
                "Grad-CAM is an explainability "
                "visualization. A highlighted region "
                "does not prove a clinically confirmed "
                "pathology."
            )


        # -------------------------------------------------
        # REMOVE HOOKS
        # -------------------------------------------------

        grad_cam.remove_hooks()


    else:

        st.info(
            "ℹ️ Grad-CAM is generated only for a "
            "high-confidence Hernia prediction "
            "(≥78%). For Normal or Borderline "
            "outputs, heatmap generation is skipped."
        )


# =========================================================
# RESEARCH PERFORMANCE DASHBOARD
# =========================================================

st.divider()

st.subheader("📊 Research Performance Dashboard")

st.caption(
    "Evaluation summary for the trained DenseNet121 research prototype. "
    "Only verified evaluation values are displayed; unavailable metrics are not invented."
)

# The project test evaluation currently has a verified unseen-test accuracy.
# Precision, Recall, F1 and Confusion Matrix require the corresponding
# test-set prediction counts/artifact and are intentionally not fabricated.
verified_test_accuracy = 86.21

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        "Test Accuracy",
        f"{verified_test_accuracy:.2f}%",
        help="Verified accuracy on the unseen test evaluation used by the research prototype."
    )

with metric_col2:
    st.metric(
        "Precision",
        "N/A",
        help="Requires class-level test predictions/confusion-matrix counts."
    )

with metric_col3:
    st.metric(
        "Recall",
        "N/A",
        help="Requires class-level test predictions/confusion-matrix counts."
    )

with metric_col4:
    st.metric(
        "F1 Score",
        "N/A",
        help="Requires class-level test predictions/confusion-matrix counts."
    )

# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

info_col1, info_col2 = st.columns(2)

with info_col1:
    with st.container(border=True):
        st.markdown("### 🧠 Model Configuration")
        st.write("**Architecture:** DenseNet121")
        st.write("**Task:** Binary classification")
        st.write("**Classes:** Hernia / Normal")
        st.write("**Training:** Real + selected GAN-synthetic images")
        st.write("**Confidence threshold:** 78%")

with info_col2:
    with st.container(border=True):
        st.markdown("### 🔬 Explainability Pipeline")
        st.write("**OOD validation:** Enabled")
        st.write("**Generative AI:** Gemini")
        st.write("**Visual explainability:** Grad-CAM")
        st.write("**Grad-CAM rule:** High-confidence Hernia only")
        st.write("**Clinical diagnosis:** Not supported")

# ---------------------------------------------------------
# CONFUSION MATRIX PLACEHOLDER / DATA REQUIREMENT
# ---------------------------------------------------------

with st.expander("📈 Confusion Matrix & Detailed Evaluation", expanded=False):

    st.info(
        "The dashboard is ready for the confusion matrix, Precision, Recall, "
        "F1 Score and ROC-AUC. These values must come from the actual unseen "
        "test-set predictions. The app intentionally does not invent or "
        "estimate these metrics from the overall accuracy."
    )

    st.markdown("#### Required evaluation data")
    st.code(
        "True labels + DenseNet121 predictions for every test image\n"
        "→ Confusion Matrix\n"
        "→ Precision\n"
        "→ Recall\n"
        "→ F1 Score\n"
        "→ ROC-AUC"
    )

    st.caption(
        "Once the test-prediction artifact is added, this section can be upgraded "
        "to render the real confusion matrix and class-wise metrics automatically."
    )


# =========================================================
# FINAL DISCLAIMER
# =========================================================

    st.divider()


    st.caption(
        "⚠️ Research prototype for academic purposes only. "
        "Not intended for medical diagnosis or clinical "
        "decision-making."
    )