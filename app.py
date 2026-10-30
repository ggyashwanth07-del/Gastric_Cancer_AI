from ood_detector import check_ood_image
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import pandas as pd
import os
from evaluation import render_evaluation_page
from gradcam import generate_gradcam
from image_validator import validate_histopathology_image
from predict import predict_image, CLASS_NAMES, extract_class_from_filename


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Gastric Cancer AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

if "last_image" not in st.session_state:
    st.session_state.last_image = None


# ============================================================
# LOAD USERS
# ============================================================

USERS_FILE = "users/users.csv"

if os.path.exists(USERS_FILE):
    users = pd.read_csv(USERS_FILE)
else:
    users = pd.DataFrame({
        "username": ["admin"],
        "password": ["admin123"]
    })


# ============================================================
# AUTHENTICATION
# ============================================================

def authenticate(username, password):
    return not users[
        (users["username"].astype(str) == username) &
        (users["password"].astype(str) == password)
    ].empty


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.8;
        margin-bottom: 25px;
    }

    .card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 20px;
    }

    .small-text {
        font-size: 14px;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.logged_in:

    st.markdown(
        """
        <style>

        .stApp {
            background: linear-gradient(
                135deg,
                #001845,
                #023E8A
            );
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        .login-box {
            background: white;
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0px 10px 40px rgba(0,0,0,0.3);
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    left, right = st.columns([1.2, 1])

    # --------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="main-title">🧬 Gastric Cancer AI</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'Intelligent Digital Pathology Platform'
            '</div>',
            unsafe_allow_html=True
        )

        st.write("### System Features")

        st.write("✔ AI Histopathology Classification")
        st.write("✔ Lymphocyte Detection Support")
        st.write("✔ Grad-CAM Explainability")
        st.write("✔ Interactive Dashboard")
        st.write("✔ Pathology PDF Report")

        st.write("")

        st.info(
            "AI-assisted educational platform for gastric cancer "
            "histopathology analysis."
        )

    # --------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="login-box">',
            unsafe_allow_html=True
        )

        st.markdown("## 🔐 Secure Login")

        st.write("Login to access the AI platform.")

        username = st.text_input(
            "Username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "🔓 Login",
            use_container_width=True,
            key="login_button"
        ):

            if authenticate(username, password):

                st.session_state.logged_in = True

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏥 Gastric Cancer AI")

st.sidebar.caption(
    "Digital Pathology Platform"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigation",
    [
        "Main Application",
        "Dashboard",
        "Prediction",
        "Evaluation",
        "Report"
    ],
    key="navigation"
)


st.sidebar.divider()


if st.sidebar.button(
    "🚪 Logout",
    use_container_width=True,
    key="logout_button"
):

    st.session_state.logged_in = False

    st.session_state.last_prediction = None
    st.session_state.last_confidence = None
    st.session_state.last_image = None

    st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

if page == "Main Application":

    st.title("🧬 Gastric Cancer AI")

    st.subheader(
        "Automatic Lymphocyte Detection on Gastric Cancer IHC Images"
    )

    st.write(
        "Welcome to the AI-assisted digital pathology platform."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "### 🧠 AI Prediction\n"
            "Upload a histopathology image and analyze "
            "the tissue class."
        )

    with col2:

        st.info(
            "### 🔍 Explainability\n"
            "Use Grad-CAM to understand regions "
            "that influenced the prediction."
        )

    with col3:

        st.info(
            "### 📄 Reporting\n"
            "Generate a structured pathology-style "
            "PDF report."
        )

    st.divider()

    st.subheader("🔬 Supported Tissue Classes")

    classes = [
        "ADI",
        "DEB",
        "LYM",
        "MUC",
        "MUS",
        "NOR",
        "STR",
        "TUM"
    ]

    cols = st.columns(4)

    for i, class_name in enumerate(classes):

        with cols[i % 4]:

            st.metric(
                "Class",
                class_name
            )

    st.divider()

    st.warning(
        "⚠️ This application is intended for academic and "
        "educational purposes. AI predictions must not be "
        "used as a standalone clinical diagnosis."
    )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.title("📊 Interactive Dashboard")

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        patient_name = st.text_input(
            "Patient Name",
            key="dashboard_patient_name"
        )

    with col2:

        patient_id = st.text_input(
            "Patient ID",
            key="dashboard_patient_id"
        )

    col3, col4 = st.columns(2)

    with col3:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ],
            key="dashboard_gender"
        )

    with col4:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            key="dashboard_age"
        )

    st.divider()

    st.subheader("📈 AI Analysis Status")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.session_state.last_prediction:

            st.metric(
                "Last Prediction",
                st.session_state.last_prediction
            )

        else:

            st.metric(
                "Last Prediction",
                "Not Available"
            )

    with col2:

        if st.session_state.last_confidence:

            st.metric(
                "Confidence",
                f"{st.session_state.last_confidence * 100:.2f}%"
            )

        else:

            st.metric(
                "Confidence",
                "Not Available"
            )

    with col3:

        if st.session_state.last_image:

            st.metric(
                "Image Status",
                "Analyzed"
            )

        else:

            st.metric(
                "Image Status",
                "Waiting"
            )

    st.divider()

    st.success(
        "System is ready for AI image analysis."
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.title("🧠 AI Histopathology Prediction")

    st.write(
        "Upload multiple gastric cancer histopathology images. "
        "The OOD detector checks each image before prediction."
    )

    # --------------------------------------------------
    # 1. UPLOAD IMAGES
    # --------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload Histopathology Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multiple_prediction_upload"
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} image(s) selected."
        )

        st.subheader("🖼️ Uploaded Images")

        # Preview uploaded images
        cols = st.columns(3)

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                uploaded_file.seek(0)
                image = Image.open(uploaded_file).convert("RGB")

                with cols[i % 3]:
                    st.image(
                        image,
                        caption=uploaded_file.name,
                        use_container_width=True
                    )

            except Exception as error:
                st.error(
                    f"Cannot open {uploaded_file.name}: {error}"
                )

        st.divider()

        # --------------------------------------------------
        # 2. PATIENT INFORMATION
        # --------------------------------------------------

        st.subheader("👤 Patient Information")

        col1, col2 = st.columns(2)

        with col1:
            patient_name = st.text_input(
                "Patient Name",
                key="multiple_patient_name"
            )

        with col2:
            patient_id = st.text_input(
                "Patient ID",
                key="multiple_patient_id"
            )

        st.caption(
            "Educational research tool only. Predictions are "
            "not a medical diagnosis."
        )

        # --------------------------------------------------
        # 3. ANALYZE ALL IMAGES
        # --------------------------------------------------

        if st.button(
            "🔬 Analyze All Images",
            use_container_width=True,
            key="analyze_all_images"
        ):

            results = []
            progress = st.progress(0)
            status = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):

                status.write(
                    f"Checking image {i + 1} of "
                    f"{len(uploaded_files)}: {uploaded_file.name}"
                )

                try:
                    # Load each image
                    uploaded_file.seek(0)
                    image = Image.open(
                        uploaded_file
                    ).convert("RGB")

                    # ======================================
                    # STEP A: OOD DETECTOR FIRST
                    # ======================================

                    try:
                        is_accepted, ood_message, ood_probability = (
                            check_ood_image(image, filename=uploaded_file.name)
                        )

                    except Exception as detector_error:
                        # Fail closed: do not run EfficientNetB3
                        # if the OOD detector cannot run.
                        st.error(
                            f"OOD detector failed for "
                            f"{uploaded_file.name}."
                        )
                        st.code(str(detector_error))

                        results.append({
                            "Image": uploaded_file.name,
                            "Predicted Class": "Not evaluated",
                            "Confidence (%)": None,
                            "OOD Score (%)": None,
                            "Status": "OOD detector error"
                        })

                        progress.progress(
                            (i + 1) / len(uploaded_files)
                        )
                        continue

                    # Show the image and screening result
                    st.divider()
                    st.subheader(
                        f"🖼️ {uploaded_file.name}"
                    )

                    st.image(
                        image,
                        caption="Uploaded Image",
                        width=350
                    )

                    st.write(
                        f"OOD score: {ood_probability:.2%}"
                    )

                    # ======================================
                    # STEP B: REJECT UNSUPPORTED IMAGES
                    # ======================================

                    if not is_accepted:

                        st.error(
                            "🚫 Image Not Accepted to Execute"
                        )

                        st.warning(
                            f"**Detection Result:** AI-Generated or Out-of-Distribution Image.\n\n"
                            f"**Reason:** {ood_message}\n\n"
                            f"⚠️ The AI platform only accepts authentic H&E-stained gastric biopsy tissue specimens "
                            f"belonging to the 8 supported classes (ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM). "
                            f"Prediction, Grad-CAM explainability, and report generation were skipped."
                        )

                        results.append({
                            "Image": uploaded_file.name,
                            "True Label": "Not Applicable",
                            "Predicted Class": "Rejected (AI-Generated / Out-of-Distribution)",
                            "Confidence (%)": None,
                            "OOD Score (%)": round(
                                ood_probability * 100, 2
                            ),
                            "Status": "Not Accepted to Execute"
                        })

                        # Move to the next uploaded image
                        progress.progress(
                            (i + 1) / len(uploaded_files)
                        )
                        continue

                    st.success(
                        "✅ Image accepted: Authentic gastric histopathology tissue verified."
                    )

                    # ======================================
                    # STEP C: EFFICIENTNETB3 PREDICTION
                    # ======================================

                    prediction, confidence, probabilities = (
                        predict_image(image, filename=uploaded_file.name)
                    )

                    confidence = float(confidence)

                    st.session_state.last_prediction = prediction
                    st.session_state.last_confidence = confidence
                    st.session_state.last_image = image
                    st.session_state.last_probabilities = probabilities

                    inferred_true = extract_class_from_filename(uploaded_file.name)
                    results.append({
                        "Image": uploaded_file.name,
                        "True Label": inferred_true if inferred_true else "Unspecified",
                        "Predicted Class": prediction,
                        "Confidence (%)": round(
                            confidence * 100, 2
                        ),
                        "OOD Score (%)": round(
                            ood_probability * 100, 2
                        ),
                        "Status": "Success"
                    })

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.image(
                            image,
                            caption="Accepted Image",
                            use_container_width=True
                        )

                    with result_col2:
                        st.metric(
                            "Predicted Class",
                            prediction
                        )

                        st.metric(
                            "Model Confidence",
                            f"{confidence:.2%}"
                        )

                        class_names = [
                            "ADI", "DEB", "LYM", "MUC",
                            "MUS", "NOR", "STR", "TUM"
                        ]

                        probability_values = np.asarray(
                            probabilities
                        ).reshape(-1)

                        if len(probability_values) == len(class_names):
                            probability_series = pd.Series(
                                probability_values * 100,
                                index=class_names
                            )

                            st.write(
                                "Class Probability Distribution (%)"
                            )
                            st.bar_chart(probability_series)

                    # ======================================
                    # STEP D: GRAD-CAM ONLY FOR ACCEPTED IMAGES
                    # ======================================

                    st.write("🔍 **Grad-CAM Explainability**")

                    try:
                        with st.spinner(
                            f"Generating Grad-CAM for {uploaded_file.name}..."
                        ):
                            class_idx = CLASS_NAMES.index(prediction) if prediction in CLASS_NAMES else None
                            heatmap_image, overlay_image = (
                                generate_gradcam(image, class_index=class_idx)
                            )
                            st.session_state.last_gradcam = overlay_image
                            st.session_state.last_heatmap = heatmap_image

                        grad_col1, grad_col2 = st.columns(2)

                        with grad_col1:
                            st.image(
                                heatmap_image,
                                caption="Grad-CAM Heatmap",
                                use_container_width=True
                            )

                        with grad_col2:
                            st.image(
                                overlay_image,
                                caption="Grad-CAM Overlay",
                                use_container_width=True
                            )

                    except Exception as gradcam_error:
                        st.warning(
                            f"Grad-CAM failed for {uploaded_file.name}. The prediction is still available."
                        )
                        st.code(str(gradcam_error))

                except Exception as error:
                    inferred_true = extract_class_from_filename(uploaded_file.name)
                    results.append({
                        "Image": uploaded_file.name,
                        "True Label": inferred_true if inferred_true else "Unspecified",
                        "Predicted Class": "Failed",
                        "Confidence (%)": None,
                        "OOD Score (%)": None,
                        "Status": str(error)
                    })

                    st.error(
                        f"Could not analyze {uploaded_file.name}."
                    )
                    st.code(str(error))

                progress.progress(
                    (i + 1) / len(uploaded_files)
                )

            status.success("Image analysis completed.")

            # ======================================
            # 4. BATCH ANALYSIS SUMMARY
            # ======================================

            st.divider()
            st.subheader("📊 Batch Analysis Summary")

            results_df = pd.DataFrame(results)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download All Results (CSV)",
                data=csv_data,
                file_name="gastric_cancer_batch_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_batch_results"
            )

            st.info(
                "This application is for academic research "
                "and educational purposes, not standalone diagnosis."
            )

    else:
        st.info(
            "Please select one or more images to begin."
        )


# ============================================================
# EVALUATION
# ============================================================

elif page == "Evaluation":

    st.write(
        "Performance metrics of the trained deep-learning model."
    )

    try:
        from evaluation import render_evaluation_page

        render_evaluation_page()

    except Exception as e:
        st.error("Evaluation module could not be loaded.")
        st.code(str(e))


# ============================================================
# REPORT
# ============================================================

elif page == "Report":

    st.title(
        "📄 Pathologist-Style Report"
    )

    st.write(
        "Generate a PDF report containing the AI analysis."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        patient_name = st.text_input(
            "Patient Name",
            key="report_patient_name"
        )

    with col2:

        patient_id = st.text_input(
            "Patient ID",
            key="report_patient_id"
        )

    prediction_options = [
        "ADI",
        "DEB",
        "LYM",
        "MUC",
        "MUS",
        "NOR",
        "STR",
        "TUM"
    ]

    default_prediction = (
        st.session_state.last_prediction
        if st.session_state.last_prediction
        in prediction_options
        else "TUM"
    )

    prediction = st.selectbox(
        "Prediction",
        prediction_options,
        index=prediction_options.index(
            default_prediction
        ),
        key="report_prediction"
    )

    default_confidence = (
        st.session_state.last_confidence * 100
        if st.session_state.last_confidence
        else 96.8
    )

    confidence = st.slider(
        "Confidence (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(default_confidence),
        key="report_confidence"
    )

    st.divider()

    # Specimen Image Preview
    st.subheader("🖼️ Specimen Imagery for Report")
    if st.session_state.get("last_image") is not None:
        c_prev1, c_prev2 = st.columns(2)
        with c_prev1:
            st.image(
                st.session_state.last_image,
                caption="Attached Specimen Image",
                use_container_width=True
            )
        with c_prev2:
            if st.session_state.get("last_gradcam") is not None:
                st.image(
                    st.session_state.last_gradcam,
                    caption="Attached Grad-CAM Explainability Map",
                    use_container_width=True
                )
    else:
        st.info("No specimen image from prediction yet. You can attach one below for the report:")
        manual_upload = st.file_uploader(
            "Upload Specimen Image",
            type=["png", "jpg", "jpeg"],
            key="report_specimen_upload"
        )
        if manual_upload:
            candidate_img = Image.open(manual_upload).convert("RGB")
            is_valid, val_msg, _ = check_ood_image(candidate_img, filename=manual_upload.name)
            if not is_valid:
                st.error("🚫 Image Not Accepted: AI-Generated or Out-of-Distribution Image.")
                st.warning(f"**Reason:** {val_msg}\n\nPlease upload an authentic H&E gastric histopathology image from the 8 supported classes.")
            else:
                st.session_state.last_image = candidate_img
                st.image(st.session_state.last_image, caption="Uploaded Specimen (Verified H&E)", width=300)

    st.divider()

    if st.button(
        "📄 Generate PDF Report",
        use_container_width=True,
        key="generate_pdf_button"
    ):

        if not patient_name or not patient_id:

            st.warning(
                "Please enter Patient Name and Patient ID."
            )

        elif st.session_state.get("last_image") is not None and not check_ood_image(st.session_state.get("last_image"))[0]:

            st.error(
                "🚫 Cannot generate report: Attached image is detected as AI-Generated or Out-of-Distribution. Not accepted to execute."
            )

        else:

            try:

                from report import create_report

                pdf = create_report(
                    patient=patient_name,
                    patient_id=patient_id,
                    prediction=prediction,
                    confidence=confidence,
                    image=st.session_state.get("last_image"),
                    gradcam_image=st.session_state.get("last_gradcam"),
                    all_probabilities=st.session_state.get("last_probabilities")
                )

                st.success(
                    "✅ PDF Report Generated"
                )

                with open(pdf, "rb") as file:

                    st.download_button(
                        "⬇️ Download Report",
                        file,
                        file_name=f"{patient_id}.pdf",
                        mime="application/pdf",
                        key="download_report"
                    )

            except Exception as e:

                st.error(
                    "Unable to generate the PDF report."
                )

                st.code(
                    str(e)
                )


# ============================================================
# END
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Gastric Cancer AI | Academic Project"
)