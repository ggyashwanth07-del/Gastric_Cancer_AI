import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from predict import CLASS_NAMES, extract_class_from_filename, predict_image

DATASET_DIR = r"C:\Gastric_Cancer_AI\dataset\HMU-GC-HE-30K\all_image"


def calculate_metrics(y_true, y_pred):
    """Calculate weighted and macro classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "precision_macro": precision_m,
        "recall_macro": recall_m,
        "f1_macro": f1_m,
    }


def plot_confusion_matrix(y_true, y_pred, normalized=False):
    """
    Generate an annotated high-resolution Confusion Matrix
    with per-cell counts and percentage breakdowns.
    """
    labels = CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=150)

    if normalized:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues", vmin=0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Accuracy Proportion", rotation=270, labelpad=15)
        thresh = 0.5
    else:
        cm_display = cm
        im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Number of Samples", rotation=270, labelpad=15)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")

    total_per_row = cm.sum(axis=1)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            row_total = total_per_row[i]
            pct = (val / row_total * 100.0) if row_total > 0 else 0.0
            val_comp = cm_display[i, j]
            color = "white" if val_comp > thresh else "black"
            cell_text = f"{val}\n({pct:.1f}%)" if not normalized else f"{pct:.1f}%\n(n={val})"
            ax.text(
                j, i, cell_text,
                ha="center", va="center",
                color=color, fontsize=8.5,
                fontweight="bold" if i == j else "normal"
            )

    ax.set_title("Confusion Matrix (Histopathology Classification)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Ground Truth Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("AI Predicted Label", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def render_evaluation_page():
    st.title("📊 Model Performance & Evaluation")
    st.write(
        "Assess classification accuracy, detailed precision-recall metrics, "
        "and confusion matrix across all 8 gastric histopathology tissue classes."
    )

    eval_tab1, eval_tab2 = st.tabs(["⚡ Live Dataset Evaluation", "📁 Upload Evaluation CSV"])

    # ---------------------------------------------------------
    # TAB 1: LIVE DATASET EVALUATION
    # ---------------------------------------------------------
    with eval_tab1:
        st.subheader("⚡ Live Model Evaluation on Ground-Truth Dataset")
        st.caption(
            "Directly benchmark the active EfficientNet-B3 model on verified histopathology "
            "images from each tissue class folder."
        )

        col_cfg1, col_cfg2 = st.columns([1.5, 2])
        with col_cfg1:
            samples_per_class = st.select_slider(
                "Test samples per class:",
                options=[5, 10, 15, 20, 30],
                value=10,
                key="eval_samples_slider"
            )
            total_eval = samples_per_class * len(CLASS_NAMES)
            st.info(f"Total benchmark images: **{total_eval}** ({samples_per_class} × 8 classes)")

        with col_cfg2:
            st.write("")
            st.write("")
            run_live_eval = st.button("🚀 Run Live Evaluation Benchmark", use_container_width=True, key="run_benchmark_btn")

        if run_live_eval:
            if not os.path.isdir(DATASET_DIR):
                st.error(f"Dataset directory not found: {DATASET_DIR}")
                return

            y_true_list = []
            y_pred_list = []
            records = []

            eval_progress = st.progress(0)
            eval_status = st.empty()

            sample_idx = 0
            for class_name in CLASS_NAMES:
                folder = os.path.join(DATASET_DIR, class_name)
                if not os.path.isdir(folder):
                    continue

                images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))][:samples_per_class]

                for img_name in images:
                    eval_status.write(f"Evaluating {class_name} ({sample_idx + 1}/{total_eval}): {img_name}")
                    img_path = os.path.join(folder, img_name)
                    pred_class, conf, _ = predict_image(img_path)

                    y_true_list.append(class_name)
                    y_pred_list.append(pred_class)
                    records.append({
                        "Image": img_name,
                        "True Label": class_name,
                        "Predicted Label": pred_class,
                        "Confidence (%)": round(conf * 100, 2),
                        "Correct": class_name == pred_class
                    })

                    sample_idx += 1
                    eval_progress.progress(sample_idx / total_eval)

            eval_status.success(f"Benchmark completed successfully! Analyzed {total_eval} images.")
            st.session_state.eval_df = pd.DataFrame(records)

        if "eval_df" in st.session_state and not st.session_state.eval_df.empty:
            c_head1, c_head2 = st.columns([3, 1])
            with c_head2:
                if st.button("🔄 Clear Benchmark", key="clear_live_eval_btn", use_container_width=True):
                    del st.session_state.eval_df
                    st.rerun()
            display_evaluation_results(st.session_state.eval_df["True Label"], st.session_state.eval_df["Predicted Label"], st.session_state.eval_df, key_prefix="live_eval")

    # ---------------------------------------------------------
    # TAB 2: UPLOAD CSV EVALUATION
    # ---------------------------------------------------------
    with eval_tab2:
        st.subheader("📁 Evaluate from Batch Analysis CSV")
        st.caption("Upload your exported batch prediction CSV or an evaluation table.")

        uploaded_file = st.file_uploader(
            "Upload Evaluation CSV",
            type=["csv"],
            key="evaluation_csv_uploader_v2"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df.columns = df.columns.astype(str).str.strip()

                st.write(f"📄 **Uploaded File:** `{uploaded_file.name}` ({len(df)} samples)")
                with st.expander("🔍 Preview CSV Table (First 5 Rows)", expanded=False):
                    st.dataframe(df.head(5), use_container_width=True)

                col_lower_map = {c.lower(): c for c in df.columns}

                # 1. Detect Prediction Column
                pred_candidates = [
                    "predicted class", "predicted_class", "predicted label", "predicted_label",
                    "prediction", "predictions", "predicted", "pred", "y_pred", "model prediction"
                ]
                default_pred_col = None
                for cand in pred_candidates:
                    if cand in col_lower_map:
                        default_pred_col = col_lower_map[cand]
                        break
                if default_pred_col is None:
                    for c in df.columns:
                        if "pred" in c.lower():
                            default_pred_col = c
                            break
                if default_pred_col is None and len(df.columns) > 0:
                    default_pred_col = df.columns[-1]

                # 2. Detect Ground Truth Column
                true_candidates = [
                    "true label", "true_label", "true class", "true_class",
                    "ground truth", "ground_truth", "actual", "actual label",
                    "actual_label", "actual class", "actual_class", "target", "label", "class", "y_true", "gt"
                ]
                default_true_col = None
                for cand in true_candidates:
                    if cand in col_lower_map and col_lower_map[cand] != default_pred_col:
                        default_true_col = col_lower_map[cand]
                        break

                # 3. Detect Image / Filename Column
                img_candidates = ["image", "filename", "file", "img", "name", "image_name", "image_path", "path"]
                default_img_col = None
                for cand in img_candidates:
                    if cand in col_lower_map:
                        default_img_col = col_lower_map[cand]
                        break

                # UI Selectors
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    pred_col_idx = list(df.columns).index(default_pred_col) if default_pred_col in df.columns else 0
                    selected_pred_col = st.selectbox(
                        "Prediction Column:",
                        options=list(df.columns),
                        index=pred_col_idx,
                        key="csv_pred_col_select"
                    )

                with col_sel2:
                    source_options = []
                    if default_img_col:
                        source_options.append(f"Infer from '{default_img_col}' filename")
                    elif any(c.lower() in ["image", "filename", "file"] for c in df.columns):
                        source_options.append("Infer from filename column")

                    for c in df.columns:
                        if c != selected_pred_col:
                            source_options.append(f"Column: {c}")

                    source_options.append("No Ground Truth (Analyze Batch Distribution Only)")

                    if default_true_col and f"Column: {default_true_col}" in source_options:
                        def_src_idx = source_options.index(f"Column: {default_true_col}")
                    elif len(source_options) > 0 and source_options[0].startswith("Infer"):
                        def_src_idx = 0
                    else:
                        def_src_idx = 0

                    selected_true_source = st.selectbox(
                        "Ground Truth Source:",
                        options=source_options,
                        index=def_src_idx,
                        key="csv_true_source_select"
                    )

                # Process predictions
                processed_df = df.copy()
                processed_df["normalized_pred"] = processed_df[selected_pred_col].apply(extract_class_from_filename)

                # Process ground truth
                if selected_true_source.startswith("Infer"):
                    img_col = default_img_col if default_img_col else "Image"
                    if img_col in processed_df.columns:
                        processed_df["normalized_true"] = processed_df[img_col].apply(extract_class_from_filename)
                    else:
                        for col in processed_df.columns:
                            if processed_df[col].astype(str).str.contains(r"\.(png|jpg|jpeg|tif|bmp)", case=False).any():
                                processed_df["normalized_true"] = processed_df[col].apply(extract_class_from_filename)
                                break
                        else:
                            processed_df["normalized_true"] = None
                elif selected_true_source.startswith("Column: "):
                    col_name = selected_true_source.replace("Column: ", "")
                    processed_df["normalized_true"] = processed_df[col_name].apply(extract_class_from_filename)
                else:
                    processed_df["normalized_true"] = None

                # Check if we have valid ground truth for evaluation
                valid_mask = (
                    processed_df["normalized_pred"].isin(CLASS_NAMES) &
                    processed_df["normalized_true"].isin(CLASS_NAMES)
                )
                eval_df = processed_df[valid_mask].copy()

                if not eval_df.empty:
                    eval_df["True Label"] = eval_df["normalized_true"]
                    eval_df["Predicted Label"] = eval_df["normalized_pred"]
                    eval_df["Correct"] = eval_df["True Label"] == eval_df["Predicted Label"]

                    if len(eval_df) < len(processed_df):
                        st.info(f"Evaluated **{len(eval_df)}** of {len(processed_df)} samples matching tissue classes.")

                    display_evaluation_results(eval_df["True Label"], eval_df["Predicted Label"], eval_df, key_prefix="csv_eval")
                else:
                    st.info("ℹ️ No ground truth class labels detected in the selected column/filename. Displaying Batch Prediction Distribution & Summary below.")
                    display_batch_distribution_results(processed_df, selected_pred_col)

            except Exception as e:
                st.error(f"Failed to process CSV: {e}")


def display_batch_distribution_results(df, pred_col):
    """Render batch prediction distributions, confidence overview, and ground-truth annotation tool."""
    st.divider()
    st.subheader("📊 Batch Prediction Summary & Class Distribution")

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    total_samples = len(df)
    col_kpi1.metric("Total Batch Samples", total_samples)

    # Check for confidence column
    conf_col = None
    for c in df.columns:
        if "conf" in c.lower():
            conf_col = c
            break

    if conf_col and pd.to_numeric(df[conf_col], errors="coerce").notnull().any():
        conf_series = pd.to_numeric(df[conf_col], errors="coerce").dropna()
        if conf_series.mean() <= 1.0:
            conf_series = conf_series * 100.0
        col_kpi2.metric("Mean Model Confidence", f"{conf_series.mean():.1f}%")
        high_conf_pct = (conf_series >= 85.0).mean() * 100.0
        col_kpi3.metric("High Confidence (≥85%)", f"{high_conf_pct:.1f}%")

    preds = df["normalized_pred"].fillna(df[pred_col].astype(str).str.upper())
    class_counts = preds.value_counts()

    dist_data = []
    for cls in CLASS_NAMES:
        cnt = int(class_counts.get(cls, 0))
        pct = (cnt / total_samples * 100.0) if total_samples > 0 else 0.0
        dist_data.append({
            "Class Code": cls,
            "Count": cnt,
            "Percentage": f"{pct:.1f}%"
        })

    c_chart, c_table = st.columns([1.2, 1])
    with c_chart:
        st.bar_chart(pd.Series({d["Class Code"]: d["Count"] for d in dist_data}))
    with c_table:
        st.dataframe(pd.DataFrame(dist_data), use_container_width=True, hide_index=True)

    # Interactive Annotation Assistant
    st.divider()
    st.subheader("✍️ Ground-Truth Benchmark Assistant")
    st.write(
        "To calculate the full Confusion Matrix and Classification Report, verify or assign the Ground Truth "
        "class for each sample in the table below and click **Evaluate Annotated Data**:"
    )

    editable_df = df.copy()
    if "Ground Truth" not in editable_df.columns:
        editable_df["Ground Truth"] = editable_df["normalized_pred"].fillna("NOR")

    edited = st.data_editor(
        editable_df,
        column_config={
            "Ground Truth": st.column_config.SelectboxColumn(
                "Ground Truth",
                options=CLASS_NAMES,
                required=True
            )
        },
        use_container_width=True,
        key="data_editor_annotation"
    )

    if st.button("🚀 Evaluate Annotated Data", key="eval_edited_btn"):
        y_t = edited["Ground Truth"].astype(str).str.strip().str.upper()
        y_p = editable_df["normalized_pred"].fillna(editable_df[pred_col].astype(str).str.upper())
        valid = y_t.isin(CLASS_NAMES) & y_p.isin(CLASS_NAMES)
        if valid.any():
            display_evaluation_results(y_t[valid], y_p[valid], edited[valid], key_prefix="annotated_eval")
        else:
            st.warning("Please select valid classes from ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM.")


def display_evaluation_results(y_true, y_pred, df_full, key_prefix="eval"):
    """Helper to render KPIs, Classification Report, and Confusion Matrix."""
    m = calculate_metrics(y_true, y_pred)

    st.divider()
    st.subheader("🎯 Overall Performance Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", f"{m['accuracy']:.2%}")
    c2.metric("Weighted F1-Score", f"{m['f1_weighted']:.2%}")
    c3.metric("Weighted Precision", f"{m['precision_weighted']:.2%}")
    c4.metric("Weighted Recall", f"{m['recall_weighted']:.2%}")

    # Classification Report
    st.subheader("📋 Comprehensive Classification Report")
    rep_dict = classification_report(
        y_true, y_pred,
        labels=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )
    rep_df = pd.DataFrame(rep_dict).transpose()

    # Format numeric columns for clean display
    for col in ["precision", "recall", "f1-score"]:
        if col in rep_df.columns:
            rep_df[col] = rep_df[col].apply(lambda x: f"{x:.2%}")
    if "support" in rep_df.columns:
        rep_df["support"] = rep_df["support"].astype(int)

    st.dataframe(rep_df, use_container_width=True)

    # Confusion Matrix
    st.subheader("🔲 Confusion Matrix")
    c_toggle1, c_toggle2 = st.columns([1, 3])
    with c_toggle1:
        norm_toggle = st.checkbox("Show as Normalized Proportions (%)", value=False, key=f"{key_prefix}_cm_norm_checkbox")

    fig = plot_confusion_matrix(y_true, y_pred, normalized=norm_toggle)
    st.pyplot(fig)
    plt.close(fig)

    # Detailed Sample Breakdown
    st.subheader("🔬 Evaluated Sample Breakdown")
    filter_choice = st.radio(
        "Filter results:",
        ["All Samples", "Correct Predictions Only", "Incorrect / Misclassified Only"],
        horizontal=True,
        key=f"{key_prefix}_eval_filter_radio"
    )

    display_df = df_full.copy()
    if "Correct" in display_df.columns:
        if filter_choice == "Correct Predictions Only":
            display_df = display_df[display_df["Correct"] == True]
        elif filter_choice == "Incorrect / Misclassified Only":
            display_df = display_df[display_df["Correct"] == False]

    st.dataframe(display_df, use_container_width=True)

    # Export options
    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            "⬇️ Download Classification Report (CSV)",
            data=pd.DataFrame(rep_dict).transpose().to_csv().encode("utf-8"),
            file_name="gastric_classification_report.csv",
            mime="text/csv",
            key=f"{key_prefix}_dl_class_report"
        )
    with c_dl2:
        st.download_button(
            "⬇️ Download Evaluated Data (CSV)",
            data=df_full.to_csv(index=False).encode("utf-8"),
            file_name="gastric_evaluated_data.csv",
            mime="text/csv",
            key=f"{key_prefix}_dl_eval_data"
        )
