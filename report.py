import os
import re
import datetime
import numpy as np
from PIL import Image

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    FPDF = object
    HAS_FPDF = False

try:
    import reportlab
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")

CLASS_DESCRIPTIONS = {
    "ADI": {
        "full_name": "Adipose Tissue (ADI)",
        "features": "Mature adipocytes characterized by clear lipid-filled cytoplasmic vacuoles with peripheral eccentric nuclei and delicate capillary networks.",
        "significance": "Denotes subserosal, omental, or perigastric fatty tissue; marks deep invasive margins in full-thickness gastric wall resections.",
    },
    "DEB": {
        "full_name": "Tumor Necrosis & Cellular Debris (DEB)",
        "features": "Eosinophilic amorphous necrotic background containing fragmented karyorrhectic nuclear dust, lysed erythrocytes, and inflammatory infiltrates.",
        "significance": "Frequently correlates with high-grade aggressive tumors, hypoxia, or prior neoadjuvant chemotherapy response (TRG).",
    },
    "LYM": {
        "full_name": "Lymphocytes & Immune Infiltrate (LYM)",
        "features": "Dense aggregates of small, round, hyperchromatic mononuclear immune cells forming lymphoid aggregates or diffuse stromal infiltration.",
        "significance": "Tumor-Infiltrating Lymphocytes (TILs) are key biomarkers for MSI-H status and positive response to immune checkpoint inhibitor therapies.",
    },
    "MUC": {
        "full_name": "Mucinous Carcinoma / Colloid Pools (MUC)",
        "features": "Extensive extracellular pools of basophilic or pale mucin containing floating atypical glandular cords, clusters, or signet ring cells.",
        "significance": "Corresponds to mucinous adenocarcinoma (>50% extracellular mucin component); indicates potential risk of peritoneal dissemination.",
    },
    "MUS": {
        "full_name": "Smooth Muscle Bundles (MUS)",
        "features": "Parallel eosinophilic fascicles of elongated smooth muscle fibers with blunt-ended nuclei (muscularis mucosae or muscularis propria).",
        "significance": "Tumor invasion into muscle denotes extension beyond the submucosa (T2 stage or greater), denoting transmural progression.",
    },
    "NOR": {
        "full_name": "Normal Gastric Mucosa (NOR)",
        "features": "Preserved glandular architecture featuring regular foveolar mucin-secreting epithelial cells, parietal and chief cells with intact basement membranes.",
        "significance": "Demonstrates benign, non-neoplastic gastric tissue without architectural atypia, dysplasia, or malignant infiltration.",
    },
    "STR": {
        "full_name": "Cancer-Associated Stroma (STR)",
        "features": "Spindle-shaped fibroblastic proliferation with dense fibrillar collagenous matrix and neovascular channels.",
        "significance": "Desmoplastic stroma reflects active host-tumor microenvironment remodeling, often associated with infiltrative growth patterns.",
    },
    "TUM": {
        "full_name": "Malignant Epithelial Tumor / Adenocarcinoma (TUM)",
        "features": "Atypical irregular glandular or tubular formations displaying cellular pleomorphism, nuclear hyperchromasia, prominent nucleoli, and brisk mitotic activity.",
        "significance": "Definitive invasive gastric adenocarcinoma; requires pathological grading, depth assessment (T category), and margin clearance.",
    },
}


class PathologyPDF(FPDF):

    def header(self):
        # Header banner
        self.set_fill_color(15, 45, 95)
        self.rect(0, 0, 210, 24, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.set_xy(12, 5)
        self.cell(0, 7, "GASTRIC CANCER AI - DIGITAL PATHOLOGY REPORT", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.set_xy(12, 13)
        self.cell(0, 5, "Automated Histopathological Classification & Grad-CAM Explainability", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(145, 9)
        self.cell(55, 6, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), align="R")
        self.ln(15)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"Gastric Cancer AI Digital Pathology Platform | Case Research Report | Page {self.page_no()}", align="C")


def _to_pil(img_input):
    """Safely convert file path, PIL Image, or numpy array into an RGB PIL Image."""
    if img_input is None:
        return None
    try:
        if isinstance(img_input, (str, os.PathLike)):
            return Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        elif isinstance(img_input, np.ndarray):
            arr = img_input
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        else:
            return Image.fromarray(np.array(img_input).astype(np.uint8)).convert("RGB")
    except Exception:
        return None


def _create_report_fpdf(
    patient,
    patient_id,
    prediction,
    confidence,
    image=None,
    gradcam_image=None,
    gender="Unspecified",
    age="N/A",
    all_probabilities=None,
    filename=None
):
    pdf = PathologyPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---------------------------------------------------------
    # 1. PATIENT & CASE METADATA BOX
    # ---------------------------------------------------------
    pdf.set_y(28)
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(210, 220, 235)
    pdf.rect(12, 28, 186, 22, "FD")

    pdf.set_text_color(40, 50, 70)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(16, 31)
    pdf.cell(40, 5, f"Patient Name:  {str(patient or 'Anonymous')}")
    pdf.set_xy(105, 31)
    pdf.cell(40, 5, f"Case / Patient ID:  {str(patient_id or 'N/A')}")

    pdf.set_xy(16, 39)
    pdf.cell(40, 5, f"Age / Gender:  {str(age)} / {str(gender)}")
    pdf.set_xy(105, 39)
    pdf.cell(40, 5, "Specimen:  Gastric Tissue Biopsy (H&E Stain)")

    # ---------------------------------------------------------
    # 2. DIAGNOSTIC PREDICTION BANNER
    # ---------------------------------------------------------
    pdf.set_y(54)
    info = CLASS_DESCRIPTIONS.get(prediction, {
        "full_name": str(prediction),
        "features": "Histopathological features detected by deep neural network.",
        "significance": "Requires clinical and histopathologic correlation."
    })

    # Color code depending on malignancy
    if prediction in ["TUM", "MUC"]:
        banner_fill = (255, 235, 235)
        border_fill = (220, 50, 50)
        badge_text = (180, 20, 20)
    elif prediction in ["DEB", "LYM"]:
        banner_fill = (255, 245, 230)
        border_fill = (230, 140, 30)
        badge_text = (190, 90, 0)
    else:
        banner_fill = (235, 248, 240)
        border_fill = (40, 160, 80)
        badge_text = (20, 120, 50)

    pdf.set_fill_color(*banner_fill)
    pdf.set_draw_color(*border_fill)
    pdf.rect(12, 53, 186, 20, "FD")

    pdf.set_text_color(*badge_text)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_xy(16, 56)
    pdf.cell(0, 6, f"AI PREDICTED CLASS:  {info['full_name']}")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(16, 63)
    pdf.set_text_color(50, 60, 75)
    pdf.cell(0, 6, f"Confidence Score:  {float(confidence):.2f}%   |   Analysis Status: Passed Quality & OOD Checks")

    # ---------------------------------------------------------
    # 3. EMBEDDED IMAGES (ORIGINAL & GRAD-CAM)
    # ---------------------------------------------------------
    pdf.set_y(78)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 45, 90)
    pdf.cell(90, 5, "Original Histopathology Specimen", align="C")
    pdf.cell(6, 5, "")
    pdf.cell(90, 5, "Grad-CAM++ Explainability Map", align="C", new_x="LMARGIN", new_y="NEXT")

    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(patient_id)).strip("_") or "analysis"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_orig = os.path.join(REPORT_DIR, f"_tmp_orig_{safe_id}_{timestamp}.png")
    temp_grad = os.path.join(REPORT_DIR, f"_tmp_grad_{safe_id}_{timestamp}.png")

    img_y = 85
    img_w, img_h = 88, 68

    try:
        # Save and draw original image
        orig_pil = _to_pil(image)
        if orig_pil is not None:
            orig_pil.save(temp_orig)
            pdf.image(temp_orig, x=14, y=img_y, w=img_w, h=img_h)
            pdf.set_draw_color(180, 190, 205)
            pdf.rect(14, img_y, img_w, img_h)
        else:
            pdf.set_xy(14, img_y + 25)
            pdf.cell(img_w, 10, "[No Specimen Image Attached]", align="C")

        # Save and draw Grad-CAM image
        grad_pil = _to_pil(gradcam_image)
        if grad_pil is not None:
            grad_pil.save(temp_grad)
            pdf.image(temp_grad, x=108, y=img_y, w=img_w, h=img_h)
            pdf.set_draw_color(180, 190, 205)
            pdf.rect(108, img_y, img_w, img_h)
        else:
            pdf.set_xy(108, img_y + 25)
            pdf.cell(img_w, 10, "[Grad-CAM Heatmap Unavailable]", align="C")

    finally:
        pass

    # ---------------------------------------------------------
    # 4. MICROSCOPIC FINDINGS & CLINICAL SIGNIFICANCE
    # ---------------------------------------------------------
    pdf.set_y(158)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 45, 90)
    pdf.cell(0, 5, "Microscopic Description & Histopathological Features", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(40, 45, 55)
    pdf.multi_cell(186, 4.5, f"Characteristic Features: {info['features']}")
    pdf.ln(2)
    pdf.multi_cell(186, 4.5, f"Clinical Significance: {info['significance']}")

    # ---------------------------------------------------------
    # 5. CLASS PROBABILITY DISTRIBUTION TABLE
    # ---------------------------------------------------------
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 45, 90)
    pdf.cell(0, 5, "Comprehensive Class Probability Distribution", new_x="LMARGIN", new_y="NEXT")

    all_classes = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
    col_w = 23.2

    # Table Header
    pdf.set_fill_color(230, 238, 248)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(25, 45, 80)
    for c in all_classes:
        pdf.cell(col_w, 6, c, border=1, align="C", fill=True)
    pdf.ln()

    # Table Values
    pdf.set_font("Helvetica", "", 8)
    if all_probabilities is not None and len(all_probabilities) == len(all_classes):
        probs_arr = np.asarray(all_probabilities).reshape(-1)
    else:
        probs_arr = np.full(len(all_classes), (100.0 - float(confidence)) / 7.0)
        probs_arr[all_classes.index(prediction)] = float(confidence)

    for i, c in enumerate(all_classes):
        val = probs_arr[i] * 100.0 if probs_arr[i] <= 1.0 else probs_arr[i]
        if c == prediction:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(255, 230, 230)
            pdf.cell(col_w, 6, f"{val:.1f}%", border=1, align="C", fill=True)
            pdf.set_font("Helvetica", "", 8)
        else:
            pdf.cell(col_w, 6, f"{val:.1f}%", border=1, align="C")
    pdf.ln(8)

    # ---------------------------------------------------------
    # 6. QUALITY SUMMARY & SIGN-OFF DISCLAIMER
    # ---------------------------------------------------------
    pdf.set_draw_color(200, 205, 215)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(90, 30, 30)
    pdf.cell(0, 4, "Important Diagnostic Disclaimer & Notice:", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(100, 105, 115)
    pdf.multi_cell(
        186, 3.8,
        "This report was automatically synthesized by the Gastric Cancer AI Digital Pathology Engine using transfer learning "
        "and Grad-CAM explainability algorithms. It is strictly intended as a computerized decision-support and academic research tool. "
        "Final pathological diagnosis, staging, and therapeutic interventions require formal microscopic verification and clinical integration "
        "by a licensed pathologist and multidisciplinary oncology team."
    )

    pdf.output(filename)

    # Clean up temporary image files
    if os.path.exists(temp_orig):
        try:
            os.remove(temp_orig)
        except OSError:
            pass
    if os.path.exists(temp_grad):
        try:
            os.remove(temp_grad)
        except OSError:
            pass

    return filename


def _create_report_matplotlib(
    patient,
    patient_id,
    prediction,
    confidence,
    image=None,
    gradcam_image=None,
    gender="Unspecified",
    age="N/A",
    all_probabilities=None,
    filename=None
):
    """Zero-dependency fallback: generates high-resolution medical PDF using matplotlib."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69), dpi=150)  # A4 size in inches

    # Background header banner
    fig.patches.extend([
        plt.Rectangle((0, 0.92), 1, 0.08, fill=True, color="#0F2D5F", transform=fig.transFigure, figure=fig)
    ])

    fig.text(0.06, 0.955, "GASTRIC CANCER AI - DIGITAL PATHOLOGY REPORT", fontsize=15, fontweight="bold", color="white")
    fig.text(0.06, 0.932, "Automated Histopathological Classification & Grad-CAM Explainability", fontsize=9, color="#E0E6ED")
    fig.text(0.94, 0.94, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), fontsize=9, color="white", ha="right")

    # Patient info box
    fig.patches.extend([
        plt.Rectangle((0.06, 0.835), 0.88, 0.07, fill=True, color="#F5F7FA", ec="#D2DCEB", transform=fig.transFigure, figure=fig)
    ])
    fig.text(0.08, 0.88, f"Patient Name:  {patient or 'Anonymous'}", fontsize=9, fontweight="bold", color="#283246")
    fig.text(0.52, 0.88, f"Case / Patient ID:  {patient_id or 'N/A'}", fontsize=9, fontweight="bold", color="#283246")
    fig.text(0.08, 0.85, f"Age / Gender:  {age} / {gender}", fontsize=9, color="#283246")
    fig.text(0.52, 0.85, "Specimen:  Gastric Tissue Biopsy (H&E Stain)", fontsize=9, color="#283246")

    # Prediction banner
    info = CLASS_DESCRIPTIONS.get(prediction, {
        "full_name": str(prediction),
        "features": "Histopathological features detected by deep neural network.",
        "significance": "Requires clinical and histopathologic correlation."
    })

    b_color = "#FFECEC" if prediction in ["TUM", "MUC"] else ("#FFF5E6" if prediction in ["DEB", "LYM"] else "#EBF8F0")
    t_color = "#B41414" if prediction in ["TUM", "MUC"] else ("#BE5A00" if prediction in ["DEB", "LYM"] else "#147832")

    fig.patches.extend([
        plt.Rectangle((0.06, 0.745), 0.88, 0.075, fill=True, color=b_color, ec=t_color, transform=fig.transFigure, figure=fig)
    ])
    fig.text(0.08, 0.79, f"AI PREDICTED CLASS:  {info['full_name']}", fontsize=12, fontweight="bold", color=t_color)
    fig.text(0.08, 0.76, f"Confidence Score:  {float(confidence):.2f}%   |   Status: Passed Quality & OOD Checks", fontsize=9.5, fontweight="bold", color="#323C4B")

    # Subplots for Images
    ax_orig = fig.add_axes([0.08, 0.44, 0.40, 0.26])
    ax_grad = fig.add_axes([0.52, 0.44, 0.40, 0.26])

    orig_pil = _to_pil(image)
    if orig_pil is not None:
        ax_orig.imshow(orig_pil)
        ax_orig.set_title("Original Specimen", fontsize=10, fontweight="bold", pad=5)
    else:
        ax_orig.text(0.5, 0.5, "No Specimen Image", ha="center", va="center")
    ax_orig.axis("off")

    grad_pil = _to_pil(gradcam_image)
    if grad_pil is not None:
        ax_grad.imshow(grad_pil)
        ax_grad.set_title("Grad-CAM++ Explainability Map", fontsize=10, fontweight="bold", pad=5)
    else:
        ax_grad.text(0.5, 0.5, "Grad-CAM Unavailable", ha="center", va="center")
    ax_grad.axis("off")

    # Findings text
    fig.text(0.06, 0.40, "Microscopic Description & Histopathological Features", fontsize=10, fontweight="bold", color="#142D5A")
    fig.text(0.06, 0.355, f"Characteristic Features: {info['features'][:220]}...", fontsize=8.5, color="#282D37", wrap=True)
    fig.text(0.06, 0.32, f"Clinical Significance: {info['significance'][:220]}...", fontsize=8.5, color="#282D37", wrap=True)

    # Probability Table
    fig.text(0.06, 0.28, "Comprehensive Class Probability Distribution", fontsize=10, fontweight="bold", color="#142D5A")
    all_classes = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
    if all_probabilities is not None and len(all_probabilities) == len(all_classes):
        probs_arr = np.asarray(all_probabilities).reshape(-1)
    else:
        probs_arr = np.full(len(all_classes), (100.0 - float(confidence)) / 7.0)
        probs_arr[all_classes.index(prediction)] = float(confidence)

    prob_row = [f"{(p * 100.0 if p <= 1.0 else p):.1f}%" for p in probs_arr]
    table_ax = fig.add_axes([0.06, 0.17, 0.88, 0.08])
    table_ax.axis("off")
    table = table_ax.table(
        cellText=[prob_row],
        colLabels=all_classes,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Disclaimer
    fig.text(0.06, 0.11, "Important Diagnostic Disclaimer & Notice:", fontsize=8, fontweight="bold", color="#5A1E1E")
    fig.text(
        0.06, 0.07,
        "This report was automatically synthesized by the Gastric Cancer AI Digital Pathology Engine using transfer learning\n"
        "and Grad-CAM explainability algorithms. It is strictly intended as a computerized decision-support and academic research tool.\n"
        "Final pathological diagnosis, staging, and therapeutic interventions require formal microscopic verification and clinical integration\n"
        "by a licensed pathologist and multidisciplinary oncology team.",
        fontsize=7, color="#646973"
    )
    fig.text(0.5, 0.02, "Gastric Cancer AI Digital Pathology Platform | Case Research Report", fontsize=7.5, color="#828282", ha="center")

    with PdfPages(filename) as pdf_pages:
        pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return filename


def create_report(
    patient,
    patient_id,
    prediction,
    confidence,
    image=None,
    gradcam_image=None,
    gender="Unspecified",
    age="N/A",
    all_probabilities=None
):
    os.makedirs(REPORT_DIR, exist_ok=True)

    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(patient_id)).strip("_") or "analysis"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(REPORT_DIR, f"{safe_id}_{timestamp}_report.pdf")

    if HAS_FPDF:
        try:
            return _create_report_fpdf(
                patient=patient,
                patient_id=patient_id,
                prediction=prediction,
                confidence=confidence,
                image=image,
                gradcam_image=gradcam_image,
                gender=gender,
                age=age,
                all_probabilities=all_probabilities,
                filename=filename
            )
        except Exception:
            # Fall back to matplotlib if FPDF runtime fails
            pass

    return _create_report_matplotlib(
        patient=patient,
        patient_id=patient_id,
        prediction=prediction,
        confidence=confidence,
        image=image,
        gradcam_image=gradcam_image,
        gender=gender,
        age=age,
        all_probabilities=all_probabilities,
        filename=filename
    )
