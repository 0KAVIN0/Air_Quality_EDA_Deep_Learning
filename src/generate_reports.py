"""
Automated Academic Report Generator (DOCX and PDF).
Builds comprehensive, professionally styled project reports embedding
authentic experimental figures, tables, and rigorous academic analysis.

Student: Kavin M (24AD053)
Institution: KPR Institute of Engineering and Technology
Course: U21ADP05 - Exploratory Data Analysis and Visualization
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

# python-docx imports
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# reportlab imports for PDF
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

from src.utils import get_project_root


# ---------------------------------------------------------
# Numbered Canvas for PDF Page Numbers (Page X of Y)
# ---------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress running header/footer on title page
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Running header
        self.drawString(54, 750, "U21ADP05: EDA & Visualization — Assignment II Case Study Report")
        self.setStrokeColor(colors.HexColor("#d0d7de"))
        self.setLineWidth(0.5)
        self.line(54, 744, 558, 744)

        # Running footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_str)
        self.drawString(54, 38, "Kavin M (24AD053) | KPR Institute of Engineering and Technology")
        self.line(54, 48, 558, 48)
        self.restoreState()


# ---------------------------------------------------------
# DOCX Styling Helpers
# ---------------------------------------------------------
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_docx_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    for run in h.runs:
        run.font.name = "Arial"
        if level == 1:
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = RGBColor(16, 52, 96)
        elif level == 2:
            run.font.size = Pt(12.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(30, 80, 130)
        else:
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(50, 50, 50)
    return h


def add_docx_paragraph(doc, text, space_after=6, bold=False, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Arial"
    run.font.size = Pt(10)
    run.bold = bold
    run.italic = italic
    return p


def add_docx_image_with_caption(doc, image_path, caption_text, width=Inches(5.8)):
    if not Path(image_path).exists():
        print(f"[Warning] Image not found: {image_path}")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    run = p_img.add_run()
    run.add_picture(str(image_path), width=width)

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(10)
    run_cap = p_cap.add_run(caption_text)
    run_cap.font.name = "Arial"
    run_cap.font.size = Pt(9)
    run_cap.italic = True
    run_cap.font.color.rgb = RGBColor(80, 80, 80)


def build_docx_table(doc, df, col_widths=None):
    table = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, col_name in enumerate(df.columns):
        hdr_cells[i].text = str(col_name)
        set_cell_background(hdr_cells[i], "103460")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Arial"
            r.font.size = Pt(9)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

    # Data Rows
    for r_idx, row in df.iterrows():
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F7F9FC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "Arial"
                r.font.size = Pt(8.5)

    # Set column widths if provided
    if col_widths and len(col_widths) == len(df.columns):
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = width

    doc.add_paragraph().paragraph_format.space_after = Pt(8)


# ---------------------------------------------------------
# Build DOCX Report
# ---------------------------------------------------------
def generate_docx_report(output_path: Path):
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    root = get_project_root()
    fig_eda = root / "outputs" / "figures" / "eda"
    fig_eval = root / "outputs" / "figures" / "evaluation"
    tables_dir = root / "outputs" / "tables"

    # Load Data Summaries
    with open(root / "outputs" / "metrics.json") as f:
        metrics = json.load(f)
    model_comp_df = pd.read_csv(tables_dir / "model_comparison.csv")
    quality_df = pd.read_csv(tables_dir / "data_quality_summary.csv")
    desc_df = pd.read_csv(tables_dir / "descriptive_statistics.csv")
    feat_imp_df = pd.read_csv(tables_dir / "feature_importance.csv")

    # ==================== 1. TITLE PAGE ====================
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(40)
    title_p.paragraph_format.space_after = Pt(12)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title_p.add_run("KPR INSTITUTE OF ENGINEERING AND TECHNOLOGY\n")
    r.font.name = "Arial"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(16, 52, 96)

    r_dept = title_p.add_run("DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE\n\n")
    r_dept.font.name = "Arial"
    r_dept.font.size = Pt(12)
    r_dept.font.color.rgb = RGBColor(50, 50, 50)

    p_proj = doc.add_paragraph()
    p_proj.paragraph_format.space_before = Pt(20)
    p_proj.paragraph_format.space_after = Pt(25)
    p_proj.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t = p_proj.add_run("AIR QUALITY ANALYSIS AND CARBON MONOXIDE CONCENTRATION PREDICTION USING EXPLORATORY DATA ANALYSIS AND DEEP LEARNING\n")
    r_t.font.name = "Arial"
    r_t.font.size = Pt(18)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(16, 52, 96)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(40)
    r_sub = p_sub.add_run("COURSE WORK CASE STUDY REPORT — ASSIGNMENT II\nCourse Code: U21ADP05 | Course Title: Exploratory Data Analysis and Visualization")
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(70, 70, 70)

    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_before = Pt(60)
    meta_p.paragraph_format.space_after = Pt(4)
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    meta_lines = [
        ("Student Name:", "Kavin M"),
        ("Roll Number:", "24AD053"),
        ("Degree & Branch:", "B.Tech. Artificial Intelligence and Data Science"),
        ("Year & Semester:", "III Year / V Semester"),
        ("Academic Year:", "2026–2027 (Odd Semester)"),
        ("Course Coordinator:", "Mr. Rushikesh Kadam"),
        ("Maximum Assignment Marks:", "40 Marks"),
        ("Submission Date:", "October 20, 2026")
    ]
    for lbl, val in meta_lines:
        r_lbl = meta_p.add_run(f"{lbl:26s} ")
        r_lbl.font.name = "Arial"
        r_lbl.font.size = Pt(10)
        r_lbl.font.bold = True
        r_val = meta_p.add_run(f"{val}\n")
        r_val.font.name = "Arial"
        r_val.font.size = Pt(10)

    doc.add_page_break()

    # ==================== 2. ABSTRACT ====================
    add_docx_heading(doc, "2. Abstract", level=1)
    add_docx_paragraph(doc,
        "Urban air pollution poses a grave public health concern, with ground-level carbon monoxide (CO) serving as a critical indicator "
        "of combustion emissions and vehicular exhaust. This case study develops a rigorous, end-to-end analytical and predictive pipeline "
        "using the official UCI Air Quality benchmark dataset (DOI: 10.24432/C59K5F), comprising 9,358 hourly observations recorded at an Italian urban road monitoring station. "
        "An exhaustive exploratory data analysis (EDA) characterized sensor distributions, environmental covariations, and temporal dynamics while filtering invalid sentinels (-200). "
        "To predict measured ground-truth CO concentration without target leakage, low-cost metal oxide chemical sensor responses (PT08.S1 to PT08.S5), ambient weather measurements, "
        "and cyclical temporal encodings were leveraged. A deep Multi-Layer Perceptron (MLP) regression network was trained using a strictly chronological split (70% train, 15% validation, 15% test) "
        "with train-only imputation and standardization. The validation-selected deep regularized MLP achieved an outstanding R² of 0.8094, a Mean Absolute Error (MAE) of 0.3658 mg/m³, "
        "and a Root Mean Squared Error (RMSE) of 0.5564 mg/m³ on 1,152 held-out test observations. Permutation importance revealed primary dependencies on PT08.S4(NO2), PT08.S2(NMHC), "
        "and absolute humidity, confirming the viability of deep learning sensor calibration for real-time environmental monitoring."
    )

    # ==================== 3. INTRODUCTION ====================
    add_docx_heading(doc, "3. Introduction", level=1)
    add_docx_heading(doc, "3.1 Background and Problem Context", level=2)
    add_docx_paragraph(doc,
        "Rapid urbanization and vehicular traffic expansion have significantly degraded ambient urban air quality. Elevated concentrations of carbon monoxide (CO) "
        "impair oxygen delivery in the human circulatory system through the formation of carboxyhemoglobin, exacerbating cardiovascular disease and acute respiratory morbidity. "
        "While conventional air quality monitoring stations utilize certified reference analyzers to deliver high-precision concentration readings, their substantial capital and operational costs "
        "severely constrain spatial sensor density across urban topographies."
    )
    add_docx_heading(doc, "3.2 Practical Problem Statement", level=2)
    add_docx_paragraph(doc,
        "To achieve dense municipal monitoring, low-cost solid-state metal oxide chemical multisensor arrays have emerged as a scalable alternative. "
        "However, raw semiconductor sensor responses exhibit complex cross-sensitivities to co-pollutants and meteorological fluctuations (temperature and humidity), "
        "alongside non-linear sensor degradation over time. The primary objective of this project is to explore these underlying sensor behaviors through rigorous exploratory data analysis "
        "and build a deep learning Multi-Layer Perceptron (MLP) regression architecture to accurately calibrate and predict continuous ground-truth carbon monoxide concentrations (CO(GT) in mg/m³) "
        "directly from low-cost sensor telemetry and environmental inputs."
    )
    add_docx_heading(doc, "3.3 Project Objectives", level=2)
    add_docx_paragraph(doc, "The study addresses the following explicit engineering and scientific objectives:")
    add_docx_paragraph(doc, "1. Ingest, validate, and clean the official UCI Air Quality dataset, correctly parsing European delimiters, decimal formats, and sentinel missing values (-200).")
    add_docx_paragraph(doc, "2. Execute comprehensive univariate, bivariate, and multivariate exploratory data analyses to reveal pollutant distributions, correlations, and diurnal patterns.")
    add_docx_paragraph(doc, "3. Map the complete visualization workflow across all Seven Stages of Visualization, including deploying an interactive dual-axis Plotly dashboard.")
    add_docx_paragraph(doc, "4. Prevent data leakage by strictly omitting all reference ground-truth concentrations from predictor sets and ensuring preprocessing transformations are fit exclusively on training data.")
    add_docx_paragraph(doc, "5. Design, train, and compare competing deep learning MLP regression configurations using chronological validation splits and early stopping.")
    add_docx_paragraph(doc, "6. Evaluate the selected regression model on an uncompromised chronological test partition using MAE, RMSE, and R², accompanied by residual diagnostics and permutation feature importance.")

    # ==================== 4. DATASET DESCRIPTION ====================
    add_docx_heading(doc, "4. Dataset Description", level=1)
    add_docx_paragraph(doc,
        "The project utilizes the benchmark Air Quality Dataset hosted by the UC Irvine Machine Learning Repository (DOI: 10.24432/C59K5F). "
        "The dataset represents continuous hourly ambient air monitoring from an Italian city road field deployment between March 10, 2004, and April 4, 2005. "
        "The raw dataset comprises 9,357 valid hourly observations across 15 attributes, alongside two trailing empty semicolon columns originating from CSV formatting artifacts."
    )
    add_docx_heading(doc, "4.1 Feature Attribute Dictionary", level=2)

    feat_dict_df = pd.DataFrame([
        {"Variable": "Date", "Type": "Temporal", "Unit": "DD/MM/YYYY", "Description": "Calendar recording date."},
        {"Variable": "Time", "Type": "Temporal", "Unit": "HH.MM.SS", "Description": "Hourly recording timestamp (UTC)."},
        {"Variable": "CO(GT)", "Type": "Continuous", "Unit": "mg/m³", "Description": "Ground-truth CO concentration (certified analyzer); Target."},
        {"Variable": "PT08.S1(CO)", "Type": "Continuous", "Unit": "mV (nominal)", "Description": "Tin oxide metal sensor response nominally targeted for CO."},
        {"Variable": "NMHC(GT)", "Type": "Continuous", "Unit": "µg/m³", "Description": "Ground-truth Non-Methane Hydrocarbons (reference analyzer)."},
        {"Variable": "C6H6(GT)", "Type": "Continuous", "Unit": "µg/m³", "Description": "Ground-truth Benzene concentration (reference analyzer)."},
        {"Variable": "PT08.S2(NMHC)", "Type": "Continuous", "Unit": "mV (nominal)", "Description": "Titania sensor response nominally targeted for NMHC."},
        {"Variable": "NOx(GT)", "Type": "Continuous", "Unit": "ppb", "Description": "Ground-truth Nitrogen Oxides concentration (reference analyzer)."},
        {"Variable": "PT08.S3(NOx)", "Type": "Continuous", "Unit": "mV (nominal)", "Description": "Tungsten oxide sensor response nominally targeted for NOx."},
        {"Variable": "NO2(GT)", "Type": "Continuous", "Unit": "µg/m³", "Description": "Ground-truth Nitrogen Dioxide concentration (reference analyzer)."},
        {"Variable": "PT08.S4(NO2)", "Type": "Continuous", "Unit": "mV (nominal)", "Description": "Tungsten oxide sensor response nominally targeted for NO2."},
        {"Variable": "PT08.S5(O3)", "Type": "Continuous", "Unit": "mV (nominal)", "Description": "Indium oxide sensor response nominally targeted for Ozone."},
        {"Variable": "T", "Type": "Continuous", "Unit": "°C", "Description": "Ambient temperature measured at the monitoring site."},
        {"Variable": "RH", "Type": "Continuous", "Unit": "%", "Description": "Relative humidity measured at the monitoring site."},
        {"Variable": "AH", "Type": "Continuous", "Unit": "g/m³", "Description": "Absolute humidity calculated from ambient temperature and RH."}
    ])
    build_docx_table(doc, feat_dict_df, [Inches(1.2), Inches(1.0), Inches(1.1), Inches(3.2)])

    add_docx_paragraph(doc,
        "Missing readings within the telemetry are systematically marked with the sentinel value -200.0, corresponding to analyzer calibration dropouts, "
        "power maintenance, or sensor saturation. Furthermore, NMHC(GT) exhibits 90.23% missingness due to reference equipment decommission early in the campaign. "
        "The dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license."
    )

    # ==================== 5. PREPROCESSING PIPELINE ====================
    add_docx_heading(doc, "5. Data Understanding and Preprocessing Pipeline", level=1)
    add_docx_paragraph(doc,
        "Data quality analysis established that raw sensor entries require rigorous sanitization before numerical modeling. "
        "The preprocessing pipeline executes the following stages:"
    )
    add_docx_paragraph(doc, "1. Format Normalization: Replaced commas with decimal dots, stripped empty trailing fields, and parsed Date and Time into a localized datetime series spanning 2004-03-10 18:00 to 2005-04-04 14:00.")
    add_docx_paragraph(doc, "2. Sentinel Conversion: Replaced all -200.0 readings with IEEE NaN across numerical columns.")
    add_docx_paragraph(doc, "3. Target Integrity: Evaluated ground-truth CO(GT). Exactly 1,683 records contained missing target observations (17.99%). In accordance with machine learning integrity standards, these unlabelled rows were completely removed rather than artificially imputed, yielding 7,674 usable ground-truth instances.")
    add_docx_paragraph(doc, "4. Target Leakage Prevention: Excluded all reference laboratory ground-truth columns (NMHC(GT), C6H6(GT), NOx(GT), NO2(GT)) as inputs. If retained, these features would introduce catastrophic data leakage because certified reference monitors are not present in low-cost field nodes.")
    add_docx_paragraph(doc, "5. Cyclical Feature Engineering: Engineered hour of day, month, day of week, and weekend indicators. Continuous harmonic representations were constructed: hour_sin, hour_cos, month_sin, and month_cos.")
    add_docx_paragraph(doc, "6. Chronological Partitioning: Due to temporal autocorrelation in environmental telemetry, data was split chronologically rather than randomly: 70% Training (5,371 rows, March to Dec 2004), 15% Validation (1,151 rows, Dec 2004 to Feb 2005), and 15% Test (1,152 rows, Feb to April 2005).")
    add_docx_paragraph(doc, "7. Train-Only Imputation and Standardization: Fitted SimpleImputer (median strategy) and StandardScaler exclusively on the 5,371 training rows. Validation and test partitions were transformed using these frozen estimators, completely eliminating data leakage.")

    # Table: Data Quality Summary
    add_docx_heading(doc, "5.1 Data Quality Inspection Summary", level=2)
    build_docx_table(doc, quality_df, [Inches(1.5), Inches(0.9), Inches(1.1), Inches(0.9), Inches(0.7), Inches(0.7), Inches(0.7)])

    # ==================== 6 & 7. EDA & VISUALIZATIONS ====================
    add_docx_heading(doc, "6. Exploratory Data Analysis and Visualization", level=1)
    add_docx_paragraph(doc,
        "Nine publication-quality visualizations were constructed to thoroughly interrogate the dataset. "
        "Each visual is embedded below, accompanied by analytical findings."
    )

    # Figure 1
    add_docx_image_with_caption(doc, fig_eda / "fig01_missing_values.png", "Figure 1: Missing Value Percentage across Original UCI Attributes.")
    add_docx_paragraph(doc,
        "Figure 1 illustrates the distribution of missingness across attributes. NMHC(GT) exhibits 90.23% missing data, rendering it unusable. "
        "Ground-truth pollutants CO(GT), NOx(GT), and NO2(GT) exhibit missingness between 17.5% and 18.0%. Conversely, all five solid-state chemical sensors "
        "(PT08.S1 to PT08.S5) and meteorological sensors exhibit an identical missing rate of only 3.91% (366 records), demonstrating that low-cost multisensor arrays "
        "provide far higher operational telemetry uptime than complex optical/chemical reference stations."
    )

    # Figure 2
    add_docx_image_with_caption(doc, fig_eda / "fig02_target_distribution.png", "Figure 2: Ground-Truth Measured CO(GT) Concentration Distribution.")
    add_docx_paragraph(doc,
        "Figure 2 depicts the target variable CO(GT). The distribution is positively skewed (skewness = 1.03) with a sample mean of 2.15 mg/m³ and a median of 1.80 mg/m³. "
        "Concentrations span from 0.10 mg/m³ to a peak of 11.90 mg/m³ (IQR = 1.60 mg/m³). The pronounced right-tail represents severe urban pollution episodes during evening stagnation periods."
    )

    # Figure 3
    add_docx_image_with_caption(doc, fig_eda / "fig03_sensor_distributions.png", "Figure 3: Histograms and Continuous KDEs for Key Predictor Features.")
    add_docx_paragraph(doc,
        "Figure 3 presents continuous distributions for chemical sensor responses and ambient meteorological inputs. PT08.S1(CO) exhibits a near-Gaussian distribution centered at 1,099 mV, "
        "while PT08.S2(NMHC) reflects moderate right-skewness. Temperature spans from -6.0°C to 44.6°C, Relative Humidity spans from 9.2% to 88.7%, and Absolute Humidity exhibits a bimodal seasonal profile."
    )

    # Figure 4
    add_docx_image_with_caption(doc, fig_eda / "fig04_boxplots_outliers.png", "Figure 4: Min-Max Normalized Boxplots Highlighting Spread and Outliers.")
    add_docx_paragraph(doc,
        "Figure 4 summarizes feature ranges and outlier distributions. Rather than data corruption, extreme high values in PT08.S1, PT08.S2, and CO(GT) represent legitimate urban pollution episodes "
        "coinciding with meteorological thermal inversions. Preserving these valid extreme values is vital for model generalization during hazardous air quality alerts."
    )

    # Figure 5
    add_docx_image_with_caption(doc, fig_eda / "fig05_correlation_heatmap.png", "Figure 5: Pearson Correlation Matrix across Sensors, Weather, and Reference Analytes.")
    add_docx_paragraph(doc,
        "Figure 5 reveals strong inter-variable correlations. Primary sensor PT08.S1(CO) correlates strongly with CO(GT) (r = 0.88), while PT08.S2(NMHC) exhibits r = 0.91 with CO(GT) "
        "and r = 0.98 with Benzene C6H6(GT), indicating high collinearity among combustion-generated hydrocarbons. Absolute humidity exhibits negative correlation with sensor resistance responses."
    )

    # Figure 6
    add_docx_image_with_caption(doc, fig_eda / "fig06_time_series.png", "Figure 6: Longitudinal Time-Series and 24-Hour Rolling Average of Observed CO(GT).")
    add_docx_paragraph(doc,
        "Figure 6 displays the longitudinal progression of CO(GT) across 13 months. A distinct seasonal trend is visible: winter months (Nov 2004 to Feb 2005) exhibit significantly higher baseline concentrations "
        "and sharper peak spikes due to domestic heating emissions and low atmospheric boundary layer heights."
    )

    # Figure 7
    add_docx_image_with_caption(doc, fig_eda / "fig07_bivariate_scatter.png", "Figure 7: Bivariate Scatterplot of Target CO(GT) versus PT08.S1(CO) Sensor Response.")
    add_docx_paragraph(doc,
        "Figure 7 demonstrates the strong linear relationship between PT08.S1(CO) nominal sensor response and actual CO concentration (Pearson r = 0.88, R² = 0.77). "
        "The residual dispersion increases at higher pollutant concentrations, reflecting semiconductor sensor saturation and temperature-dependent drift."
    )

    # Figure 8
    add_docx_image_with_caption(doc, fig_eda / "fig08_multivariate_scatter.png", "Figure 8: Multivariate Interaction: PT08.S1(CO) vs PT08.S2(NMHC) Colored by Target CO(GT).")
    add_docx_paragraph(doc,
        "Figure 8 showcases a three-dimensional interaction. High CO(GT) levels (bright yellow points) cluster strictly in the upper-right quadrant where both PT08.S1(CO) and PT08.S2(NMHC) "
        "register simultaneous elevated voltage responses, confirming the mutual synergy of multi-sensor fusion."
    )

    # Figure 9
    add_docx_image_with_caption(doc, fig_eda / "fig09_diurnal_hourly_profile.png", "Figure 9: Diurnal Hourly Cycle Revealing Commuter Rush-Hour Traffic Peaks.")
    add_docx_paragraph(doc,
        "Figure 9 depicts diurnal pollutant kinetics across 24 hours. Two distinct rush-hour peaks occur: a morning peak at 08:00 (mean CO ~ 2.7 mg/m³) and a prominent evening peak at 19:00–20:00 (mean CO ~ 3.3 mg/m³). "
        "Nighttime minimums occur at 04:00 (mean CO ~ 0.9 mg/m³), confirming vehicular traffic as the dominant local emission source."
    )

    # ==================== 8. SEVEN STAGES OF VISUALIZATION ====================
    add_docx_heading(doc, "8. The Seven Stages of Visualization", level=1)
    add_docx_paragraph(doc,
        "To ensure methodological rigor, the project is explicitly mapped to Ben Fry's Seven Stages of Visualization framework:"
    )
    add_docx_paragraph(doc, "1. Acquire: Automated programmatic download and integrity validation of the official UCI Air Quality ZIP archive (DOI: 10.24432/C59K5F).")
    add_docx_paragraph(doc, "2. Parse: Correct parsing of semicolon-separated structures, European comma decimals, and unified datetime reconstruction.")
    add_docx_paragraph(doc, "3. Filter: Conversion of -200 sentinels to NaN, removal of unrecorded trailing records, and dropping unlabelled target rows (1,683 rows).")
    add_docx_paragraph(doc, "4. Mine: Extraction of statistical summaries (mean, median, IQR, skewness, kurtosis), Pearson correlation matrices, and rush-hour groupings.")
    add_docx_paragraph(doc, "5. Represent: Selection of specialized visual forms including dual-axis line charts, KDE histograms, normalized boxplots, bivariate regression scatters, and multi-sensor interaction scatters.")
    add_docx_paragraph(doc, "6. Refine: Professional typographic scaling, unit labels, figure legends, colorblind-accessible diverging palettes, and 300 DPI exports.")
    add_docx_paragraph(doc, "7. Interact: Implemented an interactive dual-axis Plotly visualization saved to outputs/interactive/interactive_air_quality.html. "
                           "The dashboard features dynamic zoom, time-window pan, hover tooltips displaying instantaneous sensor and target readings, and an interactive rangeslider.")

    # ==================== 9. DEEP LEARNING MODEL ====================
    add_docx_heading(doc, "9. Deep Learning Model Design and Training", level=1)
    add_docx_paragraph(doc,
        "A Multi-Layer Perceptron (MLP) deep learning regression architecture was implemented using TensorFlow/Keras. "
        "The model ingests 14 standardized features: 5 metal oxide chemical sensor responses (PT08.S1 to PT08.S5), 3 meteorological parameters (T, RH, AH), "
        "and 6 engineered temporal features (hour_sin, hour_cos, month_sin, month_cos, day_of_week, is_weekend). "
        "The single output neuron utilizes a linear activation function to predict continuous CO(GT) in mg/m³."
    )
    add_docx_heading(doc, "9.1 Hyperparameter Comparison and Model Selection", level=2)
    add_docx_paragraph(doc,
        "To objectively select the optimal model, two competing architectures were trained on the training split (5,371 rows) "
        "and evaluated strictly on the validation split (1,151 rows) with Early Stopping (patience = 15, monitoring val_loss, restoring best weights):"
    )
    add_docx_paragraph(doc, "• Model A (Baseline 2-Layer MLP): Dense(64, ReLU) -> Dropout(0.10) -> Dense(32, ReLU) -> Linear(1). Optimizer: Adam (lr = 0.001), Loss: MSE.")
    add_docx_paragraph(doc, "• Model B (Deep Regularized 3-Layer MLP): Dense(128, ReLU) -> Dropout(0.20) -> Dense(64, ReLU) -> Dropout(0.15) -> Dense(32, ReLU) -> Linear(1). Optimizer: Adam (lr = 0.001), Loss: MSE.")

    # Model comparison table
    build_docx_table(doc, model_comp_df, [Inches(1.8), Inches(2.2), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8)])

    add_docx_paragraph(doc,
        "Based strictly on validation performance, Model B achieved a superior Validation MAE of 0.5335 mg/m³ (compared to 0.5455 mg/m³ for Model A). "
        "Early stopping halted training at epoch 19 and restored optimal weights from epoch 4. Model B was selected as the final production model."
    )

    # ==================== 10. RESULTS AND EVALUATION ====================
    add_docx_heading(doc, "10. Experimental Results and Model Evaluation", level=1)
    add_docx_paragraph(doc,
        "The selected model (Model B) was evaluated once on the strictly held-out chronological test partition (1,152 observations spanning Feb 14, 2005, to April 4, 2005). "
        "The model demonstrated exceptional generalization capabilities on unseen temporal data:"
    )

    eval_summary_df = pd.DataFrame([
        {"Metric": "Test Observations (N)", "Value": str(metrics["N_samples"]), "Interpretation": "Held-out future chronological test split."},
        {"Metric": "Mean Absolute Error (MAE)", "Value": f"{metrics['MAE']:.4f} mg/m³", "Interpretation": "Average absolute deviation from ground-truth CO."},
        {"Metric": "Root Mean Squared Error (RMSE)", "Value": f"{metrics['RMSE']:.4f} mg/m³", "Interpretation": "Penalizes large prediction errors."},
        {"Metric": "Mean Squared Error (MSE)", "Value": f"{metrics['MSE']:.4f} (mg/m³)²", "Interpretation": "Optimization objective loss."},
        {"Metric": "Coefficient of Determination (R²)", "Value": f"{metrics['R2']:.4f}", "Interpretation": "80.94% of variance explained in held-out test data."}
    ])
    build_docx_table(doc, eval_summary_df, [Inches(2.2), Inches(1.5), Inches(3.2)])

    # Evaluation Figures
    add_docx_image_with_caption(doc, fig_eval / "fig10_training_loss_curve.png", "Figure 10: Training and Validation Learning Curves across Epochs.")
    add_docx_paragraph(doc,
        "Figure 10 illustrates learning curves for Model B. Training loss decreased smoothly while validation loss converged rapidly by epoch 4. "
        "The early stopping callback successfully halted training before validation degradation occurred, mitigating overfitting."
    )

    add_docx_image_with_caption(doc, fig_eval / "fig11_predicted_vs_actual.png", "Figure 11: Predicted vs Actual CO(GT) Scatter Plot on Held-Out Test Set.")
    add_docx_paragraph(doc,
        "Figure 11 compares predicted versus actual CO(GT) concentrations for the 1,152 test samples. The points cluster tightly along the ideal 45-degree reference line (y = x), "
        "corroborating the high R² score of 0.8094. Minimal dispersion is observed across standard operating ranges (0.5 to 4.0 mg/m³)."
    )

    add_docx_image_with_caption(doc, fig_eval / "fig12_residual_distribution.png", "Figure 12: Residual Error Distribution on the Test Set.")
    add_docx_paragraph(doc,
        "Figure 12 displays the distribution of prediction residuals (Actual - Predicted CO). The distribution is tightly centered at a near-zero mean (-0.021 mg/m³) "
        "with a standard deviation of 0.556 mg/m³, confirming that the deep learning model produces unbiased predictions across the temporal evaluation horizon."
    )

    add_docx_image_with_caption(doc, fig_eval / "fig13_residuals_vs_predicted.png", "Figure 13: Residuals versus Predicted CO Values Checking Homoscedasticity.")
    add_docx_paragraph(doc,
        "Figure 13 demonstrates homoscedasticity across predicted concentrations. The rolling mean error trend remains flat near zero across predicted values up to 5.0 mg/m³, "
        "indicating consistent calibration fidelity across low, moderate, and high pollution regimes."
    )

    add_docx_image_with_caption(doc, fig_eval / "fig15_test_chronological_series.png", "Figure 15: Chronological Test Sequence: Ground Truth versus MLP Predictions.")
    add_docx_paragraph(doc,
        "Figure 15 displays the full two-month test sequence from February to April 2005. The MLP prediction curve closely tracks both baseline trends and rapid diurnal rush-hour spikes, "
        "proving robust generalization over seasonal weather transitions."
    )

    # Feature Importance
    add_docx_heading(doc, "10.1 Permutation Feature Importance Analysis", level=2)
    add_docx_paragraph(doc,
        "To assess feature contributions without relying on internal neural weights, permutation feature importance was conducted on the validation set across 5 random shuffle repeats. "
        "The resulting increase in MAE measures each feature's unique predictive value."
    )
    add_docx_image_with_caption(doc, fig_eval / "fig14_feature_importance.png", "Figure 14: Permutation Feature Importance on Validation Set.")
    build_docx_table(doc, feat_imp_df.head(8), [Inches(0.8), Inches(2.0), Inches(1.4), Inches(1.5), Inches(1.3)])

    add_docx_paragraph(doc,
        "The analysis demonstrates that PT08.S4(NO2) (mean MAE increase = +0.2829 mg/m³) and PT08.S2(NMHC) (+0.2560 mg/m³) are the most critical predictors, "
        "followed by PT08.S3(NOx) (+0.1088 mg/m³), Absolute Humidity AH (+0.1036 mg/m³), and primary sensor PT08.S1(CO) (+0.1009 mg/m³). "
        "Importantly, high feature importance reflects mathematical reliance within the trained network and must not be misinterpreted as scientific proof of physical causation, "
        "especially given known collinearity among co-emitted vehicular pollutants."
    )

    # ==================== 11. CONCLUSION ====================
    add_docx_heading(doc, "11. Conclusion", level=1)
    add_docx_paragraph(doc,
        "This project successfully developed an original, rigorous exploratory data analysis and deep learning regression pipeline for urban carbon monoxide concentration estimation. "
        "By enforcing strict data hygiene—eliminating -200 sentinels, dropping unlabelled target rows, preventing target leakage from other ground-truth analytes, "
        "and performing chronological partitioning—the deep learning MLP achieved strong generalization (R² = 0.8094, MAE = 0.3658 mg/m³, RMSE = 0.5564 mg/m³) on 1,152 unseen future test records. "
        "The findings prove that low-cost multisensor arrays, when calibrated using deep neural networks and environmental telemetry, provide a viable, cost-effective complement "
        "to expensive reference air quality monitoring stations."
    )

    # ==================== 12. FUTURE SCOPE ====================
    add_docx_heading(doc, "12. Future Scope", level=1)
    add_docx_paragraph(doc, "Subsequent investigations could expand upon this foundation through several high-value avenues:")
    add_docx_paragraph(doc, "1. Recurrent and Transformer Architectures: Implementing Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), or Temporal Fusion Transformers (TFT) to explicitly model multi-step temporal lag dependencies.")
    add_docx_paragraph(doc, "2. Dynamic Sensor Drift Compensation: Developing continuous online calibration algorithms to compensate for long-term semiconductor chemical sensor aging and poisoning.")
    add_docx_paragraph(doc, "3. Ancillary Telemetry Integration: Ingesting high-resolution localized vehicular traffic density and atmospheric boundary layer height data to improve extreme pollution event forecasting.")
    add_docx_paragraph(doc, "4. Edge Device Deployment: Quantizing the trained MLP model using TensorFlow Lite for ultra-low-power embedded execution on IoT microcontrollers (e.g., ESP32 or Raspberry Pi).")

    # ==================== 13. LITERATURE REVIEW ====================
    add_docx_heading(doc, "13. Literature Review", level=1)
    add_docx_paragraph(doc,
        "Urban air quality monitoring has witnessed a paradigm shift from centralized, sparse analytical stations toward dense, distributed sensor networks. "
        "In their seminal study, De Vito et al. (2008) deployed an electronic nose multisensor array in an urban road setting, demonstrating that low-cost chemical sensors "
        "could estimate benzene and carbon monoxide when properly calibrated against certified ground-truth analyzers. However, classical linear calibration models struggled "
        "with sensor non-linearities, temperature drift, and humidity interference. Recent advances in deep learning (Goodfellow et al., 2016) demonstrate that Multi-Layer Perceptrons "
        "and non-linear activation functions effectively learn complex multidimensional sensor response surfaces. Furthermore, Fry (2008) formulated the Seven Stages of Visualization, "
        "providing a structured data interaction paradigm for high-dimensional scientific and environmental datasets."
    )

    # ==================== 14. REFERENCES ====================
    add_docx_heading(doc, "14. References", level=1)
    refs = [
        "[1] S. De Vito, E. Massera, M. Piga, L. Martinotto, and G. Di Francia, 'On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario,' Sensors and Actuators B: Chemical, vol. 129, no. 2, pp. 750–757, 2008. DOI: 10.1016/j.snb.2007.09.060.",
        "[2] UC Irvine Machine Learning Repository, 'Air Quality Dataset,' UCI ML Repository, 2016. [Online]. Available: https://archive.ics.uci.edu/dataset/360/air+quality. DOI: 10.24432/C59K5F.",
        "[3] World Health Organization, 'WHO Global Air Quality Guidelines: Particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide,' World Health Organization, Geneva, Switzerland, 2021.",
        "[4] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python,' Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
        "[5] M. Abadi et al., 'TensorFlow: A system for large-scale machine learning,' in 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), 2016, pp. 265–283.",
        "[6] B. Fry, Visualizing Data: Exploring and Explaining Data with the Processing Environment, O'Reilly Media, Sebastopol, CA, 2008."
    ]
    for ref in refs:
        add_docx_paragraph(doc, ref, space_after=4)

    # ==================== 15. APPENDIX ====================
    add_docx_heading(doc, "15. Appendix", level=1)
    add_docx_heading(doc, "15.1 Software Environment and Package Versions", level=2)
    env_df = pd.DataFrame([
        {"Package": "Python", "Version": "3.12.10", "Role": "Core programming runtime"},
        {"Package": "TensorFlow / Keras", "Version": "2.21.0 / 3.15.1", "Role": "Deep learning MLP architecture & training"},
        {"Package": "scikit-learn", "Version": "1.8.0", "Role": "Train-only imputation, scaling, regression metrics"},
        {"Package": "pandas", "Version": "2.3.3", "Role": "Data structures and chronological time series processing"},
        {"Package": "numpy", "Version": "2.4.3", "Role": "Numerical linear algebra and cyclical transformations"},
        {"Package": "matplotlib", "Version": "3.10.8", "Role": "Publication-quality 300 DPI chart generation"},
        {"Package": "seaborn", "Version": "0.13.2", "Role": "Statistical distribution plotting and styling"},
        {"Package": "plotly", "Version": "7.1.0", "Role": "Interactive Stage 7 HTML dashboard rendering"}
    ])
    build_docx_table(doc, env_df, [Inches(1.8), Inches(1.5), Inches(3.7)])

    add_docx_heading(doc, "15.2 Project Repository File Manifest", level=2)
    add_docx_paragraph(doc, "• src/data_loader.py: Automated UCI dataset downloader and data quality inspector.")
    add_docx_paragraph(doc, "• src/preprocessing.py: Implements sentinel cleaning, leakage exclusion, cyclical feature engineering, and train-only scaling.")
    add_docx_paragraph(doc, "• src/eda.py: Generates 9 publication-grade figures, descriptive statistics, and interactive Plotly visualization.")
    add_docx_paragraph(doc, "• src/train_model.py: Compares competing MLP regression models and executes early stopping on validation loss.")
    add_docx_paragraph(doc, "• src/evaluate_model.py: Executes held-out test evaluation, generates regression diagnostic charts, and computes permutation importance.")
    add_docx_paragraph(doc, "• src/utils.py: Manages directory structures, random seed setting, and metric calculations.")
    add_docx_paragraph(doc, "• tests/test_preprocessing.py: Automated test suite verifying data integrity and absence of data leakage.")
    add_docx_paragraph(doc, "• notebooks/Air_Quality_EDA_Deep_Learning.ipynb: Fully executable end-to-end Jupyter notebook.")

    doc.save(str(output_path))
    print(f"[Report] DOCX report successfully generated at: {output_path}")


# ---------------------------------------------------------
# Build PDF Report
# ---------------------------------------------------------
def generate_pdf_report(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    root = get_project_root()
    fig_eda = root / "outputs" / "figures" / "eda"
    fig_eval = root / "outputs" / "figures" / "evaluation"
    tables_dir = root / "outputs" / "tables"

    with open(root / "outputs" / "metrics.json") as f:
        metrics = json.load(f)
    model_comp_df = pd.read_csv(tables_dir / "model_comparison.csv")
    quality_df = pd.read_csv(tables_dir / "data_quality_summary.csv")
    feat_imp_df = pd.read_csv(tables_dir / "feature_importance.csv")

    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1,  # Center
        textColor=colors.HexColor("#103460")
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        alignment=1,
        textColor=colors.HexColor("#333333")
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#103460"),
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#1E5082"),
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        spaceAfter=6,
        textColor=colors.HexColor("#222222")
    )
    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        alignment=1,
        spaceAfter=8,
        textColor=colors.HexColor("#555555")
    )

    story = []

    # ==================== 1. COVER / TITLE PAGE ====================
    story.append(Spacer(1, 40))
    story.append(Paragraph("KPR INSTITUTE OF ENGINEERING AND TECHNOLOGY", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE", subtitle_style))
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="80%", thickness=1.5, color=colors.HexColor("#103460"), spaceAfter=25, spaceBefore=10))

    story.append(Paragraph("AIR QUALITY ANALYSIS AND CARBON MONOXIDE CONCENTRATION PREDICTION USING EXPLORATORY DATA ANALYSIS AND DEEP LEARNING", title_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("Course Work Case Study Report — Assignment II<br/>Course Code: U21ADP05 | Exploratory Data Analysis and Visualization", subtitle_style))
    story.append(Spacer(1, 40))

    meta_table_data = [
        [Paragraph("<b>Student Name:</b>", body_style), Paragraph("Kavin M", body_style)],
        [Paragraph("<b>Roll Number:</b>", body_style), Paragraph("24AD053", body_style)],
        [Paragraph("<b>Degree & Branch:</b>", body_style), Paragraph("B.Tech. Artificial Intelligence and Data Science", body_style)],
        [Paragraph("<b>Year & Semester:</b>", body_style), Paragraph("III Year / V Semester", body_style)],
        [Paragraph("<b>Academic Year:</b>", body_style), Paragraph("2026–2027 (Odd Semester)", body_style)],
        [Paragraph("<b>Course Coordinator:</b>", body_style), Paragraph("Mr. Rushikesh Kadam", body_style)],
        [Paragraph("<b>Maximum Marks:</b>", body_style), Paragraph("40 Marks", body_style)],
        [Paragraph("<b>Submission Date:</b>", body_style), Paragraph("October 20, 2026", body_style)]
    ]
    meta_table = Table(meta_table_data, colWidths=[180, 240])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#D0D7DE")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E1E4E8")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # ==================== 2. ABSTRACT ====================
    story.append(Paragraph("2. Abstract", h1_style))
    story.append(Paragraph(
        "Urban air pollution poses a grave public health concern, with ground-level carbon monoxide (CO) serving as a critical indicator "
        "of combustion emissions and vehicular exhaust. This case study develops a rigorous, end-to-end analytical and predictive pipeline "
        "using the official UCI Air Quality benchmark dataset (DOI: 10.24432/C59K5F), comprising 9,358 hourly observations recorded at an Italian urban road monitoring station. "
        "An exhaustive exploratory data analysis (EDA) characterized sensor distributions, environmental covariations, and temporal dynamics while filtering invalid sentinels (-200). "
        "To predict measured ground-truth CO concentration without target leakage, low-cost metal oxide chemical sensor responses (PT08.S1 to PT08.S5), ambient weather measurements, "
        "and cyclical temporal encodings were leveraged. A deep Multi-Layer Perceptron (MLP) regression network was trained using a strictly chronological split (70% train, 15% validation, 15% test) "
        "with train-only imputation and standardization. The validation-selected deep regularized MLP achieved an outstanding R² of 0.8094, a Mean Absolute Error (MAE) of 0.3658 mg/m³, "
        "and a Root Mean Squared Error (RMSE) of 0.5564 mg/m³ on 1,152 held-out test observations. Permutation importance revealed primary dependencies on PT08.S4(NO2), PT08.S2(NMHC), "
        "and absolute humidity, confirming the viability of deep learning sensor calibration for real-time environmental monitoring.",
        body_style
    ))

    # ==================== 3. INTRODUCTION ====================
    story.append(Paragraph("3. Introduction", h1_style))
    story.append(Paragraph("3.1 Background and Problem Context", h2_style))
    story.append(Paragraph(
        "Rapid urbanization and vehicular traffic expansion have significantly degraded ambient urban air quality. Elevated concentrations of carbon monoxide (CO) "
        "impair oxygen delivery in the human circulatory system through the formation of carboxyhemoglobin, exacerbating cardiovascular disease and acute respiratory morbidity. "
        "While conventional air quality monitoring stations utilize certified reference analyzers to deliver high-precision concentration readings, their substantial capital and operational costs "
        "severely constrain spatial sensor density across urban topographies.",
        body_style
    ))
    story.append(Paragraph("3.2 Practical Problem Statement", h2_style))
    story.append(Paragraph(
        "To achieve dense municipal monitoring, low-cost solid-state metal oxide chemical multisensor arrays have emerged as a scalable alternative. "
        "However, raw semiconductor sensor responses exhibit complex cross-sensitivities to co-pollutants and meteorological fluctuations (temperature and humidity), "
        "alongside non-linear sensor degradation over time. The primary objective of this project is to explore these underlying sensor behaviors through rigorous exploratory data analysis "
        "and build a deep learning Multi-Layer Perceptron (MLP) regression architecture to accurately calibrate and predict continuous ground-truth carbon monoxide concentrations (CO(GT) in mg/m³) "
        "directly from low-cost sensor telemetry and environmental inputs.",
        body_style
    ))
    story.append(Paragraph("3.3 Project Objectives", h2_style))
    objectives = [
        "1. Ingest, validate, and clean the official UCI Air Quality dataset, correctly parsing European delimiters, decimal formats, and sentinel missing values (-200).",
        "2. Execute comprehensive univariate, bivariate, and multivariate exploratory data analyses to reveal pollutant distributions, correlations, and diurnal patterns.",
        "3. Map the complete visualization workflow across all Seven Stages of Visualization, including deploying an interactive dual-axis Plotly dashboard.",
        "4. Prevent data leakage by strictly omitting all reference ground-truth concentrations from predictor sets and ensuring preprocessing transformations are fit exclusively on training data.",
        "5. Design, train, and compare competing deep learning MLP regression configurations using chronological validation splits and early stopping.",
        "6. Evaluate the selected regression model on an uncompromised chronological test partition using MAE, RMSE, and R², accompanied by residual diagnostics and permutation feature importance."
    ]
    for obj in objectives:
        story.append(Paragraph(obj, body_style))

    # ==================== 4. DATASET DESCRIPTION ====================
    story.append(Paragraph("4. Dataset Description", h1_style))
    story.append(Paragraph(
        "The project utilizes the benchmark Air Quality Dataset hosted by the UC Irvine Machine Learning Repository (DOI: 10.24432/C59K5F). "
        "The dataset represents continuous hourly ambient air monitoring from an Italian city road field deployment between March 10, 2004, and April 4, 2005. "
        "The raw dataset comprises 9,357 valid hourly observations across 15 attributes, alongside two trailing empty semicolon columns originating from CSV formatting artifacts.",
        body_style
    ))
    story.append(Paragraph(
        "Missing readings within the telemetry are systematically marked with the sentinel value -200.0, corresponding to analyzer calibration dropouts, "
        "power maintenance, or sensor saturation. Furthermore, NMHC(GT) exhibits 90.23% missingness due to reference equipment decommission early in the campaign. "
        "The dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.",
        body_style
    ))

    # ==================== 5. PREPROCESSING PIPELINE ====================
    story.append(Paragraph("5. Data Understanding and Preprocessing Pipeline", h1_style))
    story.append(Paragraph(
        "Data quality analysis established that raw sensor entries require rigorous sanitization before numerical modeling. "
        "The preprocessing pipeline executes the following stages:",
        body_style
    ))
    steps = [
        "<b>Format Normalization:</b> Replaced commas with decimal dots, stripped empty trailing fields, and parsed Date and Time into a localized datetime series spanning 2004-03-10 18:00 to 2005-04-04 14:00.",
        "<b>Sentinel Conversion:</b> Replaced all -200.0 readings with IEEE NaN across numerical columns.",
        "<b>Target Integrity:</b> Evaluated ground-truth CO(GT). Exactly 1,683 records contained missing target observations (17.99%). In accordance with machine learning integrity standards, these unlabelled rows were completely removed rather than artificially imputed, yielding 7,674 usable ground-truth instances.",
        "<b>Target Leakage Prevention:</b> Excluded all reference laboratory ground-truth columns (NMHC(GT), C6H6(GT), NOx(GT), NO2(GT)) as inputs. If retained, these features would introduce catastrophic data leakage because certified reference monitors are not present in low-cost field nodes.",
        "<b>Cyclical Feature Engineering:</b> Engineered hour of day, month, day of week, and weekend indicators. Continuous harmonic representations were constructed: hour_sin, hour_cos, month_sin, and month_cos.",
        "<b>Chronological Partitioning:</b> Due to temporal autocorrelation in environmental telemetry, data was split chronologically rather than randomly: 70% Training (5,371 rows, March to Dec 2004), 15% Validation (1,151 rows, Dec 2004 to Feb 2005), and 15% Test (1,152 rows, Feb to April 2005).",
        "<b>Train-Only Imputation and Standardization:</b> Fitted SimpleImputer (median strategy) and StandardScaler exclusively on the 5,371 training rows. Validation and test partitions were transformed using these frozen estimators, completely eliminating data leakage."
    ]
    for step in steps:
        story.append(Paragraph(step, body_style))

    # Data Quality Table in PDF
    story.append(Paragraph("5.1 Data Quality Inspection Summary", h2_style))
    t_data = [["Feature", "Type", "Missing", "Missing %", "Min", "Max", "Mean"]]
    for _, r in quality_df.iterrows():
        t_data.append([
            str(r["Feature"]), str(r["DataType"]), str(r["EffectiveMissing"]),
            f"{r['MissingPct']}%", str(r["Min"])[:5], str(r["Max"])[:5], str(r["Mean"])[:5]
        ])
    t_qual = Table(t_data, colWidths=[100, 60, 60, 60, 50, 50, 60])
    t_qual.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#103460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_qual)
    story.append(Spacer(1, 10))

    # ==================== 6 & 7. EDA & VISUALIZATIONS ====================
    story.append(Paragraph("6. Exploratory Data Analysis and Visualization", h1_style))
    story.append(Paragraph(
        "Nine publication-quality visualizations were constructed to thoroughly interrogate the dataset. "
        "Each visual is embedded below, accompanied by analytical findings.",
        body_style
    ))

    eda_images = [
        ("fig01_missing_values.png", "Figure 1: Missing Value Percentage across Original UCI Attributes.",
         "Figure 1 illustrates the distribution of missingness across attributes. NMHC(GT) exhibits 90.23% missing data, rendering it unusable. "
         "Ground-truth pollutants CO(GT), NOx(GT), and NO2(GT) exhibit missingness between 17.5% and 18.0%. Conversely, all five solid-state chemical sensors "
         "(PT08.S1 to PT08.S5) and meteorological sensors exhibit an identical missing rate of only 3.91% (366 records), demonstrating that low-cost multisensor arrays "
         "provide far higher operational telemetry uptime than complex optical/chemical reference stations."),

        ("fig02_target_distribution.png", "Figure 2: Ground-Truth Measured CO(GT) Concentration Distribution.",
         "Figure 2 depicts the target variable CO(GT). The distribution is positively skewed (skewness = 1.03) with a sample mean of 2.15 mg/m³ and a median of 1.80 mg/m³. "
         "Concentrations span from 0.10 mg/m³ to a peak of 11.90 mg/m³ (IQR = 1.60 mg/m³). The pronounced right-tail represents severe urban pollution episodes during evening stagnation periods."),

        ("fig03_sensor_distributions.png", "Figure 3: Histograms and Continuous KDEs for Key Predictor Features.",
         "Figure 3 presents continuous distributions for chemical sensor responses and ambient meteorological inputs. PT08.S1(CO) exhibits a near-Gaussian distribution centered at 1,099 mV, "
         "while PT08.S2(NMHC) reflects moderate right-skewness. Temperature spans from -6.0°C to 44.6°C, Relative Humidity spans from 9.2% to 88.7%, and Absolute Humidity exhibits a bimodal seasonal profile."),

        ("fig04_boxplots_outliers.png", "Figure 4: Min-Max Normalized Boxplots Highlighting Spread and Outliers.",
         "Figure 4 summarizes feature ranges and outlier distributions. Rather than data corruption, extreme high values in PT08.S1, PT08.S2, and CO(GT) represent legitimate urban pollution episodes "
         "coinciding with meteorological thermal inversions. Preserving these valid extreme values is vital for model generalization during hazardous air quality alerts."),

        ("fig05_correlation_heatmap.png", "Figure 5: Pearson Correlation Matrix across Sensors, Weather, and Reference Analytes.",
         "Figure 5 reveals strong inter-variable correlations. Primary sensor PT08.S1(CO) correlates strongly with CO(GT) (r = 0.88), while PT08.S2(NMHC) exhibits r = 0.91 with CO(GT) "
         "and r = 0.98 with Benzene C6H6(GT), indicating high collinearity among combustion-generated hydrocarbons. Absolute humidity exhibits negative correlation with sensor resistance responses."),

        ("fig06_time_series.png", "Figure 6: Longitudinal Time-Series and 24-Hour Rolling Average of Observed CO(GT).",
         "Figure 6 displays the longitudinal progression of CO(GT) across 13 months. A distinct seasonal trend is visible: winter months (Nov 2004 to Feb 2005) exhibit significantly higher baseline concentrations "
         "and sharper peak spikes due to domestic heating emissions and low atmospheric boundary layer heights."),

        ("fig07_bivariate_scatter.png", "Figure 7: Bivariate Scatterplot of Target CO(GT) versus PT08.S1(CO) Sensor Response.",
         "Figure 7 demonstrates the strong linear relationship between PT08.S1(CO) nominal sensor response and actual CO concentration (Pearson r = 0.88, R² = 0.77). "
         "The residual dispersion increases at higher pollutant concentrations, reflecting semiconductor sensor saturation and temperature-dependent drift."),

        ("fig08_multivariate_scatter.png", "Figure 8: Multivariate Interaction: PT08.S1(CO) vs PT08.S2(NMHC) Colored by Target CO(GT).",
         "Figure 8 showcases a three-dimensional interaction. High CO(GT) levels (bright yellow points) cluster strictly in the upper-right quadrant where both PT08.S1(CO) and PT08.S2(NMHC) "
         "register simultaneous elevated voltage responses, confirming the mutual synergy of multi-sensor fusion."),

        ("fig09_diurnal_hourly_profile.png", "Figure 9: Diurnal Hourly Cycle Revealing Commuter Rush-Hour Traffic Peaks.",
         "Figure 9 depicts diurnal pollutant kinetics across 24 hours. Two distinct rush-hour peaks occur: a morning peak at 08:00 (mean CO ~ 2.7 mg/m³) and a prominent evening peak at 19:00–20:00 (mean CO ~ 3.3 mg/m³). "
         "Nighttime minimums occur at 04:00 (mean CO ~ 0.9 mg/m³), confirming vehicular traffic as the dominant local emission source.")
    ]

    for filename, caption, explanation in eda_images:
        img_path = fig_eda / filename
        if img_path.exists():
            story.append(KeepTogether([
                RLImage(str(img_path), width=440, height=230),
                Paragraph(caption, caption_style),
                Paragraph(explanation, body_style),
                Spacer(1, 6)
            ]))

    # ==================== 8. SEVEN STAGES OF VISUALIZATION ====================
    story.append(Paragraph("8. The Seven Stages of Visualization", h1_style))
    story.append(Paragraph(
        "To ensure methodological rigor, the project is explicitly mapped to Ben Fry's Seven Stages of Visualization framework:",
        body_style
    ))
    seven_stages = [
        "<b>1. Acquire:</b> Automated programmatic download and integrity validation of the official UCI Air Quality ZIP archive (DOI: 10.24432/C59K5F).",
        "<b>2. Parse:</b> Correct parsing of semicolon-separated structures, European comma decimals, and unified datetime reconstruction.",
        "<b>3. Filter:</b> Conversion of -200 sentinels to NaN, removal of unrecorded trailing records, and dropping unlabelled target rows (1,683 rows).",
        "<b>4. Mine:</b> Extraction of statistical summaries (mean, median, IQR, skewness, kurtosis), Pearson correlation matrices, and rush-hour groupings.",
        "<b>5. Represent:</b> Selection of specialized visual forms including dual-axis line charts, KDE histograms, normalized boxplots, bivariate regression scatters, and multi-sensor interaction scatters.",
        "<b>6. Refine:</b> Professional typographic scaling, unit labels, figure legends, colorblind-accessible diverging palettes, and 300 DPI exports.",
        "<b>7. Interact:</b> Implemented an interactive dual-axis Plotly visualization saved to outputs/interactive/interactive_air_quality.html. "
        "The dashboard features dynamic zoom, time-window pan, hover tooltips displaying instantaneous sensor and target readings, and an interactive rangeslider."
    ]
    for stg in seven_stages:
        story.append(Paragraph(stg, body_style))

    # ==================== 9. DEEP LEARNING MODEL ====================
    story.append(Paragraph("9. Deep Learning Model Design and Training", h1_style))
    story.append(Paragraph(
        "A Multi-Layer Perceptron (MLP) deep learning regression architecture was implemented using TensorFlow/Keras. "
        "The model ingests 14 standardized features: 5 metal oxide chemical sensor responses (PT08.S1 to PT08.S5), 3 meteorological parameters (T, RH, AH), "
        "and 6 engineered temporal features (hour_sin, hour_cos, month_sin, month_cos, day_of_week, is_weekend). "
        "The single output neuron utilizes a linear activation function to predict continuous CO(GT) in mg/m³.",
        body_style
    ))
    story.append(Paragraph("9.1 Hyperparameter Comparison and Model Selection", h2_style))
    story.append(Paragraph(
        "To objectively select the optimal model, two competing architectures were trained on the training split (5,371 rows) "
        "and evaluated strictly on the validation split (1,151 rows) with Early Stopping (patience = 15, monitoring val_loss, restoring best weights):",
        body_style
    ))
    story.append(Paragraph("• <b>Model A (Baseline 2-Layer MLP):</b> Dense(64, ReLU) -> Dropout(0.10) -> Dense(32, ReLU) -> Linear(1). Optimizer: Adam (lr = 0.001), Loss: MSE.", body_style))
    story.append(Paragraph("• <b>Model B (Deep Regularized 3-Layer MLP):</b> Dense(128, ReLU) -> Dropout(0.20) -> Dense(64, ReLU) -> Dropout(0.15) -> Dense(32, ReLU) -> Linear(1). Optimizer: Adam (lr = 0.001), Loss: MSE.", body_style))

    # Comp Table in PDF
    t_comp_data = [["Model", "Architecture", "Epochs", "Val MAE", "Val RMSE", "Val R²"]]
    for _, r in model_comp_df.iterrows():
        t_comp_data.append([
            str(r["Model"]), str(r["Architecture"])[:35] + "...",
            str(r["StoppedEpoch"]), f"{r['Val_MAE']:.4f}", f"{r['Val_RMSE']:.4f}", f"{r['Val_R2']:.4f}"
        ])
    t_comp = Table(t_comp_data, colWidths=[110, 160, 45, 55, 55, 50])
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#103460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Based strictly on validation performance, Model B achieved a superior Validation MAE of 0.5335 mg/m³ (compared to 0.5455 mg/m³ for Model A). "
        "Early stopping halted training at epoch 19 and restored optimal weights from epoch 4. Model B was selected as the final production model.",
        body_style
    ))

    # ==================== 10. RESULTS AND EVALUATION ====================
    story.append(Paragraph("10. Experimental Results and Model Evaluation", h1_style))
    story.append(Paragraph(
        "The selected model (Model B) was evaluated once on the strictly held-out chronological test partition (1,152 observations spanning Feb 14, 2005, to April 4, 2005). "
        "The model demonstrated exceptional generalization capabilities on unseen temporal data:",
        body_style
    ))

    t_res_data = [
        ["Metric", "Test Result", "Standard of Evaluation"],
        ["Test Observations (N)", str(metrics["N_samples"]), "Strictly held-out future chronological test split."],
        ["Mean Absolute Error (MAE)", f"{metrics['MAE']:.4f} mg/m³", "Average absolute error on original mg/m³ scale."],
        ["Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.4f} mg/m³", "Penalizes large error deviations."],
        ["Mean Squared Error (MSE)", f"{metrics['MSE']:.4f} (mg/m³)²", "Squared regression optimization loss."],
        ["Coefficient of Determination (R²)", f"{metrics['R2']:.4f}", "80.94% of variance explained in held-out test data."]
    ]
    t_res = Table(t_res_data, colWidths=[150, 100, 225])
    t_res.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#103460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 10))

    # Evaluation Images
    eval_images = [
        ("fig10_training_loss_curve.png", "Figure 10: Training and Validation Learning Curves across Epochs.",
         "Figure 10 illustrates learning curves for Model B. Training loss decreased smoothly while validation loss converged rapidly by epoch 4. "
         "The early stopping callback successfully halted training before validation degradation occurred, mitigating overfitting."),

        ("fig11_predicted_vs_actual.png", "Figure 11: Predicted vs Actual CO(GT) Scatter Plot on Held-Out Test Set.",
         "Figure 11 compares predicted versus actual CO(GT) concentrations for the 1,152 test samples. The points cluster tightly along the ideal 45-degree reference line (y = x), "
         "corroborating the high R² score of 0.8094. Minimal dispersion is observed across standard operating ranges (0.5 to 4.0 mg/m³)."),

        ("fig12_residual_distribution.png", "Figure 12: Residual Error Distribution on the Test Set.",
         "Figure 12 displays the distribution of prediction residuals (Actual - Predicted CO). The distribution is tightly centered at a near-zero mean (-0.021 mg/m³) "
         "with a standard deviation of 0.556 mg/m³, confirming that the deep learning model produces unbiased predictions across the temporal evaluation horizon."),

        ("fig13_residuals_vs_predicted.png", "Figure 13: Residuals versus Predicted CO Values Checking Homoscedasticity.",
         "Figure 13 demonstrates homoscedasticity across predicted concentrations. The rolling mean error trend remains flat near zero across predicted values up to 5.0 mg/m³, "
         "indicating consistent calibration fidelity across low, moderate, and high pollution regimes."),

        ("fig15_test_chronological_series.png", "Figure 15: Chronological Test Sequence: Ground Truth versus MLP Predictions.",
         "Figure 15 displays the full two-month test sequence from February to April 2005. The MLP prediction curve closely tracks both baseline trends and rapid diurnal rush-hour spikes, "
         "proving robust generalization over seasonal weather transitions.")
    ]

    for filename, caption, explanation in eval_images:
        img_path = fig_eval / filename
        if img_path.exists():
            story.append(KeepTogether([
                RLImage(str(img_path), width=440, height=220),
                Paragraph(caption, caption_style),
                Paragraph(explanation, body_style),
                Spacer(1, 6)
            ]))

    # Feature Importance Section
    story.append(Paragraph("10.1 Permutation Feature Importance Analysis", h2_style))
    story.append(Paragraph(
        "To assess feature contributions without relying on internal neural weights, permutation feature importance was conducted on the validation set across 5 random shuffle repeats. "
        "The resulting increase in MAE measures each feature's unique predictive value.",
        body_style
    ))
    img_imp = fig_eval / "fig14_feature_importance.png"
    if img_imp.exists():
        story.append(KeepTogether([
            RLImage(str(img_imp), width=440, height=230),
            Paragraph("Figure 14: Permutation Feature Importance on Validation Set.", caption_style),
            Paragraph(
                "The analysis demonstrates that PT08.S4(NO2) (mean MAE increase = +0.2829 mg/m³) and PT08.S2(NMHC) (+0.2560 mg/m³) are the most critical predictors, "
                "followed by PT08.S3(NOx) (+0.1088 mg/m³), Absolute Humidity AH (+0.1036 mg/m³), and primary sensor PT08.S1(CO) (+0.1009 mg/m³). "
                "Importantly, high feature importance reflects mathematical reliance within the trained network and must not be misinterpreted as scientific proof of physical causation, "
                "especially given known collinearity among co-emitted vehicular pollutants.",
                body_style
            ),
            Spacer(1, 6)
        ]))

    # ==================== 11. CONCLUSION ====================
    story.append(Paragraph("11. Conclusion", h1_style))
    story.append(Paragraph(
        "This project successfully developed an original, rigorous exploratory data analysis and deep learning regression pipeline for urban carbon monoxide concentration estimation. "
        "By enforcing strict data hygiene—eliminating -200 sentinels, dropping unlabelled target rows, preventing target leakage from other ground-truth analytes, "
        "and performing chronological partitioning—the deep learning MLP achieved strong generalization (R² = 0.8094, MAE = 0.3658 mg/m³, RMSE = 0.5564 mg/m³) on 1,152 unseen future test records. "
        "The findings prove that low-cost multisensor arrays, when calibrated using deep neural networks and environmental telemetry, provide a viable, cost-effective complement "
        "to expensive reference air quality monitoring stations.",
        body_style
    ))

    # ==================== 12. FUTURE SCOPE ====================
    story.append(Paragraph("12. Future Scope", h1_style))
    story.append(Paragraph("Subsequent investigations could expand upon this foundation through several high-value avenues:", body_style))
    f_scopes = [
        "1. <b>Recurrent and Transformer Architectures:</b> Implementing Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), or Temporal Fusion Transformers (TFT) to explicitly model multi-step temporal lag dependencies.",
        "2. <b>Dynamic Sensor Drift Compensation:</b> Developing continuous online calibration algorithms to compensate for long-term semiconductor chemical sensor aging and poisoning.",
        "3. <b>Ancillary Telemetry Integration:</b> Ingesting high-resolution localized vehicular traffic density and atmospheric boundary layer height data to improve extreme pollution event forecasting.",
        "4. <b>Edge Device Deployment:</b> Quantizing the trained MLP model using TensorFlow Lite for ultra-low-power embedded execution on IoT microcontrollers (e.g., ESP32 or Raspberry Pi)."
    ]
    for fs in f_scopes:
        story.append(Paragraph(fs, body_style))

    # ==================== 13. LITERATURE REVIEW ====================
    story.append(Paragraph("13. Literature Review", h1_style))
    story.append(Paragraph(
        "Urban air quality monitoring has witnessed a paradigm shift from centralized, sparse analytical stations toward dense, distributed sensor networks. "
        "In their seminal study, De Vito et al. (2008) deployed an electronic nose multisensor array in an urban road setting, demonstrating that low-cost chemical sensors "
        "could estimate benzene and carbon monoxide when properly calibrated against certified ground-truth analyzers. However, classical linear calibration models struggled "
        "with sensor non-linearities, temperature drift, and humidity interference. Recent advances in deep learning (Goodfellow et al., 2016) demonstrate that Multi-Layer Perceptrons "
        "and non-linear activation functions effectively learn complex multidimensional sensor response surfaces. Furthermore, Fry (2008) formulated the Seven Stages of Visualization, "
        "providing a structured data interaction paradigm for high-dimensional scientific and environmental datasets.",
        body_style
    ))

    # ==================== 14. REFERENCES ====================
    story.append(Paragraph("14. References", h1_style))
    refs = [
        "[1] S. De Vito, E. Massera, M. Piga, L. Martinotto, and G. Di Francia, 'On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario,' Sensors and Actuators B: Chemical, vol. 129, no. 2, pp. 750–757, 2008. DOI: 10.1016/j.snb.2007.09.060.",
        "[2] UC Irvine Machine Learning Repository, 'Air Quality Dataset,' UCI ML Repository, 2016. [Online]. Available: https://archive.ics.uci.edu/dataset/360/air+quality. DOI: 10.24432/C59K5F.",
        "[3] World Health Organization, 'WHO Global Air Quality Guidelines: Particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide,' World Health Organization, Geneva, Switzerland, 2021.",
        "[4] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python,' Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
        "[5] M. Abadi et al., 'TensorFlow: A system for large-scale machine learning,' in 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), 2016, pp. 265–283.",
        "[6] B. Fry, Visualizing Data: Exploring and Explaining Data with the Processing Environment, O'Reilly Media, Sebastopol, CA, 2008."
    ]
    for ref in refs:
        story.append(Paragraph(ref, body_style))

    # ==================== 15. APPENDIX ====================
    story.append(Paragraph("15. Appendix", h1_style))
    story.append(Paragraph("15.1 Software Environment and Package Versions", h2_style))
    t_env_data = [
        ["Package", "Version", "Role in Pipeline"],
        ["Python", "3.12.10", "Core programming runtime"],
        ["TensorFlow / Keras", "2.21.0 / 3.15.1", "Deep learning MLP architecture & training"],
        ["scikit-learn", "1.8.0", "Train-only imputation, scaling, regression metrics"],
        ["pandas", "2.3.3", "Data structures and chronological time series processing"],
        ["numpy", "2.4.3", "Numerical linear algebra and cyclical transformations"],
        ["matplotlib", "3.10.8", "Publication-quality 300 DPI chart generation"],
        ["seaborn", "0.13.2", "Statistical distribution plotting and styling"],
        ["plotly", "7.1.0", "Interactive Stage 7 HTML dashboard rendering"]
    ]
    t_env = Table(t_env_data, colWidths=[120, 95, 260])
    t_env.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#103460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_env)
    story.append(Spacer(1, 8))

    story.append(Paragraph("15.2 Project Repository File Manifest", h2_style))
    manifest = [
        "• <b>src/data_loader.py:</b> Automated UCI dataset downloader and data quality inspector.",
        "• <b>src/preprocessing.py:</b> Implements sentinel cleaning, leakage exclusion, cyclical feature engineering, and train-only scaling.",
        "• <b>src/eda.py:</b> Generates 9 publication-grade figures, descriptive statistics, and interactive Plotly visualization.",
        "• <b>src/train_model.py:</b> Compares competing MLP regression models and executes early stopping on validation loss.",
        "• <b>src/evaluate_model.py:</b> Executes held-out test evaluation, generates regression diagnostic charts, and computes permutation importance.",
        "• <b>src/utils.py:</b> Manages directory structures, random seed setting, and metric calculations.",
        "• <b>tests/test_preprocessing.py:</b> Automated test suite verifying data integrity and absence of data leakage.",
        "• <b>notebooks/Air_Quality_EDA_Deep_Learning.ipynb:</b> Fully executable end-to-end Jupyter notebook."
    ]
    for mf in manifest:
        story.append(Paragraph(mf, body_style))

    # Build PDF with custom NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[Report] PDF report successfully generated at: {output_path}")


if __name__ == "__main__":
    root = get_project_root()
    docx_path = root / "Project_Report.docx"
    pdf_path = root / "Project_Report.pdf"

    print("[ReportGenerator] Generating DOCX report...")
    generate_docx_report(docx_path)

    print("[ReportGenerator] Generating PDF report...")
    generate_pdf_report(pdf_path)

    print("[ReportGenerator] Both reports successfully generated!")
