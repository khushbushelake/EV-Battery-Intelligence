from io import BytesIO
import base64
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from pathlib import Path

# =========================================
# PROJECT PATHS
# =========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports"


def generate_technical_report(
    model_results,
    test_results,
    overall_results,
    predictions
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=50,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=23,
        leading=29,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=12
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=20
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=15,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        spaceAfter=8
    ))

    story = []

    # ---------------------------------
    # DOCUMENT TITLE
    # ---------------------------------

    story.append(Spacer(1, 35))

    story.append(Paragraph(
        "EV BATTERY INTELLIGENCE SYSTEM",
        styles["ReportTitle"]
    ))

    story.append(Paragraph(
        "AI-Powered Lithium-Ion Battery State of Charge Estimation",
        styles["ReportSubtitle"]
    ))

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        f"Technical Evaluation Report | "
        f"Generated: {datetime.now().strftime('%d %B %Y')}",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "Developed by: Khushbu Uttam Shelake",
        styles["BodyCustom"]
    ))

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "EXECUTIVE SUMMARY",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "This report presents an AI-powered battery State of Charge "
        "(SOC) estimation system developed using three recurrent "
        "deep-learning architectures: Long Short-Term Memory (LSTM), "
        "Gated Recurrent Unit (GRU), and Bidirectional LSTM (BiLSTM). "
        "The models were trained on prepared lithium-ion battery "
        "time-series data and evaluated using validation and "
        "independent test datasets.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "The independent test evaluation covers four temperature "
        "conditions: -10°C, 0°C, 10°C and 25°C. The following "
        "sections summarize the model configurations, evaluation "
        "metrics, prediction analysis and key findings.",
        styles["BodyCustom"]
    ))

    story.append(Spacer(1, 10))

    # ---------------------------------
    # PROJECT OVERVIEW
    # ---------------------------------

    story.append(Paragraph(
        "1. SYSTEM OVERVIEW",
        styles["SectionHeading"]
    ))

    overview_data = [
        ["Project Parameter", "Description"],
        ["Dataset", "LG 18650HG2 Lithium-Ion Battery Dataset"],
        ["Training Observations", "669,956"],
        ["Input Features", "5 prepared input features"],
        ["Sequence Length", "30 time steps"],
        ["Sequence Stride", "5"],
        ["Prediction Target", "Battery State of Charge (SOC)"],
        ["Models", "LSTM, GRU and BiLSTM"],
        ["Test Temperatures", "-10°C, 0°C, 10°C and 25°C"]
    ]

    table = Table(
        overview_data,
        colWidths=[150, 330],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(table)

    # ---------------------------------
    # MODEL ARCHITECTURES
    # ---------------------------------

    story.append(Paragraph(
        "2. MODEL ARCHITECTURES",
        styles["SectionHeading"]
    ))

    architecture_data = [
        ["Model", "Recurrent Configuration", "Additional Layers"],
        ["LSTM", "LSTM (64 units)", "Dropout (0.2), Dense (32), Output (1)"],
        ["GRU", "GRU (64 units)", "Dropout (0.2), Dense (32), Output (1)"],
        ["BiLSTM", "Bidirectional LSTM (64 units per direction)",
         "Dropout (0.2), Dense (32), Output (1)"]
    ]

    architecture_table = Table(
        architecture_data,
        colWidths=[65, 180, 235],
        repeatRows=1
    )

    architecture_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(architecture_table)

    story.append(Paragraph(
        "The models use the Adam optimizer and Mean Squared Error "
        "loss. Training used a maximum of five epochs with early "
        "stopping patience of two epochs.",
        styles["BodyCustom"]
    ))

    # ---------------------------------
    # VALIDATION RESULTS
    # ---------------------------------

    story.append(Paragraph(
        "3. VALIDATION PERFORMANCE",
        styles["SectionHeading"]
    ))

    def dataframe_table(df, max_rows=20):

        if df is None or df.empty:
            return None

        display_df = df.head(max_rows).copy()

        data = [
            [str(column) for column in display_df.columns]
        ]

        for row in display_df.itertuples(index=False, name=None):

            data.append([
                "-" if pd.isna(value) else str(value)
                for value in row
            ])

        col_width = 480 / len(data[0])

        result_table = Table(
            data,
            colWidths=[col_width] * len(data[0]),
            repeatRows=1
        )

        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F1F5F9")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ]))

        return result_table

    validation_table = dataframe_table(model_results)

    if validation_table:
        story.append(validation_table)
    else:
        story.append(Paragraph(
            "Validation results were not available when this report "
            "was generated.",
            styles["BodyCustom"]
        ))

    # ---------------------------------
    # INDEPENDENT TEST RESULTS
    # ---------------------------------

    story.append(Paragraph(
        "4. INDEPENDENT TEST EVALUATION",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "The trained models were evaluated on separate test files "
        "representing four battery temperature conditions.",
        styles["BodyCustom"]
    ))

    overall_table = dataframe_table(overall_results)

    if overall_table:
        story.append(Paragraph(
            "Overall Model Performance",
            styles["Heading3"]
        ))
        story.append(overall_table)

    test_table = dataframe_table(test_results)

    if test_table:
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            "Temperature-wise Performance",
            styles["Heading3"]
        ))
        story.append(test_table)

    # ---------------------------------
    # PREDICTION ANALYSIS
    # ---------------------------------

    story.append(Paragraph(
        "5. SOC PREDICTION ANALYSIS",
        styles["SectionHeading"]
    ))

    if predictions is not None and not predictions.empty:

        actual = predictions["Actual_SOC"].to_numpy()
        predicted = predictions["Predicted_SOC"].to_numpy()

        mae = float(np.mean(np.abs(actual - predicted)))

        rmse = float(
            np.sqrt(np.mean((actual - predicted) ** 2))
        )

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        r2 = (
            float(1 - ss_res / ss_tot)
            if ss_tot != 0 else float("nan")
        )

        prediction_data = [
            ["Prediction Metric", "Value"],
            ["Total Predictions", f"{len(predictions):,}"],
            ["MAE", f"{mae:.5f}"],
            ["RMSE", f"{rmse:.5f}"],
            ["R² Score", f"{r2:.5f}"]
        ]

        prediction_table = Table(
            prediction_data,
            colWidths=[200, 280]
        )

        prediction_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F1F5F9")]),
            ("PADDING", (0, 0), (-1, -1), 7)
        ]))

        story.append(prediction_table)

    else:

        story.append(Paragraph(
            "Prediction records were not available when this report "
            "was generated.",
            styles["BodyCustom"]
        ))

    # ---------------------------------
    # INCLUDE AVAILABLE CHARTS
    # ---------------------------------

    story.append(Paragraph(
        "6. PERFORMANCE VISUALIZATIONS",
        styles["SectionHeading"]
    ))

    chart_files = [
        "test_mae_comparison.png",
        "test_rmse_comparison.png",
        "test_r2_comparison.png",
        "bilstm_actual_vs_predicted.png"
    ]

    for filename in chart_files:

        image_path = REPORTS_DIR / filename

        if image_path.exists():

            story.append(Spacer(1, 8))

            story.append(Image(
                str(image_path),
                width=450,
                height=235,
                kind="proportional"
            ))

    # ---------------------------------
    # FINDINGS AND LIMITATIONS
    # ---------------------------------

    story.append(Paragraph(
        "7. KEY FINDINGS",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "The independent test results indicate that BiLSTM achieved "
        "the strongest overall performance among the three evaluated "
        "architectures, based on the reported average test MAE, RMSE "
        "and R² metrics. Temperature-specific results should be "
        "considered when interpreting model performance.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "8. LIMITATIONS AND SCOPE",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "This project evaluates prepared experimental battery "
        "datasets. The dashboard analyzes saved model predictions "
        "and is not connected to a live electric vehicle battery. "
        "The reported results apply to the dataset and test conditions "
        "used in this experiment; they do not by themselves establish "
        "performance on every battery chemistry, vehicle or operating "
        "environment.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "9. CONCLUSION",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "The EV Battery Intelligence System demonstrates the use of "
        "recurrent deep-learning architectures for lithium-ion battery "
        "SOC estimation. By comparing LSTM, GRU and BiLSTM models and "
        "evaluating their predictions under multiple temperature "
        "conditions, the project provides a structured basis for "
        "analyzing model performance and prediction errors.",
        styles["BodyCustom"]
    ))

    # ---------------------------------
    # FOOTER ON EACH PAGE
    # ---------------------------------

    def add_page_footer(canvas, document):

        canvas.saveState()

        canvas.setStrokeColor(
            colors.HexColor("#CBD5E1")
        )

        canvas.line(
            45, 32, A4[0] - 45, 32
        )

        canvas.setFont("Helvetica", 8)

        canvas.setFillColor(
            colors.HexColor("#64748B")
        )

        canvas.drawString(
            45,
            20,
            "EV Battery Intelligence System"
        )

        canvas.drawRightString(
            A4[0] - 45,
            20,
            f"Page {document.page}"
        )

        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=add_page_footer,
        onLaterPages=add_page_footer
    )

    buffer.seek(0)

    return buffer.getvalue()

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="EV Battery Intelligence",
    page_icon="🔋",
    layout="wide"
)

# =========================================
# HELPER FUNCTIONS
# =========================================

def display_image(title, filename):

    image_path = REPORTS_DIR / filename

    st.subheader(title)

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"{filename} not found.")


def load_csv(filename):

    file_path = REPORTS_DIR / filename

    if file_path.exists():
        return pd.read_csv(file_path)

    return None


# =========================================
# DASHBOARD HEADER
# =========================================

st.title("🔋 EV Battery Intelligence System")

st.markdown(
    "### AI-Powered Battery State of Charge (SOC) Estimation"
)

st.markdown(
    "An intelligent system comparing LSTM, GRU and BiLSTM "
    "deep-learning architectures for lithium-ion battery SOC estimation."
)

st.divider()

# =========================================
# SIDEBAR NAVIGATION
# =========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Project Overview",
        "Battery Data Analysis",
        "Model Comparison",
        "Independent Test Evaluation",
        "SOC Prediction",
        "Model Architecture",
        "Technical Report"
    ]
)

# =========================================
# PAGE 1: PROJECT OVERVIEW
# =========================================


# =========================================
# PAGE 1: BATTERY INTELLIGENCE OVERVIEW
# =========================================

if page == "Project Overview":

    st.header("Battery Intelligence Overview")

    st.markdown(
        "An AI-powered analytical platform for understanding "
        "and estimating lithium-ion battery State of Charge."
    )

    st.divider()

    # ---------------------------------
    # 1. LOAD PROJECT RESULTS
    # ---------------------------------

    overall_results = load_csv("overall_test_results.csv")

    test_results = load_csv("independent_test_results.csv")

    predictions_path = REPORTS_DIR / "bilstm_soc_predictions.csv"

    predictions = None

    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)

    # ---------------------------------
    # 2. KEY PERFORMANCE INDICATORS
    # ---------------------------------

    st.subheader("Project at a Glance")

    total_predictions = (
        len(predictions)
        if predictions is not None
        else 34703
    )

    best_model = "BiLSTM"
    average_mae = 0.01842
    average_rmse = 0.02435
    average_r2 = 0.99021

    if overall_results is not None and not overall_results.empty:

        best_row = overall_results.loc[
            overall_results["MAE"].idxmin()
        ]

        best_model = str(best_row["Model"])
        average_mae = float(best_row["MAE"])
        average_rmse = float(best_row["RMSE"])
        average_r2 = float(best_row["R2"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Training Observations",
        "669,956"
    )

    col2.metric(
        "Test Predictions",
        f"{total_predictions:,}"
    )

    col3.metric(
        "Best Performing Model",
        best_model
    )

    col4.metric(
        "Average Test R²",
        f"{average_r2:.5f}"
    )

    st.divider()

    # ---------------------------------
    # 3. MODEL PERFORMANCE HIGHLIGHTS
    # ---------------------------------

    st.subheader("Best Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Test MAE",
        f"{average_mae:.5f}"
    )

    col2.metric(
        "Average Test RMSE",
        f"{average_rmse:.5f}"
    )

    col3.metric(
        "Average Test R²",
        f"{average_r2:.5f}"
    )

    st.caption(
        "Metrics are unweighted averages across the four "
        "temperature-specific independent test evaluations."
    )

    st.success(
        f"{best_model} achieved the strongest overall "
        "independent test performance."
    )

    st.divider()

    # ---------------------------------
    # 4. TEMPERATURE COVERAGE
    # ---------------------------------

    st.subheader("Independent Test Temperature Coverage")

    temperature_order = ["-10°C", "0°C", "10°C", "25°C"]

    if predictions is not None:

        coverage = (
            predictions.groupby("Temperature")
            .size()
            .reindex(temperature_order, fill_value=0)
            .reset_index(name="Predictions")
        )

        col1, col2, col3, col4 = st.columns(4)

        for col, (_, row) in zip(
            [col1, col2, col3, col4],
            coverage.iterrows()
        ):

            col.metric(
                str(row["Temperature"]),
                f'{int(row["Predictions"]):,}'
            )
        st.caption(
            "Number of independent test predictions generated "
            "at each battery temperature."
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            coverage["Temperature"],
            coverage["Predictions"]
        )

        ax.set_title("Test Prediction Distribution by Temperature")
        ax.set_xlabel("Battery Temperature")
        ax.set_ylabel("Number of Predictions")
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:

        st.warning(
            "Prediction CSV not found. Temperature coverage "
            "could not be loaded."
        )

    st.divider()

    # ---------------------------------
    # 5. MODEL COMPARISON SUMMARY
    # ---------------------------------

    st.subheader("Deep Learning Model Comparison")

    if overall_results is not None:

        display_columns = [
            col for col in ["Model", "MAE", "RMSE", "R2"]
            if col in overall_results.columns
        ]

        st.dataframe(
            overall_results[display_columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Overall model comparison file is unavailable."
        )

    st.divider()

    # ---------------------------------
    # 6. PROJECT WORKFLOW
    # ---------------------------------

    st.subheader("How the System Works")

    workflow = [
        ("01", "Data Collection", "LG 18650HG2 battery dataset"),
        ("02", "Preprocessing", "Cleaning, normalization and feature preparation"),
        ("03", "Sequence Creation", "30 time steps with stride of 5"),
        ("04", "Model Training", "LSTM, GRU and BiLSTM"),
        ("05", "Model Evaluation", "Validation and independent testing"),
        ("06", "SOC Analysis", "Prediction visualization and error analysis")
    ]

    for number, title, description in workflow:

        col1, col2 = st.columns([1, 5])

        col1.markdown(f"### {number}")

        col2.markdown(f"**{title}**")
        col2.caption(description)

    st.divider()

    # ---------------------------------
    # 7. PROJECT INFORMATION
    # ---------------------------------

    st.subheader("About the Project")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Dataset**")
        st.write("LG 18650HG2 Lithium-Ion Battery Dataset")

        st.markdown("**Input Features**")
        st.write("5 prepared battery input features")

        st.markdown("**Prediction Target**")
        st.write("Battery State of Charge (SOC)")

    with col2:

        st.markdown("**Deep Learning Models**")
        st.write("LSTM, GRU and BiLSTM")

        st.markdown("**Sequence Length**")
        st.write("30 time steps")

        st.markdown("**Test Conditions**")
        st.write("-10°C, 0°C, 10°C and 25°C")

    st.info(
        "This dashboard analyzes saved experimental battery "
        "test predictions. It is not connected to a live EV battery."
    )


# =========================================
# PAGE 2: BATTERY DATA ANALYSIS
# =========================================

elif page == "Battery Data Analysis":

    st.header("Battery Data Analysis")

    graph_files = {
        "SOC Distribution": "soc_distribution.png",
        "Feature Distributions": "feature_distributions.png",
        "SOC Variation": "soc_variation.png",
        "Correlation Matrix": "correlation_matrix.png"
    }

    for title, filename in graph_files.items():

        display_image(title, filename)

        st.divider()


# =========================================
# PAGE 3: MODEL COMPARISON
# =========================================

elif page == "Model Comparison":

    st.header("Validation Model Performance Comparison")

    results = load_csv("model_comparison_results.csv")

    if results is not None:

        st.subheader("Validation Evaluation Metrics")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        display_image(
            "MAE Comparison",
            "model_mae_comparison.png"
        )

        display_image(
            "RMSE Comparison",
            "model_rmse_comparison.png"
        )

        display_image(
            "R² Comparison",
            "model_r2_comparison.png"
        )

        display_image(
            "BiLSTM: Actual vs Predicted SOC",
            "bilstm_actual_vs_predicted.png"
        )

    else:

        st.warning("Model comparison results not found.")


# =========================================
# PAGE 4: INDEPENDENT TEST EVALUATION
# =========================================

elif page == "Independent Test Evaluation":

    st.header("Independent Test Evaluation")

    st.write(
        "Evaluation of the trained models on separate test datasets "
        "under four different temperature conditions."
    )

    # Load test results
    test_results = load_csv("independent_test_results.csv")
    overall_results = load_csv("overall_test_results.csv")

    if test_results is not None:

        # Overall performance
        st.subheader("Overall Model Performance")

        if overall_results is not None:

            # Identify best model using lowest MAE
            best_row = overall_results.loc[
                overall_results["MAE"].idxmin()
            ]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Best Performing Model",
                str(best_row["Model"])
            )

            col2.metric(
                "Average Test MAE",
                f'{best_row["MAE"]:.5f}'
            )

            col3.metric(
                "Average Test RMSE",
                f'{best_row["RMSE"]:.5f}'
            )

            st.subheader("Overall Test Metrics")

            st.dataframe(
                overall_results,
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f'{best_row["Model"]} achieved the lowest average '
                f'test MAE of {best_row["MAE"]:.5f}.'
            )

            st.caption(
                "Overall metrics are unweighted averages of the "
                "four temperature-specific results."
            )

        else:

            st.warning("overall_test_results.csv not found.")

        st.divider()

        # Temperature-wise performance
        st.subheader("Temperature-wise Test Performance")

        st.dataframe(
            test_results,
            use_container_width=True,
            hide_index=True
        )

        # Total observations evaluated
        if "Observations" in test_results.columns:

            total_observations = int(
                test_results.groupby("Temperature")[
                    "Observations"
                ].first().sum()
            )

            st.metric(
                "Total Test Observations Evaluated",
                f"{total_observations:,}"
            )

        st.divider()

        # Test comparison graphs
        st.subheader("Independent Test Performance Charts")

        display_image(
            "MAE Comparison Across Temperatures",
            "test_mae_comparison.png"
        )

        display_image(
            "RMSE Comparison Across Temperatures",
            "test_rmse_comparison.png"
        )

        display_image(
            "R² Comparison Across Temperatures",
            "test_r2_comparison.png"
        )

        st.divider()

        st.subheader("Key Findings")

        st.write("""
        - BiLSTM achieved the lowest average MAE among the three models.
        - BiLSTM demonstrated the strongest overall test performance
          across all four temperature conditions.
        - LSTM provided the second-best overall performance.
        - GRU recorded higher prediction errors than the other two models.
        - The evaluation covers test datasets at −10°C, 0°C, 10°C and 25°C.
        """)

    else:

        st.error(
            "Independent test results not found. "
            "Please check independent_test_results.csv in the reports folder."
        )


# =========================================
# PAGE 5: ADVANCED SOC PREDICTION ANALYSIS
# =========================================

elif page == "SOC Prediction":

    st.header("Battery SOC Intelligence")

    st.write(
        "Explore BiLSTM predictions, prediction errors, "
        "and model performance across different temperatures."
    )

    prediction_path = REPORTS_DIR / "bilstm_soc_predictions.csv"

    if prediction_path.exists():

        predictions = pd.read_csv(prediction_path)

        required_columns = [
            "Temperature",
            "Observation",
            "Actual_SOC",
            "Predicted_SOC",
            "Absolute_Error"
        ]

        if not all(
            col in predictions.columns for col in required_columns
        ):

            st.error("Prediction CSV is missing required columns.")

        else:

            temperatures = ["-10°C", "0°C", "10°C", "25°C"]

            selected_temperature = st.selectbox(
                "Select Battery Temperature",
                temperatures
            )

            filtered = predictions[
                predictions["Temperature"] == selected_temperature
            ].copy()

            if not filtered.empty:

                actual = filtered["Actual_SOC"].to_numpy()
                predicted = filtered["Predicted_SOC"].to_numpy()

                errors = predicted - actual
                absolute_errors = np.abs(errors)

                # Calculate metrics
                mae = float(np.mean(absolute_errors))

                rmse = float(
                    np.sqrt(np.mean(errors ** 2))
                )

                ss_res = np.sum(errors ** 2)

                ss_tot = np.sum(
                    (actual - np.mean(actual)) ** 2
                )

                r2 = (
                    float(1 - ss_res / ss_tot)
                    if ss_tot != 0 else 0.0
                )

                # ---------------------------------
                # 1. MODEL PERFORMANCE
                # ---------------------------------

                st.subheader("Model Performance")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("MAE", f"{mae:.5f}")
                col2.metric("RMSE", f"{rmse:.5f}")
                col3.metric("R² Score", f"{r2:.5f}")
                col4.metric("Test Samples", f"{len(filtered):,}")

                st.divider()

                # ---------------------------------
                # 2. INDIVIDUAL SOC PREDICTION
                # ---------------------------------

                st.subheader("Individual SOC Prediction")

                observation_number = st.slider(
                    "Select Observation",
                    min_value=1,
                    max_value=len(filtered),
                    value=1
                )

                selected_row = filtered.iloc[
                    observation_number - 1
                ]

                actual_soc = float(selected_row["Actual_SOC"])

                predicted_soc = float(
                    selected_row["Predicted_SOC"]
                )

                error = abs(actual_soc - predicted_soc)

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Predicted SOC",
                    f"{predicted_soc * 100:.2f}%"
                )

                col2.metric(
                    "Actual SOC",
                    f"{actual_soc * 100:.2f}%"
                )

                col3.metric(
                    "Absolute Error",
                    f"{error * 100:.2f} pp"
                )

                st.divider()

                # ---------------------------------
                # 3. PROFESSIONAL BATTERY SOC GAUGE
                # ---------------------------------

                st.subheader("Battery Charge Indicator")

                soc_display = float(
                    np.clip(predicted_soc, 0, 1)
                )

                soc_percentage = soc_display * 100

                # Battery status
                if soc_percentage < 20:

                    battery_status = "LOW CHARGE"
                    status_color = "#DC2626"

                elif soc_percentage < 50:

                    battery_status = "MODERATE CHARGE"
                    status_color = "#EA580C"

                elif soc_percentage < 80:

                    battery_status = "GOOD CHARGE"
                    status_color = "#CA8A04"

                else:

                    battery_status = "HIGH CHARGE"
                    status_color = "#16A34A"

                # Create gauge
                fig, ax = plt.subplots(figsize=(8, 4))

                segments = [
                    (0, 20, "#DC2626"),
                    (20, 50, "#EA580C"),
                    (50, 80, "#EAB308"),
                    (80, 100, "#16A34A")
                ]

                # Draw gauge segments
                for start, end, color in segments:

                    theta_start = 180 - (end / 100) * 180
                    theta_end = 180 - (start / 100) * 180

                    arc = Wedge(
                        (0, 0),
                        1,
                        theta_start,
                        theta_end,
                        width=0.18,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=2
                    )

                    ax.add_patch(arc)

                # Gauge needle
                angle = np.pi * (1 - soc_display)

                needle_x = 0.75 * np.cos(angle)
                needle_y = 0.75 * np.sin(angle)

                ax.plot(
                    [0, needle_x],
                    [0, needle_y],
                    linewidth=3,
                    color="#1F2937"
                )

                # Needle center
                ax.scatter(
                    0,
                    0,
                    s=130,
                    color="#1F2937",
                    zorder=5
                )

                # SOC percentage
                ax.text(
                    0,
                    0.30,
                    f"{soc_percentage:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=27,
                    fontweight="bold"
                )

                # Battery status
                ax.text(
                    0,
                    0.08,
                    battery_status,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=status_color
                )

                # Gauge labels
                ax.text(
                    -1.05,
                    -0.08,
                    "0%",
                    fontsize=10,
                    ha="center"
                )

                ax.text(
                    1.05,
                    -0.08,
                    "100%",
                    fontsize=10,
                    ha="center"
                )

                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-0.15, 1.15)
                ax.set_aspect("equal")
                ax.axis("off")

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

                st.caption(
                    "Gauge represents the BiLSTM-predicted SOC "
                    "for the selected independent test observation."
                )

                st.divider()

                # ---------------------------------
                # 4. ACTUAL VS PREDICTED SCATTER
                # ---------------------------------

                st.subheader("Actual vs Predicted SOC")

                sample = filtered.sample(
                    n=min(3000, len(filtered)),
                    random_state=42
                )

                fig, ax = plt.subplots(figsize=(8, 5))

                ax.scatter(
                    sample["Actual_SOC"] * 100,
                    sample["Predicted_SOC"] * 100,
                    alpha=0.35,
                    s=12
                )

                lower = min(
                    sample["Actual_SOC"].min(),
                    sample["Predicted_SOC"].min()
                ) * 100

                upper = max(
                    sample["Actual_SOC"].max(),
                    sample["Predicted_SOC"].max()
                ) * 100

                ax.plot(
                    [lower, upper],
                    [lower, upper],
                    linestyle="--",
                    label="Ideal Prediction"
                )

                ax.set_xlabel("Actual SOC (%)")
                ax.set_ylabel("Predicted SOC (%)")
                ax.set_title("Actual vs Predicted SOC")
                ax.legend()
                ax.grid(alpha=0.25)

                st.pyplot(fig)

                plt.close(fig)

                st.divider()

                # ---------------------------------
                # 5. RESIDUAL ANALYSIS
                # ---------------------------------

                st.subheader("Prediction Residual Analysis")

                residual_data = pd.DataFrame({
                    "Predicted SOC (%)": predicted * 100,
                    "Residual (pp)": errors * 100
                })

                fig, ax = plt.subplots(figsize=(8, 5))

                ax.scatter(
                    residual_data["Predicted SOC (%)"].iloc[::5],
                    residual_data["Residual (pp)"].iloc[::5],
                    alpha=0.35,
                    s=12
                )

                ax.axhline(
                    0,
                    linestyle="--",
                    label="Zero Error"
                )

                ax.set_xlabel("Predicted SOC (%)")
                ax.set_ylabel("Residual (percentage points)")
                ax.set_title("Residuals vs Predicted SOC")
                ax.legend()
                ax.grid(alpha=0.25)

                st.pyplot(fig)

                plt.close(fig)

                st.caption(
                    "Residual = Predicted SOC − Actual SOC. "
                    "Positive values indicate overestimation; "
                    "negative values indicate underestimation."
                )

                st.divider()

                # ---------------------------------
                # 6. ERROR DISTRIBUTION
                # ---------------------------------

                st.subheader("Prediction Error Distribution")

                fig, ax = plt.subplots(figsize=(8, 5))

                ax.hist(
                    absolute_errors * 100,
                    bins=40,
                    edgecolor="black"
                )

                ax.set_xlabel("Absolute Error (percentage points)")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Absolute Prediction Errors")
                ax.grid(axis="y", alpha=0.25)

                st.pyplot(fig)

                plt.close(fig)

                st.divider()

                # ---------------------------------
                # 7. TEMPERATURE-WISE COMPARISON
                # ---------------------------------

                st.subheader("Performance Across Temperatures")

                temperature_results = []

                for temperature in temperatures:

                    temp_data = predictions[
                        predictions["Temperature"] == temperature
                    ]

                    if temp_data.empty:
                        continue

                    y_true = temp_data["Actual_SOC"].to_numpy()
                    y_pred = temp_data["Predicted_SOC"].to_numpy()

                    temp_mae = np.mean(
                        np.abs(y_true - y_pred)
                    )

                    temp_rmse = np.sqrt(
                        np.mean((y_true - y_pred) ** 2)
                    )

                    temp_ss_res = np.sum(
                        (y_true - y_pred) ** 2
                    )

                    temp_ss_tot = np.sum(
                        (y_true - np.mean(y_true)) ** 2
                    )

                    temp_r2 = (
                        1 - temp_ss_res / temp_ss_tot
                        if temp_ss_tot != 0 else np.nan
                    )

                    temperature_results.append({
                        "Temperature": temperature,
                        "MAE": temp_mae,
                        "RMSE": temp_rmse,
                        "R²": temp_r2,
                        "Observations": len(temp_data)
                    })

                temperature_df = pd.DataFrame(
                    temperature_results
                )

                st.dataframe(
                    temperature_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.divider()

                # ---------------------------------
                # 8. PREDICTION RECORDS
                # ---------------------------------

                st.subheader("Prediction Records")

                st.dataframe(
                    filtered.head(100),
                    use_container_width=True,
                    hide_index=True
                )

                # ---------------------------------
                # 9. DOWNLOAD RESULTS
                # ---------------------------------

                st.download_button(
                    label="Download Selected Temperature Predictions",
                    data=filtered.to_csv(index=False),
                    file_name=(
                        f"bilstm_predictions_{selected_temperature}.csv"
                    ),
                    mime="text/csv"
                )

                st.success(
                    f"Successfully analyzed {len(filtered):,} "
                    f"BiLSTM predictions for {selected_temperature}."
                )

            else:

                st.warning("No prediction records found.")

    else:

        st.error(
            "Prediction file not found. Place "
            "bilstm_soc_predictions.csv inside the reports folder."
        )


# =========================================
# PAGE 6: MODEL ARCHITECTURE
# =========================================


# =========================================
# PAGE 6: MODEL ARCHITECTURE
# =========================================

elif page == "Model Architecture":

    st.header("Deep Learning Model Architecture")

    st.write(
        "Explore the three recurrent neural network architectures "
        "developed for lithium-ion battery State of Charge estimation."
    )

    st.divider()

    # ---------------------------------
    # 1. INPUT DATA CONFIGURATION
    # ---------------------------------

    st.subheader("Input Data Configuration")

    col1, col2, col3 = st.columns(3)

    col1.metric("Sequence Length", "30 Time Steps")
    col2.metric("Input Features", "5")
    col3.metric("Prediction Output", "1 SOC Value")

    st.info(
        "Each model receives a sequence of 30 time steps, "
        "with 5 input features at every time step, and predicts "
        "one State of Charge value."
    )

    st.divider()

    # ---------------------------------
    # 2. LSTM ARCHITECTURE
    # ---------------------------------

    st.subheader("1. Long Short-Term Memory (LSTM)")

    st.write(
        "LSTM is a recurrent neural network architecture "
        "designed to learn long-term dependencies in sequential data."
    )

    st.markdown("**Architecture Flow**")

    lstm_steps = [
        ("Input Layer", "30 time steps × 5 features"),
        ("LSTM Layer", "64 units"),
        ("Dropout", "0.2"),
        ("Dense Layer", "32 neurons · ReLU"),
        ("Output Layer", "1 neuron · SOC prediction")
    ]

    for i, (name, description) in enumerate(lstm_steps):

        st.markdown(
            f"""
            <div style="
                border:1px solid #3B82F6;
                border-radius:10px;
                padding:12px;
                margin:5px 0;
                text-align:center;
            ">
                <strong>{name}</strong><br>
                {description}
            </div>
            """,
            unsafe_allow_html=True
        )

        if i < len(lstm_steps) - 1:
            st.markdown(
                "<div style='text-align:center;'>↓</div>",
                unsafe_allow_html=True
            )

    with st.expander("How LSTM works"):

        st.write("""
        LSTM uses a memory cell and three gates:

        - **Forget Gate:** Decides which information to discard.
        - **Input Gate:** Controls which new information is stored.
        - **Output Gate:** Determines what information is passed forward.

        These mechanisms help the model learn patterns across
        sequential battery measurements.
        """)

    st.divider()

    # ---------------------------------
    # 3. GRU ARCHITECTURE
    # ---------------------------------

    st.subheader("2. Gated Recurrent Unit (GRU)")

    st.write(
        "GRU is a recurrent neural network architecture that "
        "uses gating mechanisms to learn sequential dependencies."
    )

    st.markdown("**Architecture Flow**")

    gru_steps = [
        ("Input Layer", "30 time steps × 5 features"),
        ("GRU Layer", "64 units"),
        ("Dropout", "0.2"),
        ("Dense Layer", "32 neurons · ReLU"),
        ("Output Layer", "1 neuron · SOC prediction")
    ]

    for i, (name, description) in enumerate(gru_steps):

        st.markdown(
            f"""
            <div style="
                border:1px solid #0D9488;
                border-radius:10px;
                padding:12px;
                margin:5px 0;
                text-align:center;
            ">
                <strong>{name}</strong><br>
                {description}
            </div>
            """,
            unsafe_allow_html=True
        )

        if i < len(gru_steps) - 1:
            st.markdown(
                "<div style='text-align:center;'>↓</div>",
                unsafe_allow_html=True
            )

    with st.expander("How GRU works"):

        st.write("""
        GRU uses two main gates:

        - **Update Gate:** Controls how much previous information
          should be retained.
        - **Reset Gate:** Controls how much previous information
          should be ignored when processing new input.

        GRU combines the memory and output mechanisms into
        a simpler architecture than LSTM.
        """)

    st.divider()

    # ---------------------------------
    # 4. BiLSTM ARCHITECTURE
    # ---------------------------------

    st.subheader("3. Bidirectional Long Short-Term Memory (BiLSTM)")

    st.write(
        "BiLSTM processes sequential information in both forward "
        "and backward directions to learn information from both "
        "temporal contexts."
    )

    st.markdown("**Architecture Flow**")

    bilstm_steps = [
        ("Input Layer", "30 time steps × 5 features"),
        ("Forward LSTM", "64 units"),
        ("Backward LSTM", "64 units"),
        ("Bidirectional Output", "128 combined features"),
        ("Dropout", "0.2"),
        ("Dense Layer", "32 neurons · ReLU"),
        ("Output Layer", "1 neuron · SOC prediction")
    ]

    for i, (name, description) in enumerate(bilstm_steps):

        st.markdown(
            f"""
            <div style="
                border:1px solid #16A34A;
                border-radius:10px;
                padding:12px;
                margin:5px 0;
                text-align:center;
            ">
                <strong>{name}</strong><br>
                {description}
            </div>
            """,
            unsafe_allow_html=True
        )

        if i < len(bilstm_steps) - 1:
            st.markdown(
                "<div style='text-align:center;'>↓</div>",
                unsafe_allow_html=True
            )

    with st.expander("How BiLSTM works"):

        st.write("""
        BiLSTM contains two LSTM directions:

        - **Forward Direction:** Processes the sequence from
          the beginning toward the end.
        - **Backward Direction:** Processes the sequence in
          the opposite direction.

        Their outputs are concatenated before being passed
        to the subsequent layers.

        In your configuration, each direction has 64 units,
        resulting in 128 combined features.
        """)

    st.divider()

    # ---------------------------------
    # 5. MODEL COMPARISON
    # ---------------------------------

    st.subheader("Architecture Comparison")

    architecture_comparison = pd.DataFrame({
        "Parameter": [
            "Input Shape",
            "Recurrent Layer",
            "Recurrent Units",
            "Dropout",
            "Dense Layer",
            "Output"
        ],
        "LSTM": [
            "30 × 5",
            "LSTM",
            "64",
            "0.2",
            "32 · ReLU",
            "1 SOC value"
        ],
        "GRU": [
            "30 × 5",
            "GRU",
            "64",
            "0.2",
            "32 · ReLU",
            "1 SOC value"
        ],
        "BiLSTM": [
            "30 × 5",
            "Bidirectional LSTM",
            "64 per direction",
            "0.2",
            "32 · ReLU",
            "1 SOC value"
        ]
    })

    st.dataframe(
        architecture_comparison,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------------------------
    # 6. TRAINING CONFIGURATION
    # ---------------------------------

    st.subheader("Model Training Configuration")

    training_config = pd.DataFrame({
        "Parameter": [
            "Optimizer",
            "Loss Function",
            "Evaluation Metric",
            "Maximum Epochs",
            "Early Stopping Patience",
            "Batch Size",
            "Sequence Length",
            "Sequence Stride"
        ],
        "Configuration": [
            "Adam",
            "Mean Squared Error (MSE)",
            "Mean Absolute Error (MAE)",
            "5",
            "2",
            "256",
            "30",
            "5"
        ]
    })

    st.dataframe(
        training_config,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------------------------
    # 7. COMPLETE PREDICTION WORKFLOW
    # ---------------------------------

    st.subheader("Complete SOC Prediction Workflow")

    workflow_steps = [
        "Battery Time-Series Input",
        "30 × 5 Input Sequence",
        "LSTM / GRU / BiLSTM Model",
        "Dropout Regularization",
        "Dense Layer with ReLU",
        "SOC Output",
        "Performance Evaluation"
    ]

    for i, step in enumerate(workflow_steps):

        st.markdown(
            f"""
            <div style="
                background-color:#E8F1FC;
                color:#17365D;
                border-radius:8px;
                padding:12px;
                text-align:center;
                margin:5px 0;
                font-weight:600;
            ">
                {i + 1}. {step}
            </div>
            """,
            unsafe_allow_html=True
        )

        if i < len(workflow_steps) - 1:
            st.markdown(
                "<div style='text-align:center;'>↓</div>",
                unsafe_allow_html=True
            )

    st.success(
        "All three architectures were developed and evaluated "
        "for lithium-ion battery SOC estimation."
    )

    st.caption(
        "Architecture descriptions reflect the configurations "
        "used in this project's model training workflow."
    )


# =========================================
# PAGE 7: TECHNICAL REPORT
# =========================================

elif page == "Technical Report":

    st.header("Technical Report")

    st.write(
        "Access the finalized industry-style technical evaluation report "
        "for the EV Battery Intelligence System."
    )

    st.divider()

    # -------------------------------------------------
    # FINALIZED TECHNICAL REPORT V3
    # -------------------------------------------------

    st.subheader("Final Technical Evaluation Report")

    final_report_path = (
        REPORTS_DIR / "EV_Battery_Intelligence_Technical_Report_V3.pdf"
    )

    if final_report_path.exists():

        pdf_data = final_report_path.read_bytes()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.metric("Report Version", "V3 – Final")
        with col2:
            st.metric("File Size", f"{len(pdf_data) / 1024:.1f} KB")

        st.download_button(
            label="Download Final Technical Report (PDF)",
            data=pdf_data,
            file_name="EV_Battery_Intelligence_Technical_Report_V3.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        with st.expander("Preview Technical Report", expanded=True):
            if hasattr(st, "pdf"):
                st.pdf(pdf_data)
            else:
                st.info(
                    "PDF preview is not supported by this Streamlit version. "
                    "Please use the download button to view the report."
                )

        st.caption(
            "This is the finalized V3 report. The separate report generator "
            "below remains available for generating a report from the current "
            "CSV files in the reports folder."
        )

    else:
        st.warning(
            "Final V3 PDF not found. Copy "
            "'EV_Battery_Intelligence_Technical_Report_V3.pdf' "
            "into the project's reports folder to enable download and preview."
        )

    st.divider()

    st.subheader("Generate a Report from Current Project Files")

    st.write(
        "Optionally generate a fresh PDF using the latest validation, "
        "independent test and SOC prediction files currently available "
        "in the reports folder."
    )

    st.divider()

    st.subheader("Report Contents")

    report_contents = [
        "Executive Summary",
        "System Overview and Dataset Details",
        "LSTM, GRU and BiLSTM Architectures",
        "Validation Performance",
        "Independent Test Evaluation",
        "SOC Prediction Analysis",
        "Performance Visualizations",
        "Key Findings and Limitations",
        "Conclusion"
    ]

    for item in report_contents:
        st.markdown(f"✓ {item}")

    st.divider()

    st.subheader("Data Availability")

    model_results = load_csv("model_comparison_results.csv")
    test_results = load_csv("independent_test_results.csv")
    overall_results = load_csv("overall_test_results.csv")

    prediction_path = REPORTS_DIR / "bilstm_soc_predictions.csv"

    predictions = None

    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Validation Results",
        "Available" if model_results is not None else "Missing"
    )

    col2.metric(
        "Independent Test Results",
        "Available" if test_results is not None else "Missing"
    )

    col3.metric(
        "Overall Metrics",
        "Available" if overall_results is not None else "Missing"
    )

    col4.metric(
        "SOC Predictions",
        f"{len(predictions):,}" if predictions is not None else "Missing"
    )

    st.divider()

    if st.button(
        "Generate PDF from Current Project Data",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Generating your technical report..."):

                pdf_data = generate_technical_report(
                    model_results,
                    test_results,
                    overall_results,
                    predictions
                )

                st.session_state["technical_report_pdf"] = pdf_data

            st.success("Your technical report has been generated!")

        except Exception as error:

            st.error(
                f"Report generation failed: {error}"
            )

    if "technical_report_pdf" in st.session_state:

        st.download_button(
            label="Download Dynamically Generated Report",
            data=st.session_state["technical_report_pdf"],
            file_name="EV_Battery_Intelligence_Dynamic_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.caption(
            "The PDF includes the available project results "
            "and visualizations from your reports folder."
        )

# =========================================
# FOOTER
# =========================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        "**🔋 EV Battery Intelligence System**"
    )

    st.caption(
        "AI-Powered Battery State of Charge Estimation"
    )

with col2:

    st.markdown(
        "**Developed by Khushbu Uttam Shelake**"
    )

    st.caption(
        "LSTM • GRU • BiLSTM | Deep Learning Research Project"
    )

st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:12px;">
        Built with Python, TensorFlow and Streamlit
        <br>
        © 2026 EV Battery Intelligence System
    </div>
    """,
    unsafe_allow_html=True
)
