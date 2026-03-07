from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


class ReporterAgent:

    def __init__(self, analyzer_summary, forecast_values, recommendations):
        self.summary = analyzer_summary
        self.forecast = forecast_values
        self.recommendations = recommendations

    def generate_pdf_report(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/HVAC_Report_{timestamp}.pdf"

        doc = SimpleDocTemplate(filename)
        elements = []
        styles = getSampleStyleSheet()

        # ===============================
        # TITLE
        # ===============================
        elements.append(Paragraph("<b>HVAC AI Optimization Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"Generated on: {datetime.now()}", styles["Normal"]))
        elements.append(Spacer(1, 0.4 * inch))

        # ===============================
        # EXECUTIVE SUMMARY
        # ===============================
        elements.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        for key, value in self.summary.items():
            elements.append(Paragraph(f"{key}: {value}", styles["Normal"]))

        elements.append(Spacer(1, 0.4 * inch))

        # ===============================
        # FORECAST TABLE
        # ===============================
        elements.append(Paragraph("<b>24-Hour Load Forecast (kW)</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        forecast_table_data = [["Hour", "Predicted Load (kW)"]]

        for i, val in enumerate(self.forecast):
            forecast_table_data.append([f"{i+1}", round(float(val), 2)])

        table = Table(forecast_table_data, colWidths=[1.2 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER")
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.4 * inch))

        # ===============================
        # RECOMMENDATIONS
        # ===============================
        elements.append(Paragraph("<b>Optimization Recommendations</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        rec_list = [ListItem(Paragraph(r, styles["Normal"])) for r in self.recommendations]
        elements.append(ListFlowable(rec_list, bulletType="bullet"))

        doc.build(elements)

        return filename