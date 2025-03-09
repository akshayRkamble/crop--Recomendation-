import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import plotly.io as pio
import numpy as np

def create_prediction_pdf(prediction_data, feature_importance_fig, crop_probabilities):
    """Generate a PDF report for crop predictions"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Crop Recommendation Report", title_style))
    story.append(Spacer(1, 20))

    # Main Prediction
    story.append(Paragraph(f"Recommended Crop: {prediction_data['prediction'].title()}", styles["Heading2"]))
    story.append(Spacer(1, 10))

    # Input Parameters Table
    param_data = [["Parameter", "Value"]]
    for param, value in prediction_data['parameters'].items():
        param_data.append([param, f"{value:.2f}"])

    table = Table(param_data, colWidths=[200, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Alternative Crops
    story.append(Paragraph("Alternative Crop Recommendations:", styles["Heading3"]))
    for crop, prob in crop_probabilities:
        story.append(Paragraph(f"• {crop.title()}: {prob:.1f}% confidence", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Save and return PDF
    doc.build(story)
    pdf_output = buffer.getvalue()
    buffer.close()
    return pdf_output
