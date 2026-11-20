from __future__ import annotations

from io import BytesIO

import pandas as pd
from reportlab.lib import pagesizes
from reportlab.pdfgen import canvas


def dataframe_to_excel(frame: pd.DataFrame, sheet_name: str = "Report") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()


def text_report_to_pdf(title: str, lines: list[str]) -> bytes:
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=pagesizes.letter)
    width, height = pagesizes.letter
    y = height - 54
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(48, y, title[:90])
    y -= 32
    pdf.setFont("Helvetica", 10)
    for line in lines:
        words = str(line).split()
        wrapped, current = [], ""
        for word in words:
            if len(current) + len(word) + 1 > 95:
                wrapped.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            wrapped.append(current)
        for piece in wrapped:
            if y < 48:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = height - 54
            pdf.drawString(48, y, piece)
            y -= 15
    pdf.save()
    return output.getvalue()
