"""
Manifest — AI Document Data Extraction
Upload any business document. Claude extracts structured fields with per-field confidence.
Exports: JSON, CSV, Email Draft, PDF Summary.
"""

import os
import io
import csv
import json
import base64
from pathlib import Path

import streamlit as st
import anthropic
import pandas as pd

st.set_page_config(
    page_title="Manifest — AI Document Extraction",
    page_icon="📄",
    layout="wide"
)

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
ASSETS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Document type definitions
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = {
    "Invoice": {
        "description": "Supplier invoices, vendor bills",
        "sample": "sample_invoice.pdf",
        "array_fields": ["line_items"],
        "fields": {
            "vendor_name":    "Vendor Name",
            "invoice_date":   "Invoice Date",
            "invoice_number": "Invoice Number",
            "subtotal":       "Subtotal",
            "tax":            "Tax",
            "total":          "Total Due",
            "payment_terms":  "Payment Terms",
            "line_items":     "Line Items",
        },
        "prompt": """Extract these fields from the invoice and return ONLY valid JSON — no markdown, no explanation.

Each field: {"value": <extracted or null>, "confidence": "high"|"medium"|"low"}
  high = clearly visible and unambiguous
  medium = present but could be misread
  low = inferred, partial, or uncertain

{
  "vendor_name": {"value": null, "confidence": "high"},
  "invoice_date": {"value": null, "confidence": "high"},
  "invoice_number": {"value": null, "confidence": "high"},
  "subtotal": {"value": null, "confidence": "high"},
  "tax": {"value": null, "confidence": "high"},
  "total": {"value": null, "confidence": "high"},
  "payment_terms": {"value": null, "confidence": "high"},
  "line_items": {"value": [], "confidence": "high"}
}

line_items array items: {"description": str, "quantity": num, "unit_price": num, "amount": num}
Return numbers (not strings) for all monetary and quantity fields.""",
    },

    "Purchase Order": {
        "description": "POs sent to suppliers",
        "sample": None,
        "array_fields": ["line_items"],
        "fields": {
            "buyer_name":      "Buyer / Company",
            "vendor_name":     "Vendor Name",
            "po_number":       "PO Number",
            "order_date":      "Order Date",
            "delivery_date":   "Requested Delivery",
            "subtotal":        "Subtotal",
            "tax":             "Tax",
            "shipping":        "Shipping",
            "total":           "Total",
            "payment_terms":   "Payment Terms",
            "line_items":      "Line Items",
        },
        "prompt": """Extract these fields from the purchase order and return ONLY valid JSON.

Each field: {"value": <extracted or null>, "confidence": "high"|"medium"|"low"}

{
  "buyer_name": {"value": null, "confidence": "high"},
  "vendor_name": {"value": null, "confidence": "high"},
  "po_number": {"value": null, "confidence": "high"},
  "order_date": {"value": null, "confidence": "high"},
  "delivery_date": {"value": null, "confidence": "high"},
  "subtotal": {"value": null, "confidence": "high"},
  "tax": {"value": null, "confidence": "high"},
  "shipping": {"value": null, "confidence": "high"},
  "total": {"value": null, "confidence": "high"},
  "payment_terms": {"value": null, "confidence": "high"},
  "line_items": {"value": [], "confidence": "high"}
}

line_items: {"description": str, "quantity": num, "unit_price": num, "amount": num}
Return numbers for all monetary and quantity fields.""",
    },

    "Contract": {
        "description": "Service agreements, NDAs, leases",
        "sample": None,
        "array_fields": ["parties", "key_obligations"],
        "fields": {
            "parties":            "Parties",
            "effective_date":     "Effective Date",
            "expiration_date":    "Expiration Date",
            "contract_value":     "Contract Value",
            "payment_terms":      "Payment Terms",
            "governing_law":      "Governing Law",
            "termination_notice": "Termination Notice",
            "key_obligations":    "Key Obligations",
        },
        "prompt": """Extract these fields from the contract and return ONLY valid JSON.

Each field: {"value": <extracted or null>, "confidence": "high"|"medium"|"low"}

{
  "parties": {"value": [], "confidence": "high"},
  "effective_date": {"value": null, "confidence": "high"},
  "expiration_date": {"value": null, "confidence": "high"},
  "contract_value": {"value": null, "confidence": "high"},
  "payment_terms": {"value": null, "confidence": "high"},
  "governing_law": {"value": null, "confidence": "high"},
  "termination_notice": {"value": null, "confidence": "high"},
  "key_obligations": {"value": [], "confidence": "high"}
}

parties: list of strings (party names)
key_obligations: list of strings (one obligation per item, plain English)
contract_value: string (e.g. "$50,000/year" or null if not stated)""",
    },

    "Receipt": {
        "description": "Purchase receipts, expense receipts",
        "sample": None,
        "array_fields": ["line_items"],
        "fields": {
            "vendor_name":    "Vendor / Store",
            "date":           "Date",
            "subtotal":       "Subtotal",
            "tax":            "Tax",
            "total":          "Total",
            "payment_method": "Payment Method",
            "line_items":     "Items Purchased",
        },
        "prompt": """Extract these fields from the receipt and return ONLY valid JSON.

Each field: {"value": <extracted or null>, "confidence": "high"|"medium"|"low"}

{
  "vendor_name": {"value": null, "confidence": "high"},
  "date": {"value": null, "confidence": "high"},
  "subtotal": {"value": null, "confidence": "high"},
  "tax": {"value": null, "confidence": "high"},
  "total": {"value": null, "confidence": "high"},
  "payment_method": {"value": null, "confidence": "high"},
  "line_items": {"value": [], "confidence": "high"}
}

line_items: {"description": str, "quantity": num, "unit_price": num, "amount": num}
Return numbers for all monetary fields.""",
    },

    "Job Application": {
        "description": "CVs, resumes, application forms",
        "sample": None,
        "array_fields": ["skills", "experience"],
        "fields": {
            "full_name":        "Full Name",
            "email":            "Email",
            "phone":            "Phone",
            "position_applied": "Position Applied For",
            "years_experience": "Years of Experience",
            "education":        "Highest Education",
            "availability":     "Availability / Start Date",
            "skills":           "Skills",
            "experience":       "Work Experience",
        },
        "prompt": """Extract these fields from the job application or CV and return ONLY valid JSON.

Each field: {"value": <extracted or null>, "confidence": "high"|"medium"|"low"}

{
  "full_name": {"value": null, "confidence": "high"},
  "email": {"value": null, "confidence": "high"},
  "phone": {"value": null, "confidence": "high"},
  "position_applied": {"value": null, "confidence": "high"},
  "years_experience": {"value": null, "confidence": "high"},
  "education": {"value": null, "confidence": "high"},
  "availability": {"value": null, "confidence": "high"},
  "skills": {"value": [], "confidence": "high"},
  "experience": {"value": [], "confidence": "high"}
}

skills: list of strings
experience: list of {"company": str, "title": str, "duration": str}""",
    },
}

CONFIDENCE_CONFIG = {
    "high":   {"color": "#22c55e", "bg": "#052e16", "label": "Looks good"},
    "medium": {"color": "#eab308", "bg": "#422006", "label": "Double-check this field"},
    "low":    {"color": "#ef4444", "bg": "#450a0a", "label": "Verify before using"},
    "null":   {"color": "#6b7280", "bg": "#111827", "label": "Not found"},
}


# ---------------------------------------------------------------------------
# Claude extraction
# ---------------------------------------------------------------------------

def extract(file_bytes: bytes, media_type: str, doc_type: str) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    encoded = base64.standard_b64encode(file_bytes).decode("utf-8")
    prompt = DOCUMENT_TYPES[doc_type]["prompt"]

    content_block = (
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": encoded}}
        if media_type == "application/pdf"
        else {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}}
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": [content_block, {"type": "text", "text": prompt}]}]
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def to_csv(result: dict, doc_type: str) -> str:
    cfg = DOCUMENT_TYPES[doc_type]
    array_fields = cfg["array_fields"]
    field_labels = cfg["fields"]
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Scalar fields
    writer.writerow(["Field", "Value", "Confidence"])
    for key, label in field_labels.items():
        if key in array_fields:
            continue
        fd = result.get(key, {})
        writer.writerow([label, fd.get("value", ""), fd.get("confidence", "")])

    # Array fields
    for key in array_fields:
        if key not in result:
            continue
        fd = result[key]
        items = fd.get("value") or []
        if not items:
            continue
        writer.writerow([])
        writer.writerow([field_labels.get(key, key)])
        if items and isinstance(items[0], dict):
            writer.writerow(list(items[0].keys()))
            for item in items:
                writer.writerow(list(item.values()))
        else:
            for item in items:
                writer.writerow([item])

    return buf.getvalue()


def to_email(result: dict, doc_type: str, filename: str) -> str:
    cfg = DOCUMENT_TYPES[doc_type]
    array_fields = cfg["array_fields"]
    field_labels = cfg["fields"]
    lines = [
        f"Subject: Extracted data from {filename} — please review",
        "",
        f"Hi,",
        "",
        f"Manifest has extracted the following data from the {doc_type.lower()} '{filename}'.",
        "Please review the fields marked for verification before using this data.",
        "",
        "--- EXTRACTED FIELDS ---",
        "",
    ]
    for key, label in field_labels.items():
        if key in array_fields:
            continue
        fd = result.get(key, {})
        value = fd.get("value", "Not found")
        confidence = fd.get("confidence", "null")
        flag = " ⚠️ Please verify" if confidence in ("low", "medium") else ""
        lines.append(f"{label}: {value}{flag}")

    for key in array_fields:
        fd = result.get(key, {})
        items = fd.get("value") or []
        if not items:
            continue
        lines.append(f"\n{field_labels.get(key, key)}:")
        if isinstance(items[0], dict):
            for item in items:
                lines.append("  • " + "  |  ".join(f"{k}: {v}" for k, v in item.items()))
        else:
            for item in items:
                lines.append(f"  • {item}")

    lines += [
        "",
        "--- END OF EXTRACTION ---",
        "",
        "Extracted by Manifest (manifest.streamlit.app)",
    ]
    return "\n".join(lines)


def to_pdf(result: dict, doc_type: str, filename: str) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    cfg = DOCUMENT_TYPES[doc_type]
    array_fields = cfg["array_fields"]
    field_labels = cfg["fields"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            topMargin=0.75*inch, bottomMargin=0.75*inch,
                            leftMargin=inch, rightMargin=inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("title", fontSize=20, fontName="Helvetica-Bold", spaceAfter=4)
    sub_style   = ParagraphStyle("sub",   fontSize=10, fontName="Helvetica", textColor=colors.grey, spaceAfter=16)
    label_style = ParagraphStyle("lbl",   fontSize=8,  fontName="Helvetica-Bold", textColor=colors.grey, spaceAfter=2)
    value_style = ParagraphStyle("val",   fontSize=11, fontName="Helvetica", spaceAfter=12)

    story.append(Paragraph("Manifest", title_style))
    story.append(Paragraph(f"{doc_type} extraction — {filename}", sub_style))

    conf_colors = {"high": colors.HexColor("#22c55e"), "medium": colors.HexColor("#eab308"),
                   "low": colors.HexColor("#ef4444"), "null": colors.grey}

    for key, label in field_labels.items():
        if key in array_fields:
            continue
        fd = result.get(key, {})
        value = str(fd.get("value") or "Not found")
        confidence = fd.get("confidence", "null")
        story.append(Paragraph(label.upper(), label_style))
        story.append(Paragraph(value, value_style))

    for key in array_fields:
        fd = result.get(key, {})
        items = fd.get("value") or []
        if not items:
            continue
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(field_labels.get(key, key).upper(), label_style))
        if isinstance(items[0], dict):
            headers = list(items[0].keys())
            table_data = [headers] + [[str(item.get(h, "")) for h in headers] for item in items]
            t = Table(table_data, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 9),
                ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
                ("TOPPADDING", (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ]))
            story.append(t)
        else:
            for item in items:
                story.append(Paragraph(f"- {item}", value_style))
        story.append(Spacer(1, 0.1*inch))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def confidence_badge(level: str) -> str:
    cfg = CONFIDENCE_CONFIG.get(level, CONFIDENCE_CONFIG["null"])
    return (
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};'
        f'border:1px solid {cfg["color"]}33;border-radius:999px;'
        f'padding:2px 10px;font-size:11px;font-weight:600;">'
        f'{cfg["label"]}</span>'
    )


def render_field(label: str, value, confidence: str):
    cfg = CONFIDENCE_CONFIG.get(confidence, CONFIDENCE_CONFIG["null"])
    display = str(value) if value is not None else "—"
    st.markdown(
        f"""<div style="border:1px solid {cfg['color']}44;border-radius:12px;
                    padding:14px 18px;background:{cfg['bg']};margin-bottom:10px;">
          <div style="font-size:11px;color:#6b7280;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:4px;">{label}</div>
          <div style="font-size:16px;font-weight:700;color:#f4f4f5;
                      margin-bottom:8px;">{display}</div>
          {confidence_badge(confidence)}
        </div>""",
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title("Manifest")
st.caption("Upload any business document — Claude extracts structured data with per-field confidence.")

if not ANTHROPIC_API_KEY:
    st.error("ANTHROPIC_API_KEY not set.")
    st.stop()

with st.sidebar:
    st.header("Document type")
    doc_type = st.selectbox(
        "Select document type",
        list(DOCUMENT_TYPES.keys()),
        label_visibility="collapsed"
    )
    st.caption(DOCUMENT_TYPES[doc_type]["description"])
    st.divider()

    sample = DOCUMENT_TYPES[doc_type]["sample"]
    if sample:
        sample_path = ASSETS_DIR / sample
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button(
                    f"Download sample {doc_type.lower()}",
                    f,
                    file_name=sample,
                    mime="application/pdf",
                    use_container_width=True
                )
    st.caption("Supports PDF, PNG, JPG.")

uploaded = st.file_uploader(
    "Upload document",
    type=["pdf", "png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if uploaded:
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    media_type_map = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    media_type = media_type_map.get(ext, "application/pdf")

    with st.spinner(f"Extracting {doc_type.lower()} fields..."):
        try:
            result = extract(uploaded.read(), media_type, doc_type)
        except json.JSONDecodeError as e:
            st.error(f"Claude returned invalid JSON: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()

    st.success(f"Extracted from **{uploaded.name}**")
    st.divider()

    cfg = DOCUMENT_TYPES[doc_type]
    array_fields = cfg["array_fields"]
    field_labels = cfg["fields"]

    # Scalar fields
    scalar_keys = [k for k in field_labels if k not in array_fields]
    cols = st.columns(2)
    for i, key in enumerate(scalar_keys):
        fd = result.get(key, {"value": None, "confidence": "null"})
        value = fd.get("value")
        confidence = fd.get("confidence") if value is not None else "null"
        with cols[i % 2]:
            render_field(field_labels[key], value, confidence)

    # Array fields
    for key in array_fields:
        fd = result.get(key, {"value": [], "confidence": "null"})
        items = fd.get("value") or []
        conf = fd.get("confidence", "null")
        st.divider()
        st.subheader(field_labels.get(key, key), anchor=False)
        st.markdown(confidence_badge(conf), unsafe_allow_html=True)
        if items:
            if isinstance(items[0], dict):
                st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
            else:
                for item in items:
                    st.markdown(f"- {item}")
        else:
            st.info("None found.")

    # Exports
    st.divider()
    st.subheader("Export", anchor=False)
    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.download_button(
            "JSON",
            json.dumps(result, indent=2),
            file_name=f"{uploaded.name.rsplit('.',1)[0]}.json",
            mime="application/json",
            use_container_width=True,
            type="primary"
        )
    with e2:
        st.download_button(
            "CSV",
            to_csv(result, doc_type),
            file_name=f"{uploaded.name.rsplit('.',1)[0]}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
    with e3:
        st.download_button(
            "Email Draft",
            to_email(result, doc_type, uploaded.name),
            file_name=f"{uploaded.name.rsplit('.',1)[0]}_email.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary"
        )
    with e4:
        st.download_button(
            "PDF Summary",
            to_pdf(result, doc_type, uploaded.name),
            file_name=f"{uploaded.name.rsplit('.',1)[0]}_summary.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
