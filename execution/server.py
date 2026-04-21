"""
Manifest — FastAPI backend
POST /extract        → extract fields from uploaded document
POST /export/csv     → download CSV
POST /export/email   → download email draft
POST /export/pdf     → download PDF summary
GET  /               → serve frontend
"""

import os
import io
import csv
import json
import base64
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import anthropic

app = FastAPI(title="Manifest")

BASE_DIR  = Path(__file__).parent
STATIC    = BASE_DIR / "static"
API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")

app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ---------------------------------------------------------------------------
# Document type definitions
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = {
    "invoice": {
        "label": "Invoice",
        "description": "Supplier invoices, vendor bills",
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
        "prompt": """Extract these fields from the invoice. Return ONLY valid JSON — no markdown, no explanation.

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

line_items items: {"description": str, "quantity": num, "unit_price": num, "amount": num}
Return numbers (not strings) for all monetary and quantity fields.""",
    },

    "purchase_order": {
        "label": "Purchase Order",
        "description": "POs sent to suppliers",
        "array_fields": ["line_items"],
        "fields": {
            "buyer_name":    "Buyer / Company",
            "vendor_name":   "Vendor Name",
            "po_number":     "PO Number",
            "order_date":    "Order Date",
            "delivery_date": "Requested Delivery",
            "subtotal":      "Subtotal",
            "tax":           "Tax",
            "shipping":      "Shipping",
            "total":         "Total",
            "payment_terms": "Payment Terms",
            "line_items":    "Line Items",
        },
        "prompt": """Extract these fields from the purchase order. Return ONLY valid JSON.

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

line_items: {"description": str, "quantity": num, "unit_price": num, "amount": num}""",
    },

    "contract": {
        "label": "Contract",
        "description": "Service agreements, NDAs, leases",
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
        "prompt": """Extract these fields from the contract. Return ONLY valid JSON.

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

parties: list of strings. key_obligations: list of plain-English strings.""",
    },

    "receipt": {
        "label": "Receipt",
        "description": "Purchase receipts, expense receipts",
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
        "prompt": """Extract these fields from the receipt. Return ONLY valid JSON.

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

line_items: {"description": str, "quantity": num, "unit_price": num, "amount": num}""",
    },

    "job_application": {
        "label": "Job Application",
        "description": "CVs, resumes, application forms",
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
        "prompt": """Extract these fields from the job application or CV. Return ONLY valid JSON.

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

skills: list of strings. experience: list of {"company": str, "title": str, "duration": str}""",
    },
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _extract(file_bytes: bytes, media_type: str, doc_type: str) -> dict:
    cfg = DOCUMENT_TYPES[doc_type]
    client = anthropic.Anthropic(api_key=API_KEY)
    encoded = base64.standard_b64encode(file_bytes).decode()

    block = (
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": encoded}}
        if media_type == "application/pdf"
        else {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}}
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": [block, {"type": "text", "text": cfg["prompt"]}]}]
    )
    return _parse_json(resp.content[0].text)


def _summarize(result: dict, doc_type: str) -> str:
    cfg = DOCUMENT_TYPES[doc_type]
    total = len(cfg["fields"])
    high   = sum(1 for v in result.values() if isinstance(v, dict) and v.get("confidence") == "high" and v.get("value") is not None)
    medium = sum(1 for v in result.values() if isinstance(v, dict) and v.get("confidence") == "medium")
    low    = sum(1 for v in result.values() if isinstance(v, dict) and v.get("confidence") == "low")
    missing = sum(1 for v in result.values() if isinstance(v, dict) and v.get("value") is None)

    if low + medium + missing == 0:
        return f"All {high} fields extracted with high confidence. This document is ready to use."
    issues = []
    if low:    issues.append(f"{low} field{'s' if low>1 else ''} need verification")
    if medium: issues.append(f"{medium} field{'s' if medium>1 else ''} should be double-checked")
    if missing:issues.append(f"{missing} field{'s' if missing>1 else ''} not found")
    return f"Extracted {high} of {total} fields with high confidence. " + ", ".join(issues).capitalize() + "."


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _to_csv(result: dict, doc_type: str, filename: str) -> str:
    cfg = DOCUMENT_TYPES[doc_type]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Field", "Value", "Confidence"])
    for key, label in cfg["fields"].items():
        if key in cfg["array_fields"]: continue
        fd = result.get(key, {})
        w.writerow([label, fd.get("value", ""), fd.get("confidence", "")])
    for key in cfg["array_fields"]:
        items = (result.get(key) or {}).get("value") or []
        if not items: continue
        w.writerow([])
        w.writerow([cfg["fields"].get(key, key)])
        if items and isinstance(items[0], dict):
            w.writerow(list(items[0].keys()))
            for item in items: w.writerow(list(item.values()))
        else:
            for item in items: w.writerow([item])
    return buf.getvalue()


def _to_email(result: dict, doc_type: str, filename: str) -> str:
    cfg = DOCUMENT_TYPES[doc_type]
    lines = [
        f"Subject: Extracted data from {filename} — please review", "",
        "Hi,", "",
        f"Manifest has extracted the following data from the {cfg['label'].lower()} '{filename}'.",
        "Fields marked ⚠️ should be verified before use.", "", "--- EXTRACTED FIELDS ---", "",
    ]
    for key, label in cfg["fields"].items():
        if key in cfg["array_fields"]: continue
        fd = result.get(key, {})
        value = fd.get("value", "Not found")
        flag = " ⚠️ Please verify" if fd.get("confidence") in ("low", "medium") else ""
        lines.append(f"{label}: {value}{flag}")
    for key in cfg["array_fields"]:
        items = (result.get(key) or {}).get("value") or []
        if not items: continue
        lines.append(f"\n{cfg['fields'].get(key, key)}:")
        if isinstance(items[0], dict):
            for item in items: lines.append("  • " + "  |  ".join(f"{k}: {v}" for k,v in item.items()))
        else:
            for item in items: lines.append(f"  • {item}")
    lines += ["", "--- END OF EXTRACTION ---", "", "Extracted by Manifest — manifest.app"]
    return "\n".join(lines)


def _to_pdf(result: dict, doc_type: str, filename: str) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle

    cfg = DOCUMENT_TYPES[doc_type]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch, leftMargin=inch, rightMargin=inch)
    story = []

    title_s = ParagraphStyle("t", fontSize=20, fontName="Helvetica-Bold", spaceAfter=4)
    sub_s   = ParagraphStyle("s", fontSize=10, fontName="Helvetica", textColor=colors.grey, spaceAfter=16)
    lbl_s   = ParagraphStyle("l", fontSize=8,  fontName="Helvetica-Bold", textColor=colors.grey, spaceAfter=2)
    val_s   = ParagraphStyle("v", fontSize=11, fontName="Helvetica", spaceAfter=12)

    story.append(Paragraph("Manifest", title_s))
    story.append(Paragraph(f"{cfg['label']} extraction — {filename}", sub_s))
    story.append(Paragraph(_summarize(result, doc_type), val_s))

    for key, label in cfg["fields"].items():
        if key in cfg["array_fields"]: continue
        fd = result.get(key, {})
        value = str(fd.get("value") or "Not found")
        story.append(Paragraph(label.upper(), lbl_s))
        story.append(Paragraph(value, val_s))

    for key in cfg["array_fields"]:
        items = (result.get(key) or {}).get("value") or []
        if not items: continue
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(cfg["fields"].get(key, key).upper(), lbl_s))
        if isinstance(items[0], dict):
            headers = list(items[0].keys())
            data = [headers] + [[str(item.get(h,"")) for h in headers] for item in items]
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR",(0,0),(-1,0), colors.white),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("FONTSIZE",(0,0),(-1,-1),9),
                ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#dee2e6")),
                ("TOPPADDING",(0,0),(-1,-1),6),
                ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ]))
            story.append(t)
        else:
            for item in items: story.append(Paragraph(f"- {item}", val_s))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/sample-invoice")
async def sample_invoice():
    path = BASE_DIR / "sample_invoice.pdf"
    if not path.exists():
        raise HTTPException(404, "Sample invoice not found")
    return FileResponse(path, filename="sample_invoice.pdf", media_type="application/pdf")


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
):
    if not API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured")
    if doc_type not in DOCUMENT_TYPES:
        raise HTTPException(400, f"Unknown doc_type: {doc_type}")

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    media_map = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    media_type = media_map.get(ext)
    if not media_type:
        raise HTTPException(400, "Unsupported file type. Use PDF, PNG, or JPG.")

    file_bytes = await file.read()
    try:
        result = _extract(file_bytes, media_type, doc_type)
    except json.JSONDecodeError:
        raise HTTPException(500, "Extraction failed: Claude returned invalid JSON")
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {e}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return {
        "result": result,
        "doc_type": doc_type,
        "filename": file.filename,
        "timestamp": ts,
        "summary": _summarize(result, doc_type),
        "fields": DOCUMENT_TYPES[doc_type]["fields"],
        "array_fields": DOCUMENT_TYPES[doc_type]["array_fields"],
    }


@app.post("/export/csv")
async def export_csv(body: dict):
    content = _to_csv(body["result"], body["doc_type"], body["filename"])
    name = body["filename"].rsplit(".", 1)[0] + ".csv"
    return StreamingResponse(io.StringIO(content), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/export/email")
async def export_email(body: dict):
    content = _to_email(body["result"], body["doc_type"], body["filename"])
    name = body["filename"].rsplit(".", 1)[0] + "_email.txt"
    return StreamingResponse(io.StringIO(content), media_type="text/plain",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/export/pdf")
async def export_pdf(body: dict):
    content = _to_pdf(body["result"], body["doc_type"], body["filename"])
    name = body["filename"].rsplit(".", 1)[0] + "_summary.pdf"
    return StreamingResponse(io.BytesIO(content), media_type="application/pdf",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
