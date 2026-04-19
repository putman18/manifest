"""
DocPipeline — AI Invoice Data Extraction
Upload an invoice PDF or image. Claude extracts 8 structured fields with per-field confidence.
"""

import os
import json
import base64
from pathlib import Path

import streamlit as st
import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocPipeline — AI Invoice Extraction",
    page_icon="📄",
    layout="wide"
)

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
SAMPLE_INVOICE_PATH = Path(__file__).parent / "sample_invoice.pdf"

CONFIDENCE_CONFIG = {
    "high":   {"color": "#22c55e", "bg": "#052e16", "label": "Looks good"},
    "medium": {"color": "#eab308", "bg": "#422006", "label": "Double-check this field"},
    "low":    {"color": "#ef4444", "bg": "#450a0a", "label": "We're not sure — verify before using"},
    "null":   {"color": "#6b7280", "bg": "#111827", "label": "Not found in document"},
}

FIELD_LABELS = {
    "vendor_name":    "Vendor Name",
    "invoice_date":   "Invoice Date",
    "invoice_number": "Invoice Number",
    "line_items":     "Line Items",
    "subtotal":       "Subtotal",
    "tax":            "Tax",
    "total":          "Total Due",
    "payment_terms":  "Payment Terms",
}

EXTRACT_PROMPT = """You are an invoice data extraction system. Extract the following 8 fields from this invoice and return ONLY a valid JSON object — no explanation, no markdown fences.

For each field provide:
  "value": the extracted value (null if not found)
  "confidence": "high", "medium", or "low"
    high = clearly visible and unambiguous
    medium = present but could be misread or ambiguous
    low = inferred, partially visible, or uncertain

Return exactly this schema:
{
  "vendor_name": {"value": null, "confidence": "high"},
  "invoice_date": {"value": null, "confidence": "high"},
  "invoice_number": {"value": null, "confidence": "high"},
  "line_items": {"value": [], "confidence": "high"},
  "subtotal": {"value": null, "confidence": "high"},
  "tax": {"value": null, "confidence": "high"},
  "total": {"value": null, "confidence": "high"},
  "payment_terms": {"value": null, "confidence": "high"}
}

For line_items, each item in the array should be: {"description": "...", "quantity": ..., "unit_price": ..., "amount": ...}
For numeric fields (subtotal, tax, total, unit_price, amount), return numbers not strings."""


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

def extract_invoice(file_bytes: bytes, media_type: str) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    encoded = base64.standard_b64encode(file_bytes).decode("utf-8")

    if media_type == "application/pdf":
        content_block = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": encoded}
        }
    else:
        content_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": encoded}
        }

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": [content_block, {"type": "text", "text": EXTRACT_PROMPT}]
        }]
    )

    text = response.content[0].text.strip()
    # Strip markdown fences if Claude adds them anyway
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


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
        f"""
        <div style="border:1px solid {cfg['color']}44;border-radius:12px;
                    padding:14px 18px;background:{cfg['bg']};margin-bottom:10px;">
          <div style="font-size:11px;color:#6b7280;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:4px;">{label}</div>
          <div style="font-size:16px;font-weight:700;color:#f4f4f5;
                      margin-bottom:8px;">{display}</div>
          {confidence_badge(confidence)}
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title("DocPipeline")
st.caption("Upload an invoice — Claude extracts 8 structured fields with per-field confidence.")

if not ANTHROPIC_API_KEY:
    st.error("ANTHROPIC_API_KEY not set. Add it to Streamlit secrets or your environment.")
    st.stop()

# Sample invoice download
with st.sidebar:
    st.header("Try it")
    if SAMPLE_INVOICE_PATH.exists():
        with open(SAMPLE_INVOICE_PATH, "rb") as f:
            st.download_button(
                "Download sample invoice",
                f,
                file_name="sample_invoice.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    st.caption("Download the sample, then upload it above to see extraction in action.")
    st.divider()
    st.caption("Supports PDF, PNG, JPG. One document type: invoices.")

# Upload
uploaded = st.file_uploader(
    "Upload invoice (PDF or image)",
    type=["pdf", "png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if uploaded:
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    media_type_map = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    media_type = media_type_map.get(ext, "application/pdf")

    with st.spinner("Extracting fields..."):
        try:
            result = extract_invoice(uploaded.read(), media_type)
        except json.JSONDecodeError as e:
            st.error(f"Claude returned invalid JSON: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()

    st.success(f"Extracted from **{uploaded.name}**")
    st.divider()

    # Main fields (excluding line_items)
    scalar_fields = [k for k in FIELD_LABELS if k != "line_items"]
    cols = st.columns(2)
    for i, key in enumerate(scalar_fields):
        field_data = result.get(key, {"value": None, "confidence": "null"})
        value = field_data.get("value")
        confidence = field_data.get("confidence") if value is not None else "null"
        with cols[i % 2]:
            render_field(FIELD_LABELS[key], value, confidence)

    # Line items
    st.divider()
    li_data = result.get("line_items", {"value": [], "confidence": "high"})
    li_value = li_data.get("value") or []
    li_conf = li_data.get("confidence", "null")

    st.subheader(f"Line Items  {confidence_badge(li_conf)}", anchor=False)
    st.markdown("", unsafe_allow_html=True)

    if li_value:
        import pandas as pd
        df = pd.DataFrame(li_value)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No line items found.")

    # Download JSON
    st.divider()
    st.download_button(
        "Download extracted JSON",
        json.dumps(result, indent=2),
        file_name="extracted_invoice.json",
        mime="application/json",
        use_container_width=True,
        type="primary"
    )
