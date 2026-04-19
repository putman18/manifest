# DocPipeline — AI Invoice Extraction

A self-contained Streamlit app that extracts structured data from invoice PDFs or images using Claude Sonnet. Portfolio piece demonstrating production-grade AI document processing.

---

## Extraction Schema (8 fields)

| Field | Type | Notes |
|---|---|---|
| `vendor_name` | string | Company or individual issuing the invoice |
| `invoice_date` | string | Date as written on document |
| `invoice_number` | string | Invoice ID / reference number |
| `line_items` | array | Each item: description, quantity, unit_price, amount |
| `subtotal` | number | Pre-tax total |
| `tax` | number | Tax amount (0 if none) |
| `total` | number | Final amount due |
| `payment_terms` | string | e.g. "Net 30", "Due on receipt" |

---

## Confidence Levels

Claude returns per-field confidence. Application maps:

- `high` → green + "Looks good"
- `medium` → yellow + "Double-check this field"
- `low` → red + "We're not sure — verify before using"
- `null` value → red + "Not found in document"

Confidence definition (in prompt):
- **high**: clearly visible, unambiguous text
- **medium**: present but could be misread or ambiguous
- **low**: inferred, partially visible, or uncertain

---

## Claude Prompt

```
You are an invoice data extraction system. Extract the following 8 fields from this invoice and return ONLY a valid JSON object — no explanation, no markdown.

For each field provide:
  "value": extracted value (null if not found)
  "confidence": "high", "medium", or "low"
    high = clearly visible and unambiguous
    medium = present but could be misread
    low = inferred, partial, or uncertain

Schema:
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
```

---

## Stack

- `streamlit` — UI
- `anthropic` — Claude Sonnet 4.6 via API
- `pypdf` — PDF text extraction fallback
- `Pillow` — image handling

---

## Input Handling

- PDF: encode as base64, send as `document` type to Claude API
- Image (PNG/JPG): encode as base64, send as `image` type

---

## App Flow

1. User lands → sees description + sample invoice download button
2. User uploads PDF or image
3. File sent to Claude → JSON returned
4. 8 fields displayed in a grid with confidence color coding
5. Line items shown in a table
6. Download JSON button at bottom

---

## Deployment

- Streamlit Cloud, connected to `putman18/docpipeline` GitHub repo
- Secret: `ANTHROPIC_API_KEY`
- Sample invoice bundled at `execution/sample_invoice.pdf`
