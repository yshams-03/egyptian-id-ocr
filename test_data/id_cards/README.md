# ID card test dataset (local only — gitignored)

**Do not commit real ID photos, ground-truth JSON, or reports.**  
`test_data/id_cards/`, `runs/test/`, and `runs/test_reports/` are in `.gitignore`.

## Layout

```
test_data/id_cards/
  front_001.jpg
  front_001.json              # same stem, or ground_truth/front_001.json
  front_001_back.jpg          # optional back (or set back_image in JSON)
  ground_truth.template.csv
```

## Ground truth JSON (full schema)

```json
{
  "first_name": "مصطفى",
  "last_name": "عاطف عبدالله ابراهيم",
  "full_name": "مصطفى عاطف عبدالله ابراهيم",
  "address": "ش سعد زغلول مركز منيا القمح - الشرقية",
  "national_id": "29611091301456",
  "dob": "1996-11-09",
  "serial": "GC9412479",
  "job": "",
  "religion": "",
  "expiry_date": "",
  "back_nid": "",
  "decoded_birth_date": "1996-11-09",
  "decoded_governorate": "Ash Sharqia",
  "decoded_gender": "Male",
  "back_image": "front_001_back.jpg",
  "tags": ["good_quality", "real_card"],
  "notes": ""
}
```

Enter **`national_id` once** — templates auto-fill `dob` and `decoded_*` via `egypt_nid_decode.py`.

## Edge-case tags (for grouped reports)

| Tag | Use for |
|-----|---------|
| `blurry`, `low_resolution` | Poor photo quality |
| `rotated`, `skewed` | Test with `--auto-card-crop` (National-ID-7) |
| `glare`, `partial_occlusion` | Finger/glare on card |
| `old_layout`, `new_layout` | Card format differences |
| `compound_name` | عبد الرحمن, أبو بكر, … |
| `multiline_address` | 2–3 line addresses |
| `front_only` / `front_and_back` | Single vs merged extraction |

## Synthetic fixtures (no real PII)

Generate fake watermarked cards locally (still gitignored when written here):

```powershell
py -m tests.synthetic.generate --count 5 --tags rotated,blurry
py -m tests.synthetic.generate --count 1 --tags glare --layout new --no-back
```

Synthetic NIDs use governorate code **99** (`Synthetic Test` in `egypt_nid_decode.py`) — not a real civil-registry code. Real code **88** is `Foreign` and is not used for synthetics.


```powershell
cd C:\Users\yassi\Downloads\dataset

# JSON templates for every image missing ground truth
py -m tests.run_suite --generate-json-templates

# CSV template for bulk labeling
py -m tests.run_suite --generate-template

# Import filled CSV → ground_truth/*.json
py -m tests.run_suite --import-csv test_data\id_cards\ground_truth.template.csv

# Full suite → runs/test/report_<timestamp>.md
py -m tests.run_suite --data-dir test_data\id_cards

# Safer report if sharing (redacts 14-digit NIDs)
py -m tests.run_suite --data-dir test_data\id_cards --redact

# Pytest — fast unit tests
py -m pytest tests/test_id_metrics.py -q

# Pytest — end-to-end + per-stage (slow)
py -m pytest tests/ -m slow -q
```

## What the suite tests

| Layer | Module | Metrics |
|-------|--------|---------|
| Field detection | YOLO `Egyptian-ID-Detectr-3` | Required labels present |
| Front OCR | `extract_front` / `extract_name_address` | name, address CER; NID/DOB/serial exact |
| NID decode | `egypt_nid_decode.py` | decoded_* vs ground truth |
| Back | `extract_back.py` | job, religion, expiry, back_nid |
| Cross-check | `tests/nid_validate.py` | printed DOB vs NID decode (separate failure) |

## Built-in regression

`tests/fixtures/ground_truth/15-2-....json` — public Roboflow test image (no local PII folder required).

## Privacy

- Restrict folder permissions on shared machines.
- Never push `runs/test/report_*.md` to public repos without `--redact`.
- Prefer synthetic/fake IDs for bulk edge-case testing when possible.
