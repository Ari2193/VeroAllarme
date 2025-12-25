# How to View Pipeline Results

## 📁 Output Directories

### 1. Full Pipeline Results (Real Data)
**Location:** `data/full_pipeline_outputs/`

**Contents:**
- 28 annotated images (7 events × 4 stages each)
- `full_pipeline_report.txt` - Detailed text report
- `full_pipeline_results.json` - Machine-readable results

**Images per event:**
- `*_stage1.jpg` - Stage 1: Motion Detection (green boxes)
- `*_stage3.jpg` - Stage 3: HeatMap Analysis (cyan boxes)
- `*_stage4.jpg` - Stage 4: Memory-Based Detection (magenta boxes)
- `*_final.jpg` - Final Decision (yellow boxes)

### 2. Test Results (Synthetic Data)
**Location:** `data/test_outputs/full_pipeline/`

**Contents:**
- Test visualizations from unit tests
- Example images showing each stage

---

## 🖼️ View Images

### Command Line
```bash
# List all images
ls -lh data/full_pipeline_outputs/*.jpg

# View specific event (replace with your image viewer)
xdg-open data/full_pipeline_outputs/Factory_20251222_035349_final.jpg

# View all images in directory
xdg-open data/full_pipeline_outputs/
```

### File Browser
Navigate to:
```
/home/force-badname/hakaton/VeroAllarme/data/full_pipeline_outputs/
```

---

## �� View Reports

### Text Report
```bash
# View full report
cat data/full_pipeline_outputs/full_pipeline_report.txt

# View first few events
head -100 data/full_pipeline_outputs/full_pipeline_report.txt

# Search for specific location
grep -A 20 "Location: Factory" data/full_pipeline_outputs/full_pipeline_report.txt
```

### JSON Report
```bash
# Pretty print JSON
python3 -m json.tool data/full_pipeline_outputs/full_pipeline_results.json

# View specific fields with jq (if installed)
jq '.results[] | {event, final_decision, stage_1_confidence}' data/full_pipeline_outputs/full_pipeline_results.json
```

---

## 🎯 Example Events

### Factory Event
```bash
ls data/full_pipeline_outputs/Factory_*
```
Files:
- `Factory_20251222_035349_stage1.jpg`
- `Factory_20251222_035349_stage3.jpg`
- `Factory_20251222_035349_stage4.jpg`
- `Factory_20251222_035349_final.jpg`

### Field Event
```bash
ls data/full_pipeline_outputs/Field_*
```

### Gate and Rode Event
```bash
ls data/full_pipeline_outputs/Gate\ and\ rode_*
```

### Trees Event
```bash
ls data/full_pipeline_outputs/Trees_*
```

---

## 📈 Quick Statistics

```bash
# Count total images
ls data/full_pipeline_outputs/*.jpg | wc -l
# Output: 28

# Count events processed
grep "Location:" data/full_pipeline_outputs/full_pipeline_report.txt | wc -l
# Output: 7

# View decision summary
head -12 data/full_pipeline_outputs/full_pipeline_report.txt
```

---

## 🔄 Re-run Pipeline

```bash
# Process more events (modify script to increase count)
python3 examples/run_full_pipeline.py

# Run tests
pytest backend/tests/test_full_pipeline_e2e.py -v
```

---

## 📝 Summary of Results

From **data/full_pipeline_outputs/full_pipeline_report.txt**:

- **Total Events:** 7
- **Motion Detected:** 7/7 (100%)
- **Final Decisions:**
  - ESCALATE: 0 (0%)
  - DROP: 7 (100%)

**Stage 1 Performance:**
- Motion confidence range: 0.136 - 1.000
- Motion regions detected: 1-4 per event

**Stage 3 Analysis:**
- Cold zones (anomalous): 3 events
- Hot zones (normal): 4 events

**Stage 4 Results:**
- All events: DROP decision
- Reason: No historical training data (0 similarity, 0 support)

---

## ✅ All Files Location

```
VeroAllarme/
├── data/
│   ├── full_pipeline_outputs/          ← MAIN RESULTS HERE
│   │   ├── *.jpg                        (28 annotated images)
│   │   ├── full_pipeline_report.txt     (detailed report)
│   │   └── full_pipeline_results.json   (machine-readable)
│   │
│   └── test_outputs/
│       └── full_pipeline/               ← TEST RESULTS
│           └── *.jpg                     (test images)
│
├── examples/
│   └── run_full_pipeline.py             ← SCRIPT TO RUN AGAIN
│
├── backend/tests/
│   └── test_full_pipeline_e2e.py        ← TESTS
│
└── FULL_PIPELINE_SUMMARY.md             ← DOCUMENTATION
```

---

**Ready to view!** All images and reports are in `data/full_pipeline_outputs/`
