# master QYRO Dataset Snapshot (v2.0)

This snapshot represents the final cumulative yields, quality metrics, and licensing audit logs of the compiled **QYRO Dataset v2** candidates pool before the final freeze.

---

## 📈 Platform Specifications

- **Factory Version**: `1.0`
- **Calibration Version**: `T45`
- **Date Generated**: 2026-07-07T11:57:00
- **Latest Merged Fingerprint**: `bb93c834ee8dc7fcb52ae4a03d8380c21c991b1466646d3a5d9e2dbdc39db0bd`
- **Datasets Processed**: `DS001, DS002, DS003, DS004, DS005`

---

## 📊 Global Pool Quality Yields

- **Total Exported Accepted Images**: `835`
- **Total Images Routed to Review**: `13985`
- **Total Images Hard Rejected**: `256`
- **Total Duplicates Identified**: `1577`
- **Total Bounding Boxes (Acne Lesions)**: `5561`
- **Average Lesions per Image**: `6.66`
- **Cumulative Merged Storage Size**: `64.12 MB`

---

## 🌐 Dataset Contribution Summary

| Dataset ID | Total Ingested | Accepted Candidates | Review Queue | Hard Rejected | Duplicates | Total BBoxes | Yield % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DS001 | 8481 | 274 | 7235 | 29 | 943 | 1824 | 3.23% |
| DS002 | 4409 | 149 | 3995 | 4 | 261 | 867 | 3.38% |
| DS003 | 1130 | 1 | 853 | 223 | 53 | 1 | 0.09% |
| DS004 | 1389 | 226 | 1084 | 0 | 79 | 1741 | 16.27% |
| DS005 | 1244 | 185 | 818 | 0 | 241 | 1128 | 14.87% |

---

## 🟢 Curation Validation Status

All candidate images and annotations conform to the **QYRO Dataset v2 Acceptance Policy (Constitution)**. Programmer-level verification tests (PIL decode validation, split pairing logic, out-of-bounds bounding boxes check, and split cross-leakage) have passed successfully.
