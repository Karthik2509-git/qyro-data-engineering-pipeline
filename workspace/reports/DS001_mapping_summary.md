# DS001 Semantic Mapping Summary Report

This report summarizes the results of applying the approved medical class mapping policy to DS001.

---

## Mapping Metrics
- **Retained Acne Bounding Boxes**: 110615 (93.2% of total)
- **Excluded Bounding Boxes (Ignored/Rejected)**: 3711
- **Pending Review Bounding Boxes**: 4367

---

## Action Mappings Profile
- **KEEP_AS_ACNE**: Mapped `Acne`, `Blackhead`, `Whitehead`, `Papular`, `Purulent`, `Cystic`, and `Conglobata` to standard `acne`.
- **REVIEW**: Flagged and isolated `Milium`, `Crystanlline`, `Sebo-crystan-conglo`, and `Folliculitis` into review queue.
- **IGNORE**: Removed `Scars` and `Keloid` from yolo annotations but tracked in original metadata.
- **REJECT**: Excluded `Flat_wart` and `Syringoma` entirely.
