# Deduplication Audit Report: DS005

*Generated on: 2026-07-02 12:10:12*

## Executive Summary
Completed exact and near-duplicate matching for dataset **DS005**.

## Deduplication Statistics
- **Dataset Evaluated**: DS005
- **Duplicate Images Discovered & De-activated**: 81
- **Perceptual Hash Type**: dhash (Size: 8)
- **Hamming Distance Threshold**: 4


## Methodology Details
Each image is transformed into an 8x8 bit grayscale signature (dHash). A global index search checks Hamming Distance. If distance $\\le 4$, we compare DB overall scores and route the lower quality image to `duplicate` status.