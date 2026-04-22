# Plan: Max Samples Rolling Dataset for Data Prepper

## Overview
Add a `max_samples` parameter to RSLTXVPrepareDataset that caps the dataset size and enables rolling dataset management — auto-backfill from a pool of source videos when clips are deleted.

## New Parameters
- `max_samples` (INT, default=0, 0=unlimited) — Maximum clips in the dataset
- `raw_folder` (STRING, optional) — Pool of source videos to draw from. If empty, uses `media_folder` as before.

## Core Behavior

### Initial Run
1. Count existing usable clips (in clips folder, not in `rejected.json`)
2. If count < `max_samples`:
   - Pick a random unfinished video from the source pool
   - Extract clips sequentially from last known position
   - After each clip, check count against `max_samples`
   - Stop when `max_samples` reached or all videos exhausted
3. If `max_samples` is 0, process everything (current behavior)

### Re-run After Deletion
1. Detect missing clips (file existed in `dataset.json` but no longer on disk)
2. Move missing entries to `rejected.json` with chunk-level tracking
3. Count remaining usable clips
4. If count < `max_samples`, backfill:
   - Pick a random video that still has unprocessed chunks
   - Resume from last chunk position
   - Skip any chunk ranges in `rejected.json`
   - Stop at `max_samples`

### Video Progress Tracking
New file: `video_progress.json` in the output directory
```json
{
  "source_video_1.mp4": {
    "total_frames": 44053,
    "fps": 29.976,
    "last_frame": 12480,
    "status": "partial",
    "chunks_extracted": 15
  },
  "source_video_2.mp4": {
    "total_frames": 43906,
    "fps": 29.976,
    "last_frame": 43906,
    "status": "complete",
    "chunks_extracted": 22
  }
}
```
- `partial` = more chunks available
- `complete` = video fully processed
- Videos not in this file = never started

### Rejected Chunks Tracking
Extend `rejected.json` entries to include chunk-level info:
```json
{
  "video1_chunk0042.mp4": {
    "source": "video1.mp4",
    "start_frame": 5040,
    "end_frame": 5160,
    "reason": "deleted"
  },
  "video1_chunk0015.mp4": {
    "source": "video1.mp4",
    "start_frame": 1800,
    "end_frame": 1920,
    "reason": "no_face"
  }
}
```
- `reason: "deleted"` = user removed the clip (blacklisted permanently)
- `reason: "no_face"` etc = existing rejection reasons (already tracked)
- When resuming a video, skip frame ranges that overlap any rejected chunk

### Video Selection Logic
When choosing which video to process next:
1. Filter to videos with `status != "complete"`
2. Exclude videos where all remaining chunks are in `rejected.json`
3. Pick randomly from the remaining pool
4. Start from `last_frame` position

### Edge Cases
- `max_samples` reduced between runs: don't delete clips, just don't add more
- `max_samples` increased: backfill as normal
- All videos complete and still under max: log a warning, nothing to do
- Video file removed from source folder: mark as complete in progress, skip
- Clip deleted AND its source video removed: add to rejected, can't backfill from that video

## Files Modified
- `nodes/ltxv_prepare_dataset.py`:
  - Add `max_samples` and `raw_folder` to `INPUT_TYPES`
  - Add `video_progress.json` read/write
  - Extend `rejected.json` format with chunk info
  - Modify clip extraction loop to check max_samples cap
  - Add random video selection logic
  - Add resume-from-position logic
  - Modify missing-clip detection to write chunk-level rejections

## Migration
- Existing `rejected.json` entries (string-only, no chunk info) remain valid
- Code checks if entry is a string (legacy) or dict (new format)
- `video_progress.json` is created on first run if it doesn't exist
- Existing datasets without `max_samples` work unchanged (0 = unlimited)

## Testing Plan
- Run with max_samples=50 on a folder with multiple long videos
- Verify it stops at 50 clips
- Delete 10 clips, re-run, verify it backfills to 50 from different video chunks
- Verify deleted clips appear in rejected.json with chunk info
- Re-run again, verify rejected chunks are never re-extracted
- Test with max_samples=0 to confirm unlimited still works
