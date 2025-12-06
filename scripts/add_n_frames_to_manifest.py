#!/usr/bin/env python3
"""
Add src_n_frames and tgt_n_frames columns to TSV manifest files.

StreamSpeech 표준 형식에 맞추기 위해 n_frames 컬럼을 추가합니다.
"""

import argparse
import csv
import soundfile as sf
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np


def get_audio_frames(audio_path: Path, sample_rate: int = 16000) -> int:
    """
    Get number of frames from audio file.
    
    For 16kHz audio, frames = samples (1 frame = 1 sample).
    For mel-spectrogram features, frames = time steps.
    """
    try:
        info = sf.info(str(audio_path))
        # Return number of samples (frames for raw audio)
        # For 10ms frame shift at 16kHz: frames = samples / 160
        # But StreamSpeech uses raw sample count divided by 160
        frames = info.frames // 160  # 10ms frame shift estimation
        return int(frames)
    except Exception as e:
        print(f"Warning: Could not get frames for {audio_path}: {e}")
        return 0


def get_units_frames(units_path: Path) -> int:
    """
    Get number of frames from units file (.npy).
    """
    try:
        if units_path.suffix == ".npy":
            units = np.load(units_path)
            return len(units)
        elif units_path.suffix == ".npz":
            units_dict = np.load(units_path)
            # Take first array
            units = units_dict[list(units_dict.keys())[0]]
            return len(units)
        else:
            # Text file: count lines or tokens
            with units_path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return 0
                # Count tokens (space-separated)
                return len(content.split())
    except Exception as e:
        print(f"Warning: Could not get units frames for {units_path}: {e}")
        return 0


def process_manifest(
    in_path: Path,
    out_path: Path,
    data_root: Path,
    units_root: Optional[Path] = None,
):
    """
    Add src_n_frames and tgt_n_frames to manifest.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    data_root = Path(data_root)
    units_root = Path(units_root) if units_root else data_root
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    entries = []
    original_headers = None
    
    # Read existing manifest with original headers first
    with in_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"Manifest is empty: {in_path}")
        
        # Handle 2-line header (EchoStream format)
        if "\t" in first_line and first_line.split("\t")[0] == "id":
            original_headers = first_line.split("\t")
            pos = f.tell()
            next_line = f.readline()
            if next_line and next_line.strip() and not next_line.strip().split("\t")[0]:
                # Merge headers
                next_headers = next_line.strip().split("\t")
                for i, val in enumerate(next_headers):
                    if val and i < len(original_headers):
                        if not original_headers[i] or original_headers[i].strip() == "":
                            original_headers[i] = val
                    elif val and i >= len(original_headers):
                        original_headers.append(val)
            else:
                f.seek(pos)
        else:
            # First line is data root
            headers_line = f.readline().strip()
            original_headers = headers_line.split("\t")
        
        # Clean headers
        original_headers = [h.strip() if h else "" for h in original_headers]
        
        # Read data with original headers
        reader = csv.DictReader(f, fieldnames=original_headers, delimiter="\t")
        
        for row in reader:
            if not row or row.get("id") == "id" or not row.get("id"):
                continue
            
            # Skip empty rows
            if not row.get("src_audio") or not row.get("src_text"):
                continue
            
            entries.append(row)
    
    # Reorder headers to match StreamSpeech 표준 순서
    # StreamSpeech 순서: id, src_audio, src_n_frames, src_text, tgt_text, tgt_audio, tgt_n_frames
    standard_order = ["id", "src_audio", "src_n_frames", "src_text", "tgt_text", "tgt_audio", "tgt_n_frames"]
    new_headers = []
    
    # Add standard columns in order
    for col in standard_order:
        if col in original_headers:
            new_headers.append(col)
    
    # Add any remaining columns (like tgt_units) at the end
    for col in original_headers:
        if col and col.strip() and col not in new_headers:
            new_headers.append(col)
    
    # Ensure n_frames columns exist
    if "src_n_frames" not in new_headers:
        if "src_audio" in new_headers:
            idx = new_headers.index("src_audio") + 1
            new_headers.insert(idx, "src_n_frames")
        else:
            new_headers.append("src_n_frames")
    if "tgt_n_frames" not in new_headers:
        if "tgt_audio" in new_headers:
            idx = new_headers.index("tgt_audio") + 1
            new_headers.insert(idx, "tgt_n_frames")
        else:
            new_headers.append("tgt_n_frames")
    
    # Reorder entry data to match new headers
    for entry in entries:
        # Create new row with reordered columns
        new_row = {}
        for col in new_headers:
            new_row[col] = entry.get(col, "")
        # Update entry in place
        entry.clear()
        entry.update(new_row)
    
    # Process entries and add n_frames
    print(f"Processing {len(entries)} entries...")
    for entry in tqdm(entries, desc="Adding n_frames"):
        # Resolve paths
        src_audio_path = Path(entry["src_audio"])
        if not src_audio_path.is_absolute():
            src_audio_path = data_root / src_audio_path
        
        # Get src_n_frames
        if not entry.get("src_n_frames") or not entry["src_n_frames"].strip():
            if src_audio_path.exists():
                entry["src_n_frames"] = str(get_audio_frames(src_audio_path))
            else:
                entry["src_n_frames"] = "0"
        
        # Get tgt_n_frames
        if not entry.get("tgt_n_frames") or not entry["tgt_n_frames"].strip():
            tgt_n_frames_set = False
            # Try from tgt_audio (only if it looks like a file path)
            tgt_audio_val = entry.get("tgt_audio")
            if tgt_audio_val and tgt_audio_val.strip():
                # Check if it looks like a file path (ends with .wav, .flac, etc.)
                if any(tgt_audio_val.strip().endswith(ext) for ext in ['.wav', '.flac', '.mp3', '.ogg']):
                    tgt_audio_path = Path(tgt_audio_val)
                    if not tgt_audio_path.is_absolute():
                        tgt_audio_path = data_root / tgt_audio_path
                    try:
                        if tgt_audio_path.exists() and tgt_audio_path.is_file():
                            entry["tgt_n_frames"] = str(get_audio_frames(tgt_audio_path))
                            tgt_n_frames_set = True
                    except (OSError, ValueError) as e:
                        # File path too long or invalid, skip
                        pass
            
            # Try from tgt_units if tgt_audio didn't work
            if not tgt_n_frames_set and entry.get("tgt_units"):
                tgt_units_val = entry.get("tgt_units")
                if tgt_units_val and tgt_units_val.strip():
                    tgt_units_path = Path(tgt_units_val)
                    if not tgt_units_path.is_absolute():
                        # Try to extract filename from path
                        if "/" in tgt_units_val:
                            tgt_units_path = units_root / Path(tgt_units_val).name
                        else:
                            tgt_units_path = units_root / tgt_units_val
                    try:
                        if tgt_units_path.exists() and tgt_units_path.is_file():
                            entry["tgt_n_frames"] = str(get_units_frames(tgt_units_path))
                            tgt_n_frames_set = True
                    except (OSError, ValueError) as e:
                        pass
            
            if not tgt_n_frames_set:
                entry["tgt_n_frames"] = "0"
    
    # Write updated manifest
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(entries)
    
    print(f"✅ Updated manifest written to: {out_path}")
    print(f"   Added src_n_frames and tgt_n_frames for {len(entries)} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Add src_n_frames and tgt_n_frames columns to TSV manifest (StreamSpeech format)"
    )
    parser.add_argument(
        "--in", dest="in_path", required=True,
        help="Input TSV manifest file"
    )
    parser.add_argument(
        "--out", dest="out_path", required=True,
        help="Output TSV manifest file"
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Base directory for audio files (for resolving relative paths)"
    )
    parser.add_argument(
        "--units-root",
        help="Base directory for units files (default: same as data-root)"
    )
    
    args = parser.parse_args()
    
    process_manifest(
        in_path=args.in_path,
        out_path=args.out_path,
        data_root=args.data_root,
        units_root=args.units_root,
    )


if __name__ == "__main__":
    main()

