"""Data versioning and integrity tracking utilities.

This module provides utilities for tracking data versions, computing file hashes,
and maintaining a manifest of all data assets used in experiments for reproducibility.

Usage:
    from data_versioning import compute_file_hash, DataManifest

    # Track data files
    manifest = DataManifest()
    manifest.add_file("data/raw/environment_tun.csv")
    manifest.add_file("data/processed/processed_tunisia.csv")

    # Save manifest
    manifest.save("data/DATA_MANIFEST.json")

    # Verify data hasn't changed
    manifest_old = DataManifest.load("data/DATA_MANIFEST.json")
    manifest_new = DataManifest()
    manifest_new.add_file("data/raw/environment_tun.csv")
    old_hash = manifest_old.get_hash("data/raw/environment_tun.csv")
    new_hash = manifest_new.get_hash("data/raw/environment_tun.csv")
    if new_hash != old_hash:
        print("Data has changed!")
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Compute hash of a file for integrity verification.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use (sha256, md5, sha1, etc.)

    Returns:
        Hexadecimal hash string

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


class DataManifest:
    """Track data files, versions, and integrity hashes.

    This maintains a manifest of all data files used in experiments, including:
    - File paths and sizes
    - Compute hashes (SHA256) for integrity verification
    - Timestamps of when data was captured
    - Optional metadata (source, description, etc.)

    Example:
        manifest = DataManifest()
        manifest.add_file("data/raw/environment_tun.csv", description="World Bank data")
        manifest.add_file("data/processed/processed_tunisia.csv", source="data loader pipeline")
        manifest.save("data/DATA_MANIFEST.json")
    """

    def __init__(self):
        """Initialize empty manifest."""
        self.files: Dict[str, Dict] = {}
        self.created_at = datetime.now().isoformat()

    def add_file(
        self, file_path: str, source: Optional[str] = None, description: Optional[str] = None
    ) -> None:
        """Add file to manifest, computing its hash and metadata.

        Args:
            file_path: Path to file
            source: Optional description of data source
            description: Optional description of the file

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = str(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Compute file statistics
        stat = os.stat(file_path)
        file_hash = compute_file_hash(file_path)

        # Store metadata
        self.files[file_path] = {
            "hash": file_hash,
            "size_bytes": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "source": source,
            "description": description,
        }

    def get_hash(self, file_path: str) -> Optional[str]:
        """Get stored hash for a file.

        Args:
            file_path: Path to file

        Returns:
            Hash string or None if file not in manifest
        """
        file_path = str(file_path)
        if file_path in self.files:
            return self.files[file_path]["hash"]
        return None

    def verify_file(self, file_path: str) -> bool:
        """Verify that a file's content hasn't changed since manifest was created.

        Args:
            file_path: Path to file

        Returns:
            True if file hash matches, False otherwise

        Raises:
            FileNotFoundError: If file doesn't exist or isn't in manifest
        """
        file_path = str(file_path)
        if file_path not in self.files:
            raise FileNotFoundError(f"File not in manifest: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        current_hash = compute_file_hash(file_path)
        stored_hash = self.files[file_path]["hash"]

        return current_hash == stored_hash

    def verify_all(self) -> Dict[str, bool]:
        """Verify all files in manifest.

        Returns:
            Dictionary of file_path -> is_valid
        """
        results = {}
        for file_path in self.files.keys():
            try:
                results[file_path] = self.verify_file(file_path)
            except FileNotFoundError:
                results[file_path] = False
        return results

    def save(self, output_path: str) -> None:
        """Save manifest to JSON file.

        Args:
            output_path: Path where to save manifest
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest_dict = {
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
            "files": self.files,
        }

        with open(path, "w") as f:
            json.dump(manifest_dict, f, indent=2)

    @classmethod
    def load(cls, manifest_path: str) -> "DataManifest":
        """Load manifest from JSON file.

        Args:
            manifest_path: Path to manifest file

        Returns:
            DataManifest instance

        Raises:
            FileNotFoundError: If manifest file doesn't exist
        """
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        with open(path, "r") as f:
            manifest_dict = json.load(f)

        manifest = cls()
        manifest.files = manifest_dict.get("files", {})
        manifest.created_at = manifest_dict.get("created_at", datetime.now().isoformat())
        return manifest

    def __repr__(self) -> str:
        """String representation."""
        return f"DataManifest(files={len(self.files)}, created={self.created_at})"
