"""
Microscopy Database Fetcher

Provides unified access to public microscopy image databases:
- BBBC (Broad Bioimage Benchmark Collection)
- Allen Cell Structure Benchmark
- OpenCell
- IDR (Image Data Resource)
"""

import requests
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging
from io import BytesIO
from PIL import Image
import json
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# Cache directory for downloaded images
CACHE_DIR = Path(__file__).parent / '.cache' / 'images'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DatasetInfo:
    """Metadata for a dataset"""
    def __init__(self, db: str, dataset_id: str, name: str, description: str,
                 url: str, image_count: int, resolution: float, channels: List[str]):
        self.db = db
        self.dataset_id = dataset_id
        self.name = name
        self.description = description
        self.url = url
        self.image_count = image_count
        self.resolution = resolution  # micrometers/pixel
        self.channels = channels

    def to_dict(self) -> dict:
        return {
            'db': self.db,
            'dataset_id': self.dataset_id,
            'name': self.name,
            'description': self.description,
            'url': self.url,
            'image_count': self.image_count,
            'resolution': self.resolution,
            'channels': self.channels,
        }


class ImageRecord:
    """A single image record from a dataset"""
    def __init__(self, dataset: DatasetInfo, image_id: str, filename: str,
                 channels: Dict[str, str], field_size: Tuple[float, float]):
        self.dataset = dataset
        self.image_id = image_id
        self.filename = filename
        self.channels = channels  # {"DAPI": "url", "GFP": "url", ...}
        self.field_size = field_size  # (width_um, height_um)

    def to_dict(self) -> dict:
        return {
            'dataset_id': self.dataset.dataset_id,
            'image_id': self.image_id,
            'filename': self.filename,
            'channels': self.channels,
            'field_size': self.field_size,
        }


class BBBCDatabase:
    """
    Broad Bioimage Benchmark Collection (BBBC)

    Free, public benchmark collections:
    - BBBC039: HeLa cells (fluorescent nuclei + actin)
    - BBBC006: Chinese hamster ovary (CHO) cells
    - BBBC008: Drosophila (fruit fly) cells
    - BBBC041: Whole slide images
    """

    BASE_URL = "https://data.broadinstitute.org/bbbc/image_sets"

    DATASETS = {
        'BBBC039': DatasetInfo(
            db='BBBC',
            dataset_id='BBBC039',
            name='HeLa Cells (Fluorescence)',
            description='Human cervical cancer (HeLa) cells labeled with Hoechst (nuclei, blue) and phalloidin (actin, red). Good for nuclear dynamics, cell cycle tracking.',
            url='https://data.broadinstitute.org/bbbc/image_sets/BBBC039/',
            image_count=10,
            resolution=0.1,  # micrometers/pixel
            channels=['DAPI', 'Actin'],
        ),
        'BBBC006': DatasetInfo(
            db='BBBC',
            dataset_id='BBBC006',
            name='CHO Cells (Tubulin)',
            description='Chinese hamster ovary (CHO) cells with microtubule (tubulin) staining. Good for cytoskeleton analysis.',
            url='https://data.broadinstitute.org/bbbc/image_sets/BBBC006/',
            image_count=5,
            resolution=0.063,
            channels=['Tubulin'],
        ),
        'BBBC008': DatasetInfo(
            db='BBBC',
            dataset_id='BBBC008',
            name='Drosophila (C57) Cells',
            description='Drosophila (fruit fly) embryonic cells with different staining patterns.',
            url='https://data.broadinstitute.org/bbbc/image_sets/BBBC008/',
            image_count=8,
            resolution=0.08,
            channels=['GFP', 'DAPI'],
        ),
    }

    @classmethod
    def list_datasets(cls) -> List[Dict]:
        """List all available BBBC datasets"""
        return [ds.to_dict() for ds in cls.DATASETS.values()]

    @classmethod
    def get_dataset(cls, dataset_id: str) -> Optional[DatasetInfo]:
        """Get metadata for a specific dataset"""
        return cls.DATASETS.get(dataset_id)

    @classmethod
    async def fetch_image(cls, dataset_id: str, image_filename: str,
                         channel: str = 'DAPI') -> Optional[np.ndarray]:
        """
        Fetch an image from BBBC.

        Args:
            dataset_id: e.g., 'BBBC039'
            image_filename: e.g., 'SiR_Actin_001.tif'
            channel: Channel to extract (for multi-channel images)

        Returns:
            numpy array (2D for grayscale, 3D for multi-channel)
        """
        dataset = cls.get_dataset(dataset_id)
        if not dataset:
            return None

        # Try cache first
        cache_path = CACHE_DIR / dataset_id / image_filename
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            try:
                return np.load(str(cache_path.with_suffix('.npy')))
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")

        # Fetch from BBBC
        try:
            url = f"{dataset.url}{image_filename}"
            logger.info(f"Fetching from BBBC: {url}")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Load image
            img = Image.open(BytesIO(response.content))
            arr = np.array(img, dtype=np.float32)

            # Cache it
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path.with_suffix('.npy')), arr)
            logger.info(f"Cached to: {cache_path}")

            return arr

        except Exception as e:
            logger.error(f"Failed to fetch {image_filename}: {e}")
            return None

    @classmethod
    def list_images(cls, dataset_id: str) -> List[str]:
        """List image filenames in a dataset"""
        # For now, return hardcoded list of known images
        # In production, would parse from BBBC API
        image_lists = {
            'BBBC039': [
                'SiR_Actin_001.tif',
                'SiR_Actin_002.tif',
                'SiR_Actin_003.tif',
                'SiR_Actin_004.tif',
                'SiR_Actin_005.tif',
            ],
            'BBBC006': [
                'v1_001.tif',
                'v1_002.tif',
                'v1_003.tif',
            ],
            'BBBC008': [
                'C57_01.tif',
                'C57_02.tif',
                'C57_03.tif',
            ],
        }
        return image_lists.get(dataset_id, [])


class AllenCellDatabase:
    """
    Allen Cell Structure Benchmark

    High-resolution 3D + time-lapse cell structure data
    Covers endogenously tagged human proteins
    """

    BASE_URL = "https://www.allencell.org/api"

    @staticmethod
    async def list_datasets() -> List[Dict]:
        """List available Allen Cell datasets"""
        return [
            {
                'db': 'AllenCell',
                'dataset_id': 'allencell_3d',
                'name': '3D Cell Structures',
                'description': '3D confocal microscopy of endogenously tagged human proteins',
                'resolution': 0.065,
                'channels': ['GFP', 'DAPI'],
            }
        ]

    @staticmethod
    async def fetch_image(dataset_id: str, cell_id: str) -> Optional[np.ndarray]:
        """Fetch a 3D image from Allen Cell"""
        try:
            # Would implement actual API call to Allen Cell
            logger.info(f"Allen Cell API not yet implemented for {cell_id}")
            return None
        except Exception as e:
            logger.error(f"Allen Cell fetch failed: {e}")
            return None


class OpenCellDatabase:
    """
    OpenCell Database

    Open-source endogenously tagged human proteins
    Provides 2D confocal microscopy images
    """

    BASE_URL = "https://opencell.czbiohub.org"

    @staticmethod
    async def list_datasets() -> List[Dict]:
        """List available OpenCell datasets"""
        return [
            {
                'db': 'OpenCell',
                'dataset_id': 'opencell_proteins',
                'name': 'Tagged Human Proteins',
                'description': 'Endogenously tagged human proteins in HeLa cells',
                'resolution': 0.08,
                'channels': ['GFP', 'DAPI'],
            }
        ]

    @staticmethod
    async def fetch_image(protein_name: str) -> Optional[np.ndarray]:
        """Fetch an image of a specific protein"""
        try:
            # Would implement actual API call to OpenCell
            logger.info(f"OpenCell API not yet implemented for {protein_name}")
            return None
        except Exception as e:
            logger.error(f"OpenCell fetch failed: {e}")
            return None


class IDRDatabase:
    """
    Image Data Resource (IDR)

    Open-source repository of reference image datasets
    from published scientific studies
    """

    BASE_URL = "https://idr.openmicroscopy.org"

    @staticmethod
    async def list_datasets() -> List[Dict]:
        """List available IDR datasets"""
        return [
            {
                'db': 'IDR',
                'dataset_id': 'idr_timelapse',
                'name': 'Time-lapse Microscopy',
                'description': 'Published time-lapse microscopy studies',
                'resolution': 0.1,
                'channels': ['GFP', 'DAPI', 'RFP'],
            }
        ]


class DatabaseBrowser:
    """
    Unified interface to all microscopy databases
    """

    DATABASES = {
        'BBBC': BBBCDatabase,
        'AllenCell': AllenCellDatabase,
        'OpenCell': OpenCellDatabase,
        'IDR': IDRDatabase,
    }

    @classmethod
    def list_all_datasets(cls) -> Dict[str, List[Dict]]:
        """List all datasets from all databases"""
        result = {}

        # BBBC (synchronous)
        try:
            result['BBBC'] = BBBCDatabase.list_datasets()
        except Exception as e:
            logger.error(f"Failed to list BBBC datasets: {e}")
            result['BBBC'] = []

        return result

    @classmethod
    async def fetch_image(cls, db: str, dataset_id: str,
                         image_id: str, **kwargs) -> Optional[np.ndarray]:
        """Fetch an image from the specified database"""

        if db == 'BBBC':
            return await BBBCDatabase.fetch_image(dataset_id, image_id, **kwargs)
        elif db == 'AllenCell':
            return await AllenCellDatabase.fetch_image(dataset_id, image_id)
        elif db == 'OpenCell':
            return await OpenCellDatabase.fetch_image(image_id)
        elif db == 'IDR':
            return await IDRDatabase.fetch_image(image_id)
        else:
            logger.error(f"Unknown database: {db}")
            return None

    @classmethod
    def list_images(cls, db: str, dataset_id: str) -> List[str]:
        """List images in a dataset"""
        if db == 'BBBC':
            return BBBCDatabase.list_images(dataset_id)
        return []

    @classmethod
    def get_dataset_info(cls, db: str, dataset_id: str) -> Optional[Dict]:
        """Get metadata for a specific dataset"""
        if db == 'BBBC':
            ds = BBBCDatabase.get_dataset(dataset_id)
            return ds.to_dict() if ds else None
        return None
