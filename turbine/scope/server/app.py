"""
SCOPE Backend Server

Exposes SCOPE runtime via HTTP API for Analysis Studio integration.
Supports execution of SCOPE programs with timing events and image frames.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import numpy as np
from typing import Dict, Any, List
import json
import base64
import asyncio

from ..runtime.scope_runtime import SCOPERuntime
from ..programs.nuclear_separation import create_nuclear_separation_program
from ..types.timing_cell import TimingDeviation
from .databases import DatabaseBrowser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Analysis Studio frontend

# Global program and runtime registry
PROGRAMS: Dict[str, Any] = {}
RUNTIMES: Dict[str, SCOPERuntime] = {}


def register_programs():
    """Register available SCOPE programs"""
    global PROGRAMS, RUNTIMES

    # Nuclear separation dynamics program
    nuclear_sep_prog = create_nuclear_separation_program()
    PROGRAMS['nuclear_separation_dynamics'] = nuclear_sep_prog
    RUNTIMES['nuclear_separation_dynamics'] = SCOPERuntime(nuclear_sep_prog)

    logger.info(f"Registered {len(PROGRAMS)} SCOPE programs")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'SCOPE Backend'})


@app.route('/programs', methods=['GET'])
def list_programs():
    """List available SCOPE programs"""
    programs_info = []
    for prog_id, prog in PROGRAMS.items():
        programs_info.append({
            'id': prog_id,
            'name': prog.name,
            'depth': prog.depth,
            'field_size': {
                'x': prog.field_size_x,
                'y': prog.field_size_y,
            },
            'resolution': prog.resolution,
            'morphisms': list(prog.morphisms.keys()),
            'timing_cells': [
                {
                    'cell_id': cell.cell_id,
                    'bounds_delta_p': list(cell.bounds_delta_p),
                }
                for cell in prog.dispatch_table.cell_partition.cells
            ]
        })

    return jsonify({
        'programs': programs_info,
        'count': len(programs_info)
    })


@app.route('/programs/<program_id>', methods=['GET'])
def get_program(program_id: str):
    """Get details of a specific program"""
    if program_id not in PROGRAMS:
        return jsonify({'error': f'Program not found: {program_id}'}), 404

    prog = PROGRAMS[program_id]
    return jsonify({
        'id': program_id,
        'name': prog.name,
        'depth': prog.depth,
        'field_size_x': prog.field_size_x,
        'field_size_y': prog.field_size_y,
        'resolution': prog.resolution,
        'lambda_s': prog.lambda_s,
        'lambda_t': prog.lambda_t,
    })


@app.route('/execute', methods=['POST'])
def execute():
    """
    Execute a SCOPE program.

    Request JSON:
    {
        "program_id": "nuclear_separation_dynamics",
        "timing_events": [
            {"delta_p": -1.2e-6, "channel_id": 0, "intensity": 100},
            ...
        ],
        "frame": {
            "data": [base64-encoded-array],
            "shape": [height, width],
            "dtype": "float32"
        }
    }

    Response JSON:
    {
        "success": true,
        "result": {
            "structure": "nucleus_a",
            "position": [x, y, z],
            "distance": 8.5e-6,
            "uncertainty": 1.4e-7,
            "s_entropy": {"S_k": 0.5, "S_t": 1e-6, "S_e": 0.5},
            "partition_state": {"n": 1000, "ℓ": 10, "m": 42, "s": 1}
        },
        "timing_ms": 123.45
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Extract program ID
        program_id = data.get('program_id')
        if not program_id or program_id not in RUNTIMES:
            return jsonify({'error': f'Invalid program_id: {program_id}'}), 400

        runtime = RUNTIMES[program_id]

        # Extract timing events
        timing_events_data = data.get('timing_events', [])
        timing_events = [
            TimingDeviation(
                delta_p=float(event['delta_p']),
                channel_id=int(event.get('channel_id', 0)),
                intensity=event.get('intensity')
            )
            for event in timing_events_data
        ]

        if not timing_events:
            return jsonify({'error': 'No timing events provided'}), 400

        # Extract frame
        frame_data = data.get('frame')
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400

        # Decode frame from base64
        frame_bytes = base64.b64decode(frame_data['data'])
        shape = tuple(frame_data['shape'])
        dtype = np.dtype(frame_data['dtype'])
        frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

        logger.info(
            f"Executing {program_id}: "
            f"{len(timing_events)} events, frame shape {frame.shape}"
        )

        # Execute SCOPE program
        import time
        start_time = time.time()
        result = runtime.run(timing_events, frame)
        elapsed_ms = (time.time() - start_time) * 1000

        # Serialize result
        result_dict = result.to_dict()

        return jsonify({
            'success': True,
            'result': result_dict,
            'timing_ms': elapsed_ms,
        })

    except Exception as e:
        logger.exception(f"Execution error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/execute-batch', methods=['POST'])
def execute_batch():
    """
    Execute a SCOPE program on multiple event streams.

    Request JSON:
    {
        "program_id": "nuclear_separation_dynamics",
        "timing_events_list": [
            [...event list 1...],
            [...event list 2...],
        ],
        "frame": {...same as /execute...}
    }

    Response: array of results
    """
    try:
        data = request.get_json()
        program_id = data.get('program_id')
        if not program_id or program_id not in RUNTIMES:
            return jsonify({'error': f'Invalid program_id: {program_id}'}), 400

        runtime = RUNTIMES[program_id]

        # Extract frame (shared for all)
        frame_data = data.get('frame')
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400

        frame_bytes = base64.b64decode(frame_data['data'])
        shape = tuple(frame_data['shape'])
        dtype = np.dtype(frame_data['dtype'])
        frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

        # Extract timing event lists
        timing_events_list_data = data.get('timing_events_list', [])
        timing_events_list = []
        for events_data in timing_events_list_data:
            events = [
                TimingDeviation(
                    delta_p=float(event['delta_p']),
                    channel_id=int(event.get('channel_id', 0)),
                    intensity=event.get('intensity')
                )
                for event in events_data
            ]
            timing_events_list.append(events)

        logger.info(
            f"Executing batch {program_id}: "
            f"{len(timing_events_list)} streams, frame shape {frame.shape}"
        )

        # Execute batch
        import time
        start_time = time.time()
        results = runtime.run_batch(timing_events_list, frame)
        elapsed_ms = (time.time() - start_time) * 1000

        # Serialize results
        results_dict = [r.to_dict() for r in results]

        return jsonify({
            'success': True,
            'results': results_dict,
            'count': len(results_dict),
            'timing_ms': elapsed_ms,
        })

    except Exception as e:
        logger.exception(f"Batch execution error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/encode-frame', methods=['POST'])
def encode_frame():
    """
    Helper endpoint to encode a frame for API calls.

    Request: FormData with 'frame' file (NumPy .npy)
    Response: Base64-encoded frame data for use in other endpoints
    """
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame file provided'}), 400

        file = request.files['frame']
        frame = np.load(file)

        # Encode to base64
        frame_bytes = frame.tobytes()
        frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')

        return jsonify({
            'data': frame_b64,
            'shape': list(frame.shape),
            'dtype': str(frame.dtype),
        })

    except Exception as e:
        logger.exception(f"Frame encoding error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/databases', methods=['GET'])
def list_databases():
    """
    List all available microscopy databases and their datasets.

    Response: {
        "BBBC": [
            {"dataset_id": "BBBC039", "name": "HeLa Cells", ...},
            ...
        ]
    }
    """
    try:
        all_datasets = DatabaseBrowser.list_all_datasets()
        return jsonify(all_datasets)
    except Exception as e:
        logger.exception(f"Database listing error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/databases/<db>/<dataset_id>/images', methods=['GET'])
def list_dataset_images(db: str, dataset_id: str):
    """
    List images in a specific dataset.

    Response: {
        "images": ["image1.tif", "image2.tif", ...],
        "count": 5
    }
    """
    try:
        images = DatabaseBrowser.list_images(db, dataset_id)
        return jsonify({
            'images': images,
            'count': len(images),
        })
    except Exception as e:
        logger.exception(f"Image listing error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/databases/<db>/<dataset_id>', methods=['GET'])
def get_dataset_info(db: str, dataset_id: str):
    """
    Get metadata for a specific dataset.

    Response: {
        "db": "BBBC",
        "dataset_id": "BBBC039",
        "name": "HeLa Cells",
        "resolution": 0.1,
        ...
    }
    """
    try:
        info = DatabaseBrowser.get_dataset_info(db, dataset_id)
        if not info:
            return jsonify({'error': f'Dataset not found: {db}/{dataset_id}'}), 404
        return jsonify(info)
    except Exception as e:
        logger.exception(f"Dataset info error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/databases/<db>/<dataset_id>/<image_id>', methods=['GET'])
def fetch_dataset_image(db: str, dataset_id: str, image_id: str):
    """
    Fetch an image from a database and return as base64.

    Query parameters:
    - channel: Specific channel to extract (e.g., "DAPI")

    Response: {
        "success": true,
        "data": "base64-encoded-array",
        "shape": [height, width],
        "dtype": "float32",
        "source": "BBBC/BBBC039",
        "filename": "image.tif"
    }
    """
    try:
        channel = request.args.get('channel', 'DAPI')

        logger.info(
            f"Fetching image: {db}/{dataset_id}/{image_id} "
            f"(channel={channel})"
        )

        # Run async fetch in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            arr = loop.run_until_complete(
                DatabaseBrowser.fetch_image(db, dataset_id, image_id, channel=channel)
            )
        finally:
            loop.close()

        if arr is None:
            return jsonify({
                'success': False,
                'error': f'Failed to fetch image: {image_id}'
            }), 404

        # Normalize to [0, 1] if needed
        if arr.max() > 1.0:
            arr = arr / arr.max()

        # Convert to base64
        arr_bytes = arr.astype(np.float32).tobytes()
        arr_b64 = base64.b64encode(arr_bytes).decode('utf-8')

        return jsonify({
            'success': True,
            'data': arr_b64,
            'shape': list(arr.shape),
            'dtype': 'float32',
            'source': f'{db}/{dataset_id}',
            'filename': image_id,
        })

    except Exception as e:
        logger.exception(f"Image fetch error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """404 handler"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """500 handler"""
    return jsonify({'error': 'Internal server error'}), 500


def create_app(debug=False):
    """Factory function to create app with optional debug mode"""
    app.debug = debug
    register_programs()
    return app


if __name__ == '__main__':
    app = create_app(debug=True)
    logger.info("Starting SCOPE Backend Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
