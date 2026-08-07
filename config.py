"""Global configuration for Adaptive Quantum Semantic Communication."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_DATA = PROJECT_ROOT / "data"
_SIBLING_DATA = PROJECT_ROOT.parent / "adaptive_qsc_main" / "data"
DATA_ROOT = (
    _LOCAL_DATA
    if (_LOCAL_DATA / "cifar-10-batches-py").exists()
    else _SIBLING_DATA
)
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
DOWNLOAD_CIFAR = not (DATA_ROOT / "cifar-10-batches-py").exists()

RANDOM_SEED = 42

# Semantic front-end
NUM_SEMANTIC_SAMPLES = 3000
NUM_SEMANTIC_CLUSTERS = 10

# MAB action space: fraction of concepts retained as |S|
COMPRESSION_LEVELS = [1.0, 0.8, 0.6, 0.4, 0.2]

# SeQUeNCe memory defaults
MEMORY_FREQUENCY = 2000.0
MEMORY_WAVELENGTH = 500.0
CLASSICAL_DISTANCE = 1000.0
CLASSICAL_DELAY = 1e8
TIME_GAP = 1e11

# Service window sized for multi-factor network conditions
# (fidelity + entanglement + decoherence + traffic):
#   good:     leftover slots >= K → full |S| works
#   moderate: leftover slots ~ mid |S| → compress
#   poor:     leftover slots ~ 2 → heavy compress
PROBE_TRIALS = 14
SERVICE_WINDOW = 12
MAX_DISTANCE_M = 2500
DELAY_REFERENCE_S = 1.0
TRANSMIT_REPEATS = 3

# LinUCB
LINUCB_ALPHA = 0.3
CONTEXT_DIM = 14
TRAIN_EPISODES = 900
