"""Shared episode loop: probe → compress → transmit → reward."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from config import (
    DELAY_REFERENCE_S,
    MAX_DISTANCE_M,
    MEMORY_FREQUENCY,
    MEMORY_WAVELENGTH,
    PROBE_TRIALS,
    SERVICE_WINDOW,
    TRANSMIT_REPEATS,
)
from learning.reward import (
    build_context,
    compute_reward,
    normalize_delay,
    observed_stress,
)
from qnetwork.entanglement_env import EntanglementEnvironment
from qnetwork.topology import TwoNodeTopology
from semantic.compressor import CompressionResult, SemanticCompressor
from semantic.concept_builder import ConceptSet


@dataclass
class EpisodeOutcome:
    episode: int
    compression_level: float
    num_states: int
    semantic_fidelity: float
    communication_fidelity: float
    reward: float
    success_rate: float
    pair_fidelity: float
    drop_ratio: float
    delay_seconds: float
    network_stress: float
    offered_load: float
    distance: float
    num_concepts: int
    service_window: int
    available_semantic_slots: int
    context: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_env(episode: dict[str, Any], *, seed_key: str, num_trials: int) -> EntanglementEnvironment:
    topology = TwoNodeTopology(
        distance=float(episode["distance"]),
        attenuation=float(episode["attenuation"]),
        num_trials=int(num_trials),
    )
    return EntanglementEnvironment(
        topology=topology,
        seed_offset=int(episode[seed_key]),
        memory_fidelity=float(episode["memory_fidelity"]),
        memory_frequency=float(MEMORY_FREQUENCY),
        memory_efficiency=float(episode["memory_efficiency"]),
        memory_coherence_time=float(episode["memory_coherence_time"]),
        memory_wavelength=float(MEMORY_WAVELENGTH),
        detector_efficiency=float(episode["detector_efficiency"]),
    )


def probe_episode(
    episode: dict[str, Any],
    *,
    probe_trials: int = PROBE_TRIALS,
    max_distance: int = MAX_DISTANCE_M,
    delay_reference: float = DELAY_REFERENCE_S,
    num_concepts: int | None = None,
    service_window: int | None = None,
) -> tuple[list[float], float, dict[str, float]]:
    """Return (context, stress, probe_stats)."""
    env = make_env(episode, seed_key="probe_seed_offset", num_trials=probe_trials)
    metrics = env.probe()

    success = float(metrics.success_rate)
    fidelity = float(metrics.average_fidelity)
    delay_n = normalize_delay(float(metrics.average_delay), delay_reference)
    dist_n = min(1.0, float(episode["distance"]) / float(max_distance))
    load = float(episode["offered_load"])
    k = int(
        num_concepts
        if num_concepts is not None
        else episode.get("num_concepts", 10)
    )
    window = int(
        service_window
        if service_window is not None
        else episode.get("service_window", SERVICE_WINDOW)
    )
    semantic_slots = max(0, window - int(round(load * window)))

    context = build_context(
        probe_success_rate=success,
        probe_average_fidelity=fidelity,
        normalized_distance=dist_n,
        normalized_probe_delay=delay_n,
        offered_load=load,
        num_concepts=k,
        service_window=window,
    )
    stress = observed_stress(
        probe_success_rate=success,
        probe_average_fidelity=fidelity,
        normalized_probe_delay=delay_n,
        offered_load=load,
    )
    stats = {
        "probe_success_rate": success,
        "probe_average_fidelity": fidelity,
        "probe_communication_fidelity": success * fidelity,
        "probe_delay": float(metrics.average_delay),
        "network_stress": stress,
        "num_concepts": float(k),
        "service_window": float(window),
        "available_semantic_slots": float(semantic_slots),
    }
    return context, stress, stats


def evaluate_action(
    episode: dict[str, Any],
    concepts: ConceptSet,
    compression_level: float,
    *,
    context: list[float] | None = None,
    network_stress: float | None = None,
    service_window: int | None = None,
    transmit_repeats: int = TRANSMIT_REPEATS,
) -> EpisodeOutcome:
    """
    Apply one compression action under the episode's physical network.

    Resource model: required_pairs = |S|.
    Communication metrics are averaged over ``transmit_repeats`` SeQUeNCe
    seeds to reduce EG noise.
    """
    window = int(
        service_window
        if service_window is not None
        else episode.get("service_window", SERVICE_WINDOW)
    )
    compressor = SemanticCompressor(concepts.num_concepts)
    mapping: CompressionResult = compressor.compress(
        concepts.centroids,
        concepts.probabilities,
        compression_level,
    )

    success_rates = []
    pair_fidelities = []
    delays = []
    dropped = []
    used = []

    base_seed = int(episode["transmit_seed_offset"])
    for rep in range(max(1, int(transmit_repeats))):
        ep_rep = dict(episode)
        ep_rep["transmit_seed_offset"] = base_seed + 17 * rep
        env = make_env(
            ep_rep,
            seed_key="transmit_seed_offset",
            num_trials=window,
        )
        tx = env.transmit(
            required_pairs=mapping.num_states,
            offered_load=float(episode["offered_load"]),
        )
        success_rates.append(float(tx.success_rate))
        pair_fidelities.append(float(tx.average_fidelity))
        delays.append(float(tx.average_delay))
        dropped.append(float(tx.dropped_pairs))
        used.append(float(tx.entangled_pairs_used))

    success_rate = sum(success_rates) / len(success_rates)
    pair_fidelity = sum(pair_fidelities) / len(pair_fidelities)
    delay_seconds = sum(delays) / len(delays)
    drop_ratio = (sum(dropped) / len(dropped)) / max(1, mapping.num_states)
    fc = success_rate * pair_fidelity

    fs = float(mapping.semantic_fidelity)
    reward = compute_reward(fs, fc)

    if context is None or network_stress is None:
        context, network_stress, _ = probe_episode(
            episode,
            num_concepts=concepts.num_concepts,
            service_window=window,
        )

    semantic_slots = max(
        0,
        window - int(round(float(episode["offered_load"]) * window)),
    )

    return EpisodeOutcome(
        episode=int(episode["episode"]),
        compression_level=float(compression_level),
        num_states=int(mapping.num_states),
        semantic_fidelity=fs,
        communication_fidelity=fc,
        reward=reward,
        success_rate=success_rate,
        pair_fidelity=pair_fidelity,
        drop_ratio=drop_ratio,
        delay_seconds=delay_seconds,
        network_stress=float(network_stress),
        offered_load=float(episode["offered_load"]),
        distance=float(episode["distance"]),
        num_concepts=int(concepts.num_concepts),
        service_window=window,
        available_semantic_slots=semantic_slots,
        context=list(context),
    )


def load_concepts(
    *,
    num_samples: int,
    num_clusters: int,
    seed: int,
    data_root: str,
) -> ConceptSet:
    from semantic.concept_builder import ConceptBuilder

    return ConceptBuilder(
        num_samples=num_samples,
        num_clusters=num_clusters,
        seed=seed,
        data_root=data_root,
    ).build()


def recluster_concepts(
    source: ConceptSet,
    *,
    num_clusters: int,
    seed: int,
) -> ConceptSet:
    """Recluster cached embeddings without rerunning the feature extractor."""
    import numpy as np
    from sklearn.cluster import KMeans

    if not 1 <= num_clusters <= len(source.embeddings):
        raise ValueError("num_clusters must be within embedding count")

    model = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    assignments = model.fit_predict(source.embeddings)
    counts = np.bincount(assignments, minlength=num_clusters).astype(float)
    probabilities = counts / counts.sum()
    return ConceptSet(
        embeddings=source.embeddings,
        labels=source.labels,
        assignments=assignments.astype(np.int64),
        centroids=model.cluster_centers_.astype(np.float64),
        probabilities=probabilities.astype(np.float64),
    )
