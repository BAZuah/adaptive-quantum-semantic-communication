"""Two-node SeQUeNCe entanglement generation environment."""

from __future__ import annotations

import random

from sequence.components.memory import Memory
from sequence.components.optical_channel import ClassicalChannel, QuantumChannel
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.kernel.timeline import Timeline
from sequence.message import Message
from sequence.topology.node import BSMNode, Node

from config import (
    CLASSICAL_DELAY,
    CLASSICAL_DISTANCE,
    MEMORY_FREQUENCY,
    MEMORY_WAVELENGTH,
    TIME_GAP,
)
from qnetwork.metrics import NetworkMetrics
from qnetwork.topology import TwoNodeTopology

PICOSECONDS_PER_SECOND = 1e12


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class SimpleManager:
    def __init__(self, owner, memo_name: str):
        self.owner = owner
        self.memo_name = memo_name
        self.raw_counter = 0
        self.ent_counter = 0
        self.last_entanglement_fidelity = 0.0

    def update(self, protocol, memory, state):
        if state == "RAW":
            self.raw_counter += 1
            memory.reset()
        else:
            self.ent_counter += 1
            self.last_entanglement_fidelity = float(memory.fidelity)

    def create_protocol(self, middle: str, other: str):
        self.owner.protocols = [
            EntanglementGenerationA.create(
                self.owner,
                f"{self.owner.name}.eg",
                middle,
                other,
                self.owner.components[self.memo_name],
            )
        ]


class EntangleGenNode(Node):
    def __init__(
        self,
        name: str,
        timeline: Timeline,
        *,
        memory_fidelity: float,
        memory_frequency: float,
        memory_efficiency: float,
        memory_coherence_time: float,
        memory_wavelength: float,
    ):
        super().__init__(name, timeline)
        memo_name = f"{name}.memo"
        memory = Memory(
            memo_name,
            timeline,
            memory_fidelity,
            memory_frequency,
            memory_efficiency,
            memory_coherence_time,
            memory_wavelength,
        )
        memory.add_receiver(self)
        self.add_component(memory)
        self.resource_manager = SimpleManager(self, memo_name)

    def init(self):
        self.get_components_by_type("Memory")[0].reset()

    def receive_message(self, src, msg: Message):
        self.protocols[0].received_message(src, msg)

    def get(self, photon, **kwargs):
        self.send_qubit(kwargs["dst"], photon)


def pair_protocol(node1, node2):
    p1, p2 = node1.protocols[0], node2.protocols[0]
    memo1 = node1.get_components_by_type("Memory")[0].name
    memo2 = node2.get_components_by_type("Memory")[0].name
    p1.set_others(p2.name, node2.name, [memo2])
    p2.set_others(p1.name, node1.name, [memo1])


class EntanglementEnvironment:
    """
    Physical SeQUeNCe EG substrate.

    Quality comes only from simulator knobs (distance, attenuation,
    memory, detectors). Offered load is background contention inside
    a finite service window. No post-hoc fidelity hacks.
    """

    def __init__(
        self,
        topology: TwoNodeTopology,
        seed_offset: int = 0,
        *,
        memory_fidelity: float = 0.9,
        memory_frequency: float = MEMORY_FREQUENCY,
        memory_efficiency: float = 1.0,
        memory_coherence_time: float = 1.0,
        memory_wavelength: float = MEMORY_WAVELENGTH,
        detector_efficiency: float = 1.0,
    ):
        self.topology = topology
        self.seed_offset = int(seed_offset)
        self.memory_fidelity = clamp(float(memory_fidelity))
        self.memory_frequency = float(memory_frequency)
        self.memory_efficiency = clamp(float(memory_efficiency))
        self.memory_coherence_time = float(memory_coherence_time)
        self.memory_wavelength = float(memory_wavelength)
        self.detector_efficiency = clamp(float(detector_efficiency))

        if self.memory_frequency <= 0:
            raise ValueError("memory_frequency must be positive")
        if self.memory_coherence_time == 0:
            raise ValueError("memory_coherence_time cannot be zero")

    def setup_network(self):
        timeline = Timeline()
        timeline.show_progress = False

        node1 = EntangleGenNode(
            "node1",
            timeline,
            memory_fidelity=self.memory_fidelity,
            memory_frequency=self.memory_frequency,
            memory_efficiency=self.memory_efficiency,
            memory_coherence_time=self.memory_coherence_time,
            memory_wavelength=self.memory_wavelength,
        )
        node2 = EntangleGenNode(
            "node2",
            timeline,
            memory_fidelity=self.memory_fidelity,
            memory_frequency=self.memory_frequency,
            memory_efficiency=self.memory_efficiency,
            memory_coherence_time=self.memory_coherence_time,
            memory_wavelength=self.memory_wavelength,
        )
        bsm_node = BSMNode("bsm_node", timeline, ["node1", "node2"])

        node1.set_seed(self.seed_offset)
        node2.set_seed(self.seed_offset + 1)
        bsm_node.set_seed(self.seed_offset + 2)

        bsm = bsm_node.get_components_by_type("SingleAtomBSM")[0]
        bsm.update_detectors_params("efficiency", self.detector_efficiency)

        qc1 = QuantumChannel(
            "qc1",
            timeline,
            attenuation=self.topology.attenuation,
            distance=self.topology.distance,
        )
        qc2 = QuantumChannel(
            "qc2",
            timeline,
            attenuation=self.topology.attenuation,
            distance=self.topology.distance,
        )
        qc1.set_ends(node1, bsm_node.name)
        qc2.set_ends(node2, bsm_node.name)

        nodes = [node1, node2, bsm_node]
        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                if i == j:
                    continue
                cc = ClassicalChannel(
                    f"cc_{src.name}_{dst.name}",
                    timeline,
                    CLASSICAL_DISTANCE,
                    CLASSICAL_DELAY,
                )
                cc.set_ends(src, dst.name)

        timeline.init()
        return timeline, node1, node2

    def generate_one_pair(self, timeline, node1, node2):
        start_time = timeline.now()
        timeline.time = timeline.now() + TIME_GAP

        node1.resource_manager.create_protocol("bsm_node", "node2")
        node2.resource_manager.create_protocol("bsm_node", "node1")
        pair_protocol(node1, node2)

        memory1 = node1.get_components_by_type("Memory")[0]
        memory2 = node2.get_components_by_type("Memory")[0]
        memory1.reset()
        memory2.reset()
        node1.resource_manager.last_entanglement_fidelity = 0.0
        node2.resource_manager.last_entanglement_fidelity = 0.0

        before = node1.resource_manager.ent_counter
        node1.protocols[0].start()
        node2.protocols[0].start()
        timeline.run()

        success = node1.resource_manager.ent_counter > before
        fidelity = (
            float(node1.resource_manager.last_entanglement_fidelity)
            if success
            else 0.0
        )
        elapsed = max(0.0, (timeline.now() - start_time) / PICOSECONDS_PER_SECOND)
        return success, fidelity, elapsed

    def probe(self) -> NetworkMetrics:
        """Measure raw EG quality without semantic demand or load."""
        successes = 0
        fidelities = []
        delays = []
        timeline, node1, node2 = self.setup_network()

        for _ in range(int(self.topology.num_trials)):
            ok, fid, dt = self.generate_one_pair(timeline, node1, node2)
            delays.append(dt)
            if ok:
                successes += 1
                fidelities.append(fid)

        n = int(self.topology.num_trials)
        return NetworkMetrics(
            success_rate=successes / n if n else 0.0,
            average_fidelity=(sum(fidelities) / len(fidelities)) if fidelities else 0.0,
            average_delay=(sum(delays) / len(delays)) if delays else 0.0,
            entangled_pairs_used=n,
            available_pairs=successes,
            dropped_pairs=0,
        )

    def transmit(
        self,
        required_pairs: int,
        offered_load: float = 0.0,
    ) -> NetworkMetrics:
        """
        Serve |S| semantic pair requests under background traffic.

        Finite EG window = topology.num_trials attempts.
        Background traffic is served first (higher-priority / offered load),
        then remaining slots are available for semantic requests.

        This models scarce entanglement under dynamic traffic: a static
        large |S| is starved when the network is busy, while adaptive
        compression can still fit in the leftover window.
        """
        if required_pairs < 0:
            raise ValueError("required_pairs cannot be negative")
        offered_load = clamp(float(offered_load))

        if required_pairs == 0:
            return NetworkMetrics(0.0, 0.0, 0.0, 0, 0, 0)

        window = max(1, int(self.topology.num_trials))
        background = int(round(offered_load * window))
        available_for_semantic = max(0, window - background)
        scheduled = min(required_pairs, available_for_semantic)
        unscheduled = required_pairs - scheduled

        semantic_attempts = 0
        semantic_successes = 0
        fidelities = []
        completion_delays = []
        timeline, node1, node2 = self.setup_network()
        elapsed_total = 0.0

        # Consume background slots first (no semantic credit).
        for _ in range(background):
            ok, fid, dt = self.generate_one_pair(timeline, node1, node2)
            elapsed_total += dt

        for _ in range(scheduled):
            ok, fid, dt = self.generate_one_pair(timeline, node1, node2)
            elapsed_total += dt
            semantic_attempts += 1
            if ok:
                semantic_successes += 1
                fidelities.append(fid)
                completion_delays.append(elapsed_total)

        return NetworkMetrics(
            success_rate=semantic_successes / required_pairs,
            average_fidelity=(sum(fidelities) / len(fidelities)) if fidelities else 0.0,
            average_delay=(
                (sum(completion_delays) / len(completion_delays))
                if completion_delays
                else elapsed_total
            ),
            entangled_pairs_used=semantic_attempts,
            available_pairs=semantic_successes,
            dropped_pairs=unscheduled + (semantic_attempts - semantic_successes),
        )
