"""Station and regional-source-node registry.

Local nodes: a representative set of Delhi-NCR CAAQMS monitoring stations
(receptors). Regional nodes: external source regions whose emissions/fires
transport into Delhi under favorable wind + stability conditions (per Dr.
Priya's suggestion to treat Punjab/Haryana/Rajasthan as explicit external
source nodes, extended with western UP per the literature review's G_R
definition).

Replace `LOCAL_STATIONS` with the real CPCB station list + lat/lon once
`src/aqf/data/cpcb.py` is wired to a live API key — nothing downstream depends
on these being fake.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    lat: float
    lon: float


# A representative spread of real Delhi CAAQMS station locations (lat/lon are
# the actual public station coordinates; swap for the live CPCB station list
# via src/aqf/data/cpcb.py for production use).
LOCAL_STATIONS: list[Node] = [
    Node("DL001", "Anand Vihar", 28.6469, 77.3157),
    Node("DL002", "R K Puram", 28.5629, 77.1855),
    Node("DL003", "Punjabi Bagh", 28.6740, 77.1310),
    Node("DL004", "Dwarka Sector 8", 28.5709, 77.0720),
    Node("DL005", "ITO", 28.6304, 77.2495),
    Node("DL006", "Mandir Marg", 28.6362, 77.2010),
    Node("DL007", "Okhla Phase 2", 28.5310, 77.2710),
    Node("DL008", "Rohini", 28.7330, 77.1200),
    Node("DL009", "Shadipur", 28.6514, 77.1580),
    Node("DL010", "Wazirpur", 28.6994, 77.1650),
    Node("DL011", "Najafgarh", 28.6090, 76.9790),
    Node("DL012", "Narela", 28.8480, 77.0910),
    Node("DL013", "Sonia Vihar", 28.7150, 77.2500),
    Node("DL014", "Vivek Vihar", 28.6720, 77.3150),
    Node("DL015", "Ashok Vihar", 28.6950, 77.1820),
]

# External regional source regions (centroid coordinates). These are the
# nodes that push influence into LOCAL_STATIONS via the dynamic transport
# graph G_R(t), gated by wind alignment, distance, and atmospheric stability.
REGIONAL_SOURCES: list[Node] = [
    Node("SRC_PB", "Punjab (stubble burning belt)", 30.9010, 75.8573),
    Node("SRC_HR", "Haryana (stubble burning belt)", 29.0588, 76.0856),
    Node("SRC_RJ", "Rajasthan (dust / NW source)", 27.0238, 74.2179),
    Node("SRC_UP", "Western UP (industrial corridor)", 28.4595, 77.5250),
]


def local_ids() -> list[str]:
    return [n.id for n in LOCAL_STATIONS]


def regional_ids() -> list[str]:
    return [n.id for n in REGIONAL_SOURCES]
