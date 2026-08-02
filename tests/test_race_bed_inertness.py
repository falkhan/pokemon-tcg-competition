"""M39 P2: which gate beds the race rules can and cannot touch — structurally.

The racemode family triggers on OPPONENT board card ids, so whether a rule can
possibly act against a given bed is a property of that bed's 60-card list, not
a sample. This test states the partition once, in code, so a gate cell can
never be read as evidence about a rule that provably cannot fire in it (G-8,
and the G-11 mechanism-proof requirement).

It also guards the P2a id additions in the direction that actually costs us:
a new trigger id that quietly lights up the mirror or rocket lists would turn
two inert cells into live ones and change what every previous gate meant.
`scripts/m39_race_probe.py` measures the same thing at runtime (race_engaged
prompts per bed); this is the version that fails in CI.
"""
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from rl.plan import _RACEMODE_PRESSURE_IDS, _RACEMODE_WALL_IDS

ROOT = Path(__file__).resolve().parent.parent

# bed deck -> the trigger half it is expected to light up ("wall" fires the
# blanket branch, "pressure" the margin-gated one, None = provably inert).
BED_DECKS = {
    "greattusk_wall": "wall",       # Great Tusk / Dwebble / Crustle / Terrakion
    "grim_live": "pressure",        # Marnie's Impidimp / Morgrem / Grimmsnarl ex
    "hops_stall": "pressure",       # Hop's Phantump / Trevenant / Snorlax
    "archaludon": None,
    "clone54618168": None,          # mirror, m28 and the 900+ instrument bed
    "alakazam_v2_h4": None,         # our own list, i.e. the mirror bed's deck
    "lucario": None,                # rule:tuned
    "dragapult": None,
    "iono": None,
}


def _deck(name: str) -> set:
    path = ROOT / "decks" / f"{name}.csv"
    return {int(x) for x in path.read_text().split() if x.strip()}


@pytest.mark.parametrize("name,expected", sorted(BED_DECKS.items()))
def test_bed_trigger_partition(name, expected):
    ids = _deck(name)
    wall = ids & _RACEMODE_WALL_IDS
    pressure = ids & _RACEMODE_PRESSURE_IDS
    if expected == "wall":
        assert wall, f"{name} should fire the blanket branch"
    elif expected == "pressure":
        assert pressure and not wall, f"{name} should fire margin-gated only"
    else:
        assert not wall and not pressure, (
            f"{name} was an INERT gate bed and now carries trigger ids "
            f"{sorted(wall | pressure)} — every earlier cell measured against "
            f"it meant something different from what a new one will mean")
