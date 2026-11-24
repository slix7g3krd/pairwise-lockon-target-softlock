import pytest
import gameplay.lockon_target_softlock as mod

CASES = [
    ("los_every_frame", (1,)),
    ("handle_generation", (1,)),
    ("stable_tiebreak", (1,)),
    ("reset_interp", (1,)),
    ("query_seq", (1,)),
    ("fallback_freecam", (1,)),
    ("acquire", (1,)),
    ("release", (1,)),
    ("update_lock", (1,)),
    ("camera_lock", (1,)),
    ("validate_target", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
