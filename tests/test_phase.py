from ct_eus_vessel.phase import choose_primary_series, classify_phase


def test_classify_phase_from_series_description_and_protocol() -> None:
    assert classify_phase("1.0 x 1.0_A", "Arterial Phase") == "arterial"
    assert classify_phase("1.0 x 1.0_A", "Portal Phase") == "portal"
    assert classify_phase("Venous 1.0", "Abdomen Venous Phase") == "venous"
    assert classify_phase("5.0 x 5.0_Med", "ChestUnionAbd") == "other"


def test_choose_primary_series_prefers_more_slices_for_same_phase() -> None:
    series = [
        {"series_uid": "small", "phase": "portal", "num_instances": 50},
        {"series_uid": "large", "phase": "portal", "num_instances": 998},
        {"series_uid": "arterial", "phase": "arterial", "num_instances": 227},
    ]

    selected = choose_primary_series(series)

    assert selected["portal"]["series_uid"] == "large"
    assert selected["arterial"]["series_uid"] == "arterial"
