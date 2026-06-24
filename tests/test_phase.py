from ct_eus_vessel.phase import choose_primary_series, classify_phase, infer_dynamic_phases


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


def test_infer_dynamic_phases_relabels_continuous_1mm_contrast_series() -> None:
    rows = [
        {
            "series_uid": "arterial",
            "phase": "arterial",
            "series_description": "1.0 x 1.0_A",
            "protocol_name": "Arterial Phase",
            "series_number": "501",
            "num_instances": 227,
        },
        {
            "series_uid": "early-portal",
            "phase": "portal",
            "series_description": "1.0 x 1.0_A",
            "protocol_name": "Portal Phase",
            "series_number": "601",
            "num_instances": 227,
        },
        {
            "series_uid": "portal",
            "phase": "portal",
            "series_description": "1.0 x 1.0_A",
            "protocol_name": "Portal Phase",
            "series_number": "701",
            "num_instances": 257,
        },
        {
            "series_uid": "late-portal",
            "phase": "portal",
            "series_description": "1.0 x 1.0_A",
            "protocol_name": "Portal Phase",
            "series_number": "801",
            "num_instances": 257,
        },
        {
            "series_uid": "venous",
            "phase": "portal",
            "series_description": "1.0 x 1.0_A",
            "protocol_name": "Portal Phase",
            "series_number": "901",
            "num_instances": 257,
        },
        {
            "series_uid": "mip",
            "phase": "arterial",
            "series_description": "3D_CT_Slice_MIP_Collection_1",
            "protocol_name": "Arterial Phase",
            "series_number": "10003",
            "num_instances": 13,
        },
    ]

    inferred = infer_dynamic_phases(rows)
    phases = {row["series_uid"]: row["phase"] for row in inferred}
    dynamic = {row["series_uid"]: row.get("dynamic_phase") for row in inferred}
    metadata = {row["series_uid"]: row.get("metadata_phase") for row in inferred}

    assert phases["arterial"] == "arterial"
    assert phases["portal"] == "portal"
    assert phases["venous"] == "venous"
    assert dynamic["venous"] == "venous"
    assert metadata["venous"] == "portal"
    assert phases["mip"] == "arterial"
