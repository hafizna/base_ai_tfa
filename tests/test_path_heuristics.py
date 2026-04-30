import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.path_heuristics import (
    infer_path_kind,
    infer_path_tag,
    infer_status_data,
    infer_suspected_label,
    is_transformer_path,
)


def test_transformer_folder_keywords_are_detected():
    path = (
        r"C:\data\raw_data\UPT PURWOKERTO\2024\09. SEPTEMBER"
        r"\28092024_06.24_TRIP TRF #1 GI COMAL\Trafo 1_Diff\ABC123.CFG"
    )

    assert is_transformer_path(path) is True
    assert infer_path_kind(path) == "TRANSFORMER"
    assert infer_path_tag(path) == "87T CONFIRMED"


def test_transformer_ocr_only_is_detected():
    path = (
        r"C:\data\raw_data\UPT BOGOR\2024\04. April\25 April 2024 - Sentul trafo 2"
        r"\drive-download-20240522T101156Z-001\OCR HV TRF 2\DR\04.25.2024 08.50.34.366 Disturbance.000.cfg"
    )

    assert is_transformer_path(path) is True
    assert infer_path_kind(path) == "TRANSFORMER"
    assert infer_path_tag(path) == "OCR ONLY"


def test_transient_labels_still_take_priority():
    path = (
        r"C:\data\raw_data\UPT BANDUNG\2024\05. MEI"
        r"\Case Petir\Trafo 1_Diff\ABC123.CFG"
    )

    assert infer_path_kind(path) == "TRANSIENT"
    assert infer_path_tag(path) == "PETIR"


def test_non_transformer_path_stays_unlabeled():
    path = (
        r"C:\data\raw_data\UPT PURWOKERTO\2024\07. JULI"
        r"\26072024_16.07_TRIP INC PHT BTG\DFR INC 1 BTG.CFG"
    )

    assert is_transformer_path(path) is False
    assert infer_path_kind(path) == ""
    assert infer_path_tag(path) == ""


def test_transformer_candidate_without_ocr_or_diff():
    path = (
        r"C:\data\raw_data\UPT CIREBON\2024\04. APRIL"
        r"\10042024_GI BANJAR_BAY TRAFO 4_DISTRIBUSI PENYULANG\SomeGeneric.CFG"
    )

    assert is_transformer_path(path) is True
    assert infer_path_kind(path) == "TRANSFORMER"
    assert infer_path_tag(path) == "TRAFO CANDIDATE"


def test_status_data_and_suspected_label_for_transformer_case():
    path = (
        r"C:\data\raw_data\UPT PURWOKERTO\2024\09. SEPTEMBER"
        r"\28092024_06.24_TRIP TRF #1 GI COMAL\Trafo 1_Diff\ABC123.CFG"
    )

    assert infer_status_data(path) == "TRANSFORMER"
    assert infer_suspected_label(path) == "DIDUGA 87T CONFIRMED"


def test_status_data_and_suspected_label_for_unknown_case():
    path = (
        r"C:\data\raw_data\UPT JATINANGOR\2024\07. JULI"
        r"\rekaman biasa tanpa petunjuk\FILE.CFG"
    )

    assert infer_status_data(path) == "UNKNOWN"
    assert infer_suspected_label(path) == "DIDUGA UNKNOWN"


def test_equipment_keywords_are_mapped_to_peralatan_label():
    path = (
        r"C:\data\raw_data\UPT SEMARANG\2025\08. AGUSTUS"
        r"\Gangguan Pilot Wire PLCC\teleprotection\FILE.CFG"
    )

    assert infer_path_kind(path) == "TRANSIENT"
    assert infer_path_tag(path) == "PERALATAN"
    assert infer_suspected_label(path) == "DIDUGA PERALATAN"


def test_isolator_keyword_is_mapped_to_peralatan_label():
    path = (
        r"C:\data\raw_data\UPT PURWOKERTO\2025\08. AGUSTUS"
        r"\TRIP SUTT KSGHN-LMNIS ALAT ISOLATOR\DISTANCE\FR000037.cfg"
    )

    assert infer_path_kind(path) == "TRANSIENT"
    assert infer_path_tag(path) == "PERALATAN"
    assert infer_suspected_label(path) == "DIDUGA PERALATAN"
