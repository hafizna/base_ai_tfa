import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.api.routers.event_simulator import SCENARIOS, process_scenario


def _scenario(scenario_id: str):
    return next(item for item in SCENARIOS if item["id"] == scenario_id)


def test_nr_mapping_uses_doc_rows_and_groups_tier1_trip_events():
    result = process_scenario(_scenario("nr-ponorogo-87l-distance"))

    incident = result["incidents"][0]
    tier1_notifications = [item for item in result["notifications"] if item["tier"] == 1]
    ar_block_step = next(step for step in result["trace"] if step["raw_event"] and step["raw_event"]["signal_ref"] == "NR_AR_BLOCK")

    assert incident["title"] == "Gangguan Line: Differential + Distance"
    assert incident["primary_tier"] == 1
    assert len(tier1_notifications) == 1
    assert tier1_notifications[0]["event_ids"] == ["e001", "e002", "e003", "e004"]
    assert ar_block_step["mapping"]["source_doc_section"] == "4.1 note"
    assert ar_block_step["classification"]["doc_expected_tier"] == 4
    assert ar_block_step["decision"] == "attach_alarm_to_existing_gangguan_group"


def test_abb_rrec_is_tier2_and_xcbr_position_stays_tier3():
    result = process_scenario(_scenario("abb-red670-double-reclose"))

    notifications = result["notifications"]
    reclose_notifications = [item for item in notifications if item["tier"] == 2]
    status_notifications = [item for item in notifications if item["tier"] == 3]
    ar_success_step = next(step for step in result["trace"] if step["raw_event"] and step["raw_event"]["signal_ref"] == "ABB_AR_SUCCESS")

    assert len(reclose_notifications) == 1
    assert reclose_notifications[0]["event_ids"] == ["e105", "e107", "e111"]
    assert reclose_notifications[0]["emit_ms"] == 11200.0
    assert status_notifications
    assert ar_success_step["classification"]["condition_match"] is True
    assert ar_success_step["classification"]["doc_expected_tier"] == 2


def test_measurement_rows_attach_as_context_not_notifications():
    result = process_scenario(_scenario("abb-red670-double-reclose"))

    measurement_ids = {item["event_id"] for item in result["artifacts"]["ignored_measurements"]}
    notification_event_ids = {
        event_id
        for notification in result["notifications"]
        for event_id in notification["event_ids"]
    }

    assert "e104" in measurement_ids
    assert not measurement_ids.intersection(notification_event_ids)


def test_micom_vendor_ggio_tor_is_tier1_but_comm_failure_is_tier4():
    result = process_scenario(_scenario("micom-p545-vendor-ddb"))

    tor_step = next(step for step in result["trace"] if step["raw_event"] and step["raw_event"]["signal_ref"] == "MICOM_TOR_Z1")
    comm_step = next(step for step in result["trace"] if step["raw_event"] and step["raw_event"]["signal_ref"] == "MICOM_COM_FAILURE")
    alarm_notifications = [item for item in result["notifications"] if item["tier"] == 4]

    assert tor_step["mapping"]["ln"] == "GGIO2"
    assert tor_step["classification"]["tier"] == 1
    assert tor_step["classification"]["cluster"] == "GANGGUAN"
    assert comm_step["mapping"]["ln"] == "GGIO_CF"
    assert comm_step["classification"]["tier"] == 4
    assert alarm_notifications
