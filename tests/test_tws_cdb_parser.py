import io
import struct
import zipfile

from core.tws_cdb_parser import parse_tws_cdb_bytes


def _rows_xml(rows: list[tuple[str, dict[str, object]]]) -> str:
    parts = ['<?xml version="1.0"?><NewDataSet>']
    for table, values in rows:
        parts.append(f"<{table}>")
        for key, value in values.items():
            parts.append(f"<{key}>{value}</{key}>")
        parts.append(f"</{table}>")
    parts.append("</NewDataSet>")
    return "".join(parts)


def _cdf_bytes(record_number: int, station: str, device_id: int, samples: list[float]) -> bytes:
    descriptor = f"""<?xml version="1.0" encoding="utf-8"?>
<FLXMLCDFEntity>
  <DeviceDescriptors>
    <DeviceDescriptor>
      <StationName>{station}</StationName>
      <DeviceName>{station.replace("GI ", "")}</DeviceName>
      <FeederName>LINE</FeederName>
      <DeviceID>{device_id}</DeviceID>
      <DeviceType>Cashel</DeviceType>
      <TimeLocked>GPS-LOCK</TimeLocked>
    </DeviceDescriptor>
  </DeviceDescriptors>
  <RecordType>FL</RecordType>
  <FLRecordDataHeader>
    <RecordNumber>{record_number}</RecordNumber>
    <LineModule>LINE</LineModule>
    <TriggerTime>2026-04-05T06:46:28</TriggerTime>
    <TriggerTimeUS>1775371588.1</TriggerTimeUS>
    <GPSTag>{record_number}999</GPSTag>
    <SampleRateInHz>1000</SampleRateInHz>
    <TotalNumberOfSamples>4</TotalNumberOfSamples>
    <TotalNumberOfFrames>4</TotalNumberOfFrames>
    <decimation>1</decimation>
    <PostPreTrgFactor>10</PostPreTrgFactor>
    <CorrectedGPS>{record_number}111</CorrectedGPS>
    <DataDescriptor>
      <TriggerPhase>A</TriggerPhase>
      <SignallingValue>1 2 3</SignallingValue>
      <SoftwareTriggerPhase>A</SoftwareTriggerPhase>
      <SoftwareTriggerPoint>1</SoftwareTriggerPoint>
      <TriggerDelay>0</TriggerDelay>
    </DataDescriptor>
    <FLChannelsInformation>
      <NoOfChannels>3</NoOfChannels>
    </FLChannelsInformation>
    <Gain>76</Gain>
  </FLRecordDataHeader>
  <BinaryData>
    <BinaryDataLink>record.dat</BinaryDataLink>
  </BinaryData>
</FLXMLCDFEntity>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("record.xml", descriptor)
        zf.writestr("record.dat", struct.pack("<" + "f" * len(samples), *samples))
    return buf.getvalue()


def test_parse_tws_cdb_zip_with_nested_cdf_float_blocks():
    x_samples = [1, 2, 3, 4, 10, 20, 30, 40, -1, -2, -3, -4]
    y_samples = [5, 6, 7, 8, 50, 60, 70, 80, -5, -6, -7, -8]

    cdb = io.BytesIO()
    with zipfile.ZipFile(cdb, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "DEVICE.XML",
            _rows_xml(
                [
                    ("device", {"DeviceId": 3, "DeviceName": "SUMADRA", "StationName": "GI SUMADRA"}),
                    ("device", {"DeviceId": 4, "DeviceName": "PAMENGPEUK", "StationName": "GI PAMENGPEUK"}),
                ]
            ),
        )
        zf.writestr(
            "FEEDER.XML",
            _rows_xml(
                [
                    ("Feeders", {"FeederId": 11, "DeviceId": 3, "FeederName": "PAMEUNGPEUK 1"}),
                    ("Feeders", {"FeederId": 19, "DeviceId": 4, "FeederName": "SUMADRA 1"}),
                ]
            ),
        )
        zf.writestr(
            "CIRCUIT2.XML",
            _rows_xml(
                [
                    ("FLCircuits", {"CircuitId": 1, "CircuitName": "PMPEK-SMDRA_1", "VelocityFactor": 99.5}),
                    ("PMPEK-SMDRA1", {"SegmentID": 1, "CircuitID": 1, "Name": "PMPEK-SMDRA1", "Length": 30.7}),
                ]
            ),
        )
        zf.writestr(
            "FLResults\\1\\0.XML",
            _rows_xml(
                [
                    (
                        "FLResults",
                        {
                            "ResultID": 9491,
                            "CircuitID": 1,
                            "DTFX": 14.36862,
                            "DTFY": 16.33138,
                            "IndexIdX": 111785,
                            "IndexIdY": 110910,
                            "FaultedSegment": 1,
                            "DistanceFromSegmentEndA": 16.33138,
                            "IsComponentFault": "true",
                        },
                    )
                ]
            ),
        )
        zf.writestr(
            "FL\\3\\0.XML",
            _rows_xml(
                [
                    (
                        "FLEvents",
                        {
                            "IndexId": 111785,
                            "DeviceId": 3,
                            "FeederId": 11,
                            "EventTimeUS": 1775371588.1,
                            "RecordNumber": 39362,
                            "GPSLocked": 1,
                            "RecordFileName": "x.cdf",
                        },
                    ),
                    (
                        "FLEvents",
                        {
                            "IndexId": 110910,
                            "DeviceId": 4,
                            "FeederId": 19,
                            "EventTimeUS": 1775371588.2,
                            "RecordNumber": 48734,
                            "GPSLocked": 1,
                            "RecordFileName": "y.cdf",
                        },
                    ),
                ]
            ),
        )
        zf.writestr("DATA/x.cdf", _cdf_bytes(39362, "GI SUMADRA", 3, x_samples))
        zf.writestr("DATA/y.cdf", _cdf_bytes(48734, "GI PAMENGPEUK", 4, y_samples))

    parsed = parse_tws_cdb_bytes(cdb.getvalue(), "sample.cdb")
    result = parsed["results"][0]

    assert parsed["source_type"] == "tws_cdb"
    assert result["circuit_name"] == "PMPEK-SMDRA_1"
    assert result["line_length_km"] == 30.7
    assert len(result["endpoints"]) == 2
    assert result["endpoints"][0]["role"] == "X"
    assert result["endpoints"][0]["station_display_name"] == "GI SUMADRA"
    assert result["endpoints"][0]["fault_distance_km"] == 14.36862
    assert result["endpoints"][0]["channels"][1]["samples"] == [10.0, 20.0, 30.0, 40.0]
    assert result["endpoints"][1]["role"] == "Y"
    assert result["endpoints"][1]["station_display_name"] == "GI PAMENGPEUK"
    assert result["endpoints"][1]["channels"][2]["samples"] == [-5.0, -6.0, -7.0, -8.0]
