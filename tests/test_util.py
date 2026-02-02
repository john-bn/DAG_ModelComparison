from DAG_ModelComparison.comparator.util import major_airports_df


def test_major_airports_df_shape_and_columns():
    df = major_airports_df()
    assert list(df.columns) == ["icao", "city", "lat", "lon"]
    assert len(df) >= 10  # sanity check


def test_major_airports_df_unique_icao():
    df = major_airports_df()
    assert df["icao"].is_unique


def test_major_airports_df_types_and_ranges():
    df = major_airports_df()

    # lat/lon should be numeric
    assert df["lat"].dtype.kind in "fi"
    assert df["lon"].dtype.kind in "fi"

    # rough bounds sanity for CONUS airports
    assert df["lat"].between(10, 60).all()
    assert df["lon"].between(-140, -60).all()
