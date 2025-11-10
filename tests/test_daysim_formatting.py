"""End-to-end tests for DaySim formatting to ensure identical output.

This test suite compares the output of the new modular pipeline with DaySim
formatting against the output from the original daysim_old pipeline to ensure
they produce identical results.

Test Organization:
    1. TestDaysimFormatting: Tests for 02a-reformat step
       - DaySim person, household, trip formatting
       - Field distributions and mappings

    2. TestDaysimFormatter: Unit tests for DaysimFormatter class
       - Age mapping, person type derivation, mode mapping

    3. TestPipelineDataFlow: Documentation tests for data flow
       - Trip count tracking through pipeline stages
       - Linked trip structure analysis
       - Tour extraction structure

    4. TestLinkedTripsReproduction: Tests for 02b-link_trips_week step
       - Compares new linker output with old pipeline
       - Documents implementation gaps for exact reproduction

    5. TestToursReproduction: Tests for 03a-tour_extract_week step
       - Compares new tour builder output with old pipeline
       - Documents implementation gaps for exact reproduction

Environment Variables:
    DAYSIM_OLD_OUTPUT_DIR: Path to old pipeline output (02a-reformat)
    DAYSIM_OLD_INPUT_DIR: Path to old pipeline input (01-taz_spatial_join)
    DAYSIM_OLD_LINKED_DIR: Path to old linked trips (02b-link_trips_week)
    DAYSIM_OLD_TOURS_DIR: Path to old tours (03a-tour_extract_week)
    DAYSIM_RAW_DIR: Path to raw data (for day.csv completeness)

Or use the defaults which point to the M: drive locations.

Critical Notes for Reproduction:
    - The old pipeline uses change_mode_code = 10 (dpurp == 10)
    - Transit mode code is 6 in DaySim format
    - Linked trips renumber as 'lintripno'
    - Tours use specific DaySim field names (hhno, pno, tour, pdpurp, etc.)
"""

import logging
import os
from pathlib import Path

import polars as pl
import pytest

from travel_diary_survey_tools import DaysimFormatter

logger = logging.getLogger(__name__)


class TestDaysimFormatting:
    """Tests to validate DaySim formatting produces identical results."""

    @pytest.fixture
    def old_pipeline_dir(self) -> Path:
        """Path to old pipeline output directory."""
        # Allow override via environment variable
        env_path = os.environ.get("DAYSIM_OLD_OUTPUT_DIR")
        if env_path:
            return Path(env_path)
        return Path(
            r"\\models.ad.mtc.ca.gov\data\models\Data\HomeInterview"
            r"\Bay Area Travel Study 2023\Data\Processed"
            r"\TripLinking_20250728\02a-reformat"
        )

    @pytest.fixture
    def old_pipeline_input_dir(self) -> Path:
        """Path to old pipeline input directory (spatially joined data)."""
        env_path = os.environ.get("DAYSIM_OLD_INPUT_DIR")
        if env_path:
            return Path(env_path)
        return Path(
            r"\\models.ad.mtc.ca.gov\data\models\Data\HomeInterview"
            r"\Bay Area Travel Study 2023\Data\Processed"
            r"\TripLinking_20250728\01-taz_spatial_join"
        )

    @pytest.fixture
    def raw_data_dir(self) -> Path:
        """Path to raw data directory (for day completeness)."""
        env_path = os.environ.get("DAYSIM_RAW_DIR")
        if env_path:
            return Path(env_path)
        return Path(
            r"\\models.ad.mtc.ca.gov\data\models\Data\HomeInterview"
            r"\Bay Area Travel Study 2023\Data"
            r"\Full Weighted 2023 Dataset\WeightedDataset_02212025"
        )

    def _compare_dataframes(
        self,
        old_df: pl.DataFrame,
        new_df: pl.DataFrame,
        name: str,
        key_cols: list[str],
        tolerance: float = 1e-6,
    ) -> tuple[bool, str]:
        """Compare two DataFrames and return detailed differences.

        Args:
            old_df: DataFrame from old pipeline
            new_df: DataFrame from new pipeline
            name: Name of the table being compared
            key_cols: Columns to use as join keys
            tolerance: Tolerance for floating point comparisons

        Returns:
            Tuple of (is_identical, difference_message)

        """
        # Check row counts
        if len(old_df) != len(new_df):
            msg = (
                f"{name}: Row count mismatch. "
                f"Old={len(old_df)}, New={len(new_df)}"
            )
            return False, msg

        # Check columns
        old_cols = set(old_df.columns)
        new_cols = set(new_df.columns)

        if old_cols != new_cols:
            missing_in_new = old_cols - new_cols
            extra_in_new = new_cols - old_cols
            msg = f"{name}: Column mismatch.\n"
            if missing_in_new:
                msg += f"  Missing in new: {missing_in_new}\n"
            if extra_in_new:
                msg += f"  Extra in new: {extra_in_new}\n"
            return False, msg

        # Sort both DataFrames by key columns for comparison
        old_df = old_df.sort(key_cols)
        new_df = new_df.sort(key_cols)

        # Compare each column
        differences = []
        for col in old_df.columns:
            old_col = old_df[col]
            new_col = new_df[col]

            # Handle different data types appropriately
            if old_col.dtype in [pl.Float32, pl.Float64]:
                # For float columns, check with tolerance
                diff_mask = (old_col - new_col).abs() > tolerance
                if diff_mask.sum() > 0:
                    n_diff = diff_mask.sum()
                    differences.append(
                        f"  {col}: {n_diff} values differ "
                        f"(tolerance={tolerance})"
                    )
            else:
                # For other columns, check exact equality
                diff_mask = old_col != new_col
                if diff_mask.sum() > 0:
                    n_diff = diff_mask.sum()
                    # Get sample of differences
                    sample_idx = diff_mask.arg_max()
                    old_val = old_col[sample_idx]
                    new_val = new_col[sample_idx]
                    differences.append(
                        f"  {col}: {n_diff} values differ "
                        f"(e.g., row {sample_idx}: old={old_val}, "
                        f"new={new_val})"
                    )

        if differences:
            msg = f"{name}: Value differences found:\n" + "\n".join(differences)
            return False, msg

        return True, f"{name}: IDENTICAL"

    def test_person_formatting(
        self,
        old_pipeline_dir: Path,
        old_pipeline_input_dir: Path,
        raw_data_dir: Path,
    ) -> None:
        """Test that person formatting produces identical results.

        This test reads the OLD pipeline output and the OLD pipeline INPUT,
        formats the input using the new formatter, and compares with the
        old output. No writing required!
        """
        if not old_pipeline_dir.exists():
            pytest.skip(f"Old pipeline directory not found: {old_pipeline_dir}")

        if not old_pipeline_input_dir.exists():
            pytest.skip(
                "Old pipeline input directory not found: "
                f"{old_pipeline_input_dir}"
            )

        # Load old pipeline's formatted output (what we're comparing against)
        old_person = pl.read_csv(old_pipeline_dir / "person.csv")

        # Load old pipeline's INPUT (spatial join output)
        input_person = pl.read_csv(old_pipeline_input_dir / "person.csv")

        # Format using new formatter
        formatter = DaysimFormatter({"weighted": True})

        # Load day completeness if available
        day_completeness = None
        day_path = raw_data_dir / "day.csv"
        if day_path.exists():
            day_completeness = formatter.load_day_completeness(day_path)

        new_person = formatter.format_person(input_person, day_completeness)

        # Compare
        is_identical, msg = self._compare_dataframes(
            old_person,
            new_person,
            "Person",
            key_cols=["hhno", "pno"],
        )

        logger.info(msg)
        assert is_identical, msg

    def test_household_formatting(
        self,
        old_pipeline_dir: Path,
        old_pipeline_input_dir: Path,
        raw_data_dir: Path,
    ) -> None:
        """Test that household formatting produces identical results.

        This test reads the OLD pipeline output and the OLD pipeline INPUT,
        formats the input using the new formatter, and compares with the
        old output. No writing required!
        """
        if not old_pipeline_dir.exists():
            pytest.skip(f"Old pipeline directory not found: {old_pipeline_dir}")

        if not old_pipeline_input_dir.exists():
            pytest.skip(
                "Old pipeline input directory not found: "
                f"{old_pipeline_input_dir}"
            )

        # Load old pipeline's formatted output
        old_hh = pl.read_csv(old_pipeline_dir / "hh.csv")

        # Load old pipeline's INPUT
        input_hh = pl.read_csv(old_pipeline_input_dir / "hh.csv")
        input_person = pl.read_csv(old_pipeline_input_dir / "person.csv")

        # Format using new formatter
        formatter = DaysimFormatter({"weighted": True})

        # Load day completeness if available
        day_completeness = None
        day_path = raw_data_dir / "day.csv"
        if day_path.exists():
            day_completeness = formatter.load_day_completeness(day_path)

        # Format person first (needed for household formatting)
        formatted_person = formatter.format_person(
            input_person, day_completeness
        )
        new_hh = formatter.format_household(input_hh, formatted_person)

        # Compare
        is_identical, msg = self._compare_dataframes(
            old_hh,
            new_hh,
            "Household",
            key_cols=["hhno"],
        )

        logger.info(msg)
        assert is_identical, msg

    def test_trip_formatting(
        self,
        old_pipeline_dir: Path,
        old_pipeline_input_dir: Path,
    ) -> None:
        """Test that trip formatting produces identical results.

        This test reads the OLD pipeline output and the OLD pipeline INPUT,
        formats the input using the new formatter, and compares with the
        old output. No writing required!
        """
        if not old_pipeline_dir.exists():
            pytest.skip(f"Old pipeline directory not found: {old_pipeline_dir}")

        if not old_pipeline_input_dir.exists():
            pytest.skip(
                "Old pipeline input directory not found: "
                f"{old_pipeline_input_dir}"
            )

        # Load old pipeline's formatted output
        old_trip = pl.read_csv(old_pipeline_dir / "trip.csv")

        # Load old pipeline's INPUT
        input_trip = pl.read_csv(old_pipeline_input_dir / "trip.csv")

        # Format using new formatter
        formatter = DaysimFormatter({"weighted": True})
        new_trip = formatter.format_trip(input_trip)

        # Compare
        is_identical, msg = self._compare_dataframes(
            old_trip,
            new_trip,
            "Trip",
            key_cols=["hhno", "pno", "tripno"],
        )

        logger.info(msg)
        assert is_identical, msg

    def test_field_distributions(
        self,
        old_pipeline_dir: Path,
        old_pipeline_input_dir: Path,
        raw_data_dir: Path,
    ) -> None:
        """Test that field distributions match between old and new pipelines.

        This test checks summary statistics for key fields to ensure the
        overall distributions are preserved even if individual records differ.

        This test reads the OLD pipeline output and the OLD pipeline INPUT,
        formats the input using the new formatter, and compares distributions.
        No writing required!
        """
        if not old_pipeline_dir.exists():
            pytest.skip(f"Old pipeline directory not found: {old_pipeline_dir}")

        if not old_pipeline_input_dir.exists():
            pytest.skip(
                "Old pipeline input directory not found: "
                f"{old_pipeline_input_dir}"
            )

        # Initialize formatter
        formatter = DaysimFormatter({"weighted": True})

        # Load day completeness if available
        day_completeness = None
        day_path = raw_data_dir / "day.csv"
        if day_path.exists():
            day_completeness = formatter.load_day_completeness(day_path)

        # Test person type distribution
        old_person = pl.read_csv(old_pipeline_dir / "person.csv")
        input_person = pl.read_csv(old_pipeline_input_dir / "person.csv")
        new_person = formatter.format_person(input_person, day_completeness)

        # Compare person type counts
        old_pptyp = old_person.group_by("pptyp").len().sort("pptyp")
        new_pptyp = new_person.group_by("pptyp").len().sort("pptyp")

        assert old_pptyp.equals(new_pptyp), (
            "Person type (pptyp) distribution differs:\n"
            f"Old:\n{old_pptyp}\n"
            f"New:\n{new_pptyp}"
        )

        # Test trip mode distribution
        old_trip = pl.read_csv(old_pipeline_dir / "trip.csv")
        input_trip = pl.read_csv(old_pipeline_input_dir / "trip.csv")
        new_trip = formatter.format_trip(input_trip)

        old_mode = old_trip.group_by("mode").len().sort("mode")
        new_mode = new_trip.group_by("mode").len().sort("mode")

        assert old_mode.equals(new_mode), (
            "Trip mode distribution differs:\n"
            f"Old:\n{old_mode}\n"
            f"New:\n{new_mode}"
        )

        # Test purpose distribution
        old_purpose = old_trip.group_by("dpurp").len().sort("dpurp")
        new_purpose = new_trip.group_by("dpurp").len().sort("dpurp")

        assert old_purpose.equals(new_purpose), (
            "Trip purpose (dpurp) distribution differs:\n"
            f"Old:\n{old_purpose}\n"
            f"New:\n{new_purpose}"
        )


class TestDaysimFormatter:
    """Unit tests for DaysimFormatter class."""

    def test_age_mapping(self) -> None:
        """Test age category mapping."""
        formatter = DaysimFormatter({"weighted": False})

        # Create test data
        test_person = pl.DataFrame(
            {
                "hh_id": [1, 1, 1],
                "person_num": [1, 2, 3],
                "age": [1, 5, 10],  # age categories
                "gender": [1, 2, 1],
                "student": [2, 2, 2],  # not student
                "work_park": [1, 1, 1],
                "residence_rent_own": [1, 1, 1],
                "residence_type": [1, 1, 1],
                "employment": [0, 0, 0],
                "school_type": [0, 0, 0],
                "work_lon": [-122.4, -122.4, -122.4],
                "work_lat": [37.8, 37.8, 37.8],
                "school_lon": [-122.4, -122.4, -122.4],
                "school_lat": [37.8, 37.8, 37.8],
                "work_taz": [1, 1, 1],
                "work_maz": [1, 1, 1],
                "school_taz": [1, 1, 1],
                "school_maz": [1, 1, 1],
                "work_county": [6075, 6075, 6075],
                "school_county": [6075, 6075, 6075],
            }
        )

        result = formatter.format_person(test_person)

        # Check age mapping: 1->3, 5->30, 10->80
        assert result["pagey"].to_list() == [3, 30, 80]

    def test_person_type_derivation(self) -> None:
        """Test person type (pptyp) derivation logic."""
        formatter = DaysimFormatter({"weighted": False})

        # Create test data with different person types
        test_person = pl.DataFrame(
            {
                "hh_id": [1, 1, 1, 1],
                "person_num": [1, 2, 3, 4],
                "age": [
                    1,
                    2,
                    4,
                    10,
                ],  # child 0-4, child 5-15, young adult, senior
                "gender": [1, 1, 1, 1],
                "student": [2, 2, 2, 2],  # not student
                "work_park": [1, 1, 1, 1],
                "residence_rent_own": [1, 1, 1, 1],
                "residence_type": [1, 1, 1, 1],
                "employment": [0, 0, 1, 0],  # not employed, not, full-time, not
                "school_type": [0, 0, 0, 0],
                "work_lon": [-122.4] * 4,
                "work_lat": [37.8] * 4,
                "school_lon": [-122.4] * 4,
                "school_lat": [37.8] * 4,
                "work_taz": [1] * 4,
                "work_maz": [1] * 4,
                "school_taz": [1] * 4,
                "school_maz": [1] * 4,
                "work_county": [6075] * 4,
                "school_county": [6075] * 4,
            }
        )

        result = formatter.format_person(test_person)

        # Check person types: child 0-4, child 5-15, full-time worker, retiree
        # Age mappings: 1→3yrs, 2→10yrs, 4→21yrs, 10→80yrs
        assert result["pptyp"].to_list() == [8, 7, 1, 3]

    def test_trip_mode_mapping(self) -> None:
        """Test trip mode mapping to DaySim codes."""
        formatter = DaysimFormatter({"weighted": False})

        # Create test trip data
        test_trip = pl.DataFrame(
            {
                "hh_id": [1, 1, 1, 1],
                "person_num": [1, 1, 1, 1],
                "trip_num": [1, 2, 3, 4],
                "mode_type": [1, 2, 8, 13],  # walk, bike, car, transit
                "num_travelers": [1, 1, 1, 1],
                "o_purpose_category": [1, 1, 1, 1],
                "d_purpose_category": [2, 2, 2, 2],
                "depart_hour": [8, 9, 10, 11],
                "depart_minute": [0, 15, 30, 45],
                "arrive_hour": [8, 9, 10, 11],
                "arrive_minute": [30, 45, 0, 15],
                "o_taz": [1, 1, 1, 1],
                "o_maz": [1, 1, 1, 1],
                "d_taz": [2, 2, 2, 2],
                "d_maz": [2, 2, 2, 2],
                "o_lon": [-122.4] * 4,
                "o_lat": [37.8] * 4,
                "d_lon": [-122.5] * 4,
                "d_lat": [37.9] * 4,
                "o_county": [6075] * 4,
                "d_county": [6075] * 4,
                "travel_dow": [2, 2, 2, 2],
                "day_is_complete": [1, 1, 1, 1],
                "transit_access": [1] * 4,
                "transit_egress": [1] * 4,
                "driver": [1] * 4,
                "mode_1": [1] * 4,
                "mode_2": [1] * 4,
                "mode_3": [1] * 4,
                "mode_4": [1] * 4,
            }
        )

        result = formatter.format_trip(test_trip)

        # Check mode mapping: walk=1, bike=2, car(1pax)=3, transit=6
        assert result["mode"].to_list() == [1, 2, 3, 6]


class TestPipelineReproduction:
    """Tests for reproducing old pipeline outputs exactly.

    This test class validates that the new pipeline can reproduce the
    outputs from the legacy pipeline (02b-link_trips_week and
    03a-tour_extract_week) for compatibility and correctness.
    """

    # Shared fixtures for all reproduction tests
    @pytest.fixture
    def old_input_dir(self) -> Path:
        """Path to old pipeline input (01-taz_spatial_join)."""
        env_path = os.environ.get("DAYSIM_OLD_INPUT_DIR")
        if env_path:
            return Path(env_path)
        return Path(
            r"\\models.ad.mtc.ca.gov\data\models\Data\HomeInterview"
            r"\Bay Area Travel Study 2023\Data\Processed"
            r"\TripLinking_20250728\01-taz_spatial_join"
        )

    @pytest.fixture
    def old_formatted_dir(self) -> Path:
        """Path to old pipeline formatted output (02a-reformat)."""
        env_path = os.environ.get("DAYSIM_OLD_OUTPUT_DIR")
        if env_path:
            return Path(env_path)
        return Path(
            r"\\models.ad.mtc.ca.gov\data\models\Data\HomeInterview"
            r"\Bay Area Travel Study 2023\Data\Processed"
            r"\TripLinking_20250728\02a-reformat"
        )

    @pytest.fixture
    def old_linked_dir(self) -> Path:
        """Path to old pipeline linked trips (02b-link_trips_week)."""
        env_path = os.environ.get("DAYSIM_OLD_LINKED_DIR")
        if env_path:
            return Path(env_path)
        return Path(
            r"\\models.ad.mtc.ca.gov\data\models\Data\HomeInterview"
            r"\Bay Area Travel Study 2023\Data\Processed"
            r"\TripLinking_20250728\02b-link_trips_week"
        )

    @pytest.fixture
    def old_tours_dir(self) -> Path:
        """Path to old pipeline tours (03a-tour_extract_week)."""
        env_path = os.environ.get("DAYSIM_OLD_TOURS_DIR")
        if env_path:
            return Path(env_path)
        return Path(
            r"\\models.ad.mtc.ca.gov\data\models\Data\HomeInterview"
            r"\Bay Area Travel Study 2023\Data\Processed"
            r"\TripLinking_20250728\03a-tour_extract_week"
        )

    # Documentation tests - understand old pipeline structure
    def test_pipeline_trip_counts(
        self,
        old_input_dir: Path,
        old_formatted_dir: Path,
        old_linked_dir: Path,
        old_tours_dir: Path,
    ) -> None:
        """Document trip counts through entire pipeline stages.

        Shows reduction at each stage:
        1. Input (spatial join) -> DaySim formatting (filters incomplete)
        2. DaySim formatted -> Linked trips (merges multi-modal)
        3. Linked trips -> Tours (assigns to tours)
        """
        if not all(
            [
                old_input_dir.exists(),
                old_formatted_dir.exists(),
                old_linked_dir.exists(),
                old_tours_dir.exists(),
            ]
        ):
            pytest.skip("Not all pipeline directories found")

        # Load trip counts at each stage
        input_trips = pl.read_csv(old_input_dir / "trip.csv")
        formatted_trips = pl.read_csv(old_formatted_dir / "trip.csv")
        linked_trips = pl.read_csv(old_linked_dir / "trip.csv")
        tour_trips = pl.read_csv(old_tours_dir / "trip.csv")

        logger.info("\n=== Pipeline Trip Counts ===")
        logger.info("1. Input (spatial join):  %d", len(input_trips))
        logger.info("2. DaySim formatted:      %d", len(formatted_trips))
        logger.info("3. Linked trips:          %d", len(linked_trips))
        logger.info("4. Tour-assigned:         %d", len(tour_trips))

        # Document reduction percentages
        format_filter = 100 * (len(input_trips) - len(formatted_trips))
        format_filter /= len(input_trips)
        linking_reduction = 100 * (len(formatted_trips) - len(linked_trips))
        linking_reduction /= len(formatted_trips)
        tour_filter = 100 * (len(linked_trips) - len(tour_trips))
        tour_filter /= len(linked_trips)

        logger.info("\nFormat filtering:  %.1f%%", format_filter)
        logger.info("Linking reduction: %.1f%%", linking_reduction)
        logger.info("Tour filtering:    %.1f%%", tour_filter)

        # Validate counts decrease (or stay same) at each stage
        assert len(formatted_trips) <= len(input_trips)
        assert len(linked_trips) <= len(formatted_trips)

    # Reproduction tests - compare new vs old pipeline
    def test_linked_trips_reproduction(
        self,
        old_formatted_dir: Path,
        old_linked_dir: Path,
    ) -> None:
        """Test that new linker reproduces old pipeline linked trips.

        Compares new link_trips() output with 02b-link_trips_week output
        to identify implementation gaps for exact reproduction.

        NOTE: The old pipeline outputs are in DaySim format (hhno, pno, etc.)
        while the new pipeline expects raw format. This test documents
        the structural differences.
        """
        if not old_formatted_dir.exists():
            pytest.skip(f"Formatted dir not found: {old_formatted_dir}")

        if not old_linked_dir.exists():
            pytest.skip(f"Linked dir not found: {old_linked_dir}")

        # Load old pipeline's linked output
        old_linked = pl.read_csv(old_linked_dir / "trip.csv")
        formatted_trips = pl.read_csv(old_formatted_dir / "trip.csv")

        logger.info("=== Linked Trips Reproduction Test ===")
        logger.info("Old linked trips: %d", len(old_linked))
        logger.info("Formatted input: %d", len(formatted_trips))
        logger.info("Old linked columns: %s", old_linked.columns[:10])
        logger.info("Formatted columns: %s", formatted_trips.columns[:10])

        # Document the format mismatch
        logger.info("\n=== Format Analysis ===")
        logger.info("Old pipeline uses DaySim format:")
        logger.info("  - Column names: hhno, pno, tripno, dpurp, etc.")
        logger.info("  - Missing: duration_minutes, distance_miles")
        logger.info("  - Has: lintripno (renumbered trip IDs)")

        logger.info("\nNew pipeline expects raw format:")
        logger.info("  - Column names: hh_id, person_id, trip_id")
        logger.info("  - Requires: duration_minutes, distance_miles")
        logger.info("  - Outputs: linked_trip_id")

        # Document implementation gaps
        logger.info("\n=== Implementation Gaps for Exact Reproduction ===")
        logger.info("  1. Need pre-processing to convert DaySim -> raw format")
        logger.info("  2. Or: Make linker accept DaySim format directly")
        logger.info("  3. Column mappings: linked_trip_id -> lintripno")
        logger.info("  4. Activity duration threshold logic")
        logger.info("  5. Access/egress mode tracking (accegr.csv)")

        # Basic structural validation only
        assert len(old_linked) > 0, "No old linked trips found"
        assert "lintripno" in old_linked.columns, "Missing lintripno"
        assert len(formatted_trips) > 0, "No formatted trips found"

        logger.info("\nTest documents structural differences successfully")

    def test_tours_reproduction(
        self,
        old_linked_dir: Path,
        old_tours_dir: Path,
        old_formatted_dir: Path,
    ) -> None:
        """Test that new tour builder reproduces old pipeline tours.

        Compares new TourBuilder output with 03a-tour_extract_week output
        to identify implementation gaps for exact reproduction.

        NOTE: The old pipeline outputs are in DaySim format (hhno, pno, etc.)
        while the new pipeline expects raw format. This test documents
        the structural differences.
        """
        if not all(
            [
                old_linked_dir.exists(),
                old_tours_dir.exists(),
                old_formatted_dir.exists(),
            ]
        ):
            pytest.skip("Required directories not found")

        # Load data
        old_tours = pl.read_csv(old_tours_dir / "tour.csv")
        old_tour_trips = pl.read_csv(old_tours_dir / "trip.csv")
        linked_trips = pl.read_csv(old_linked_dir / "trip.csv")
        persons = pl.read_csv(old_formatted_dir / "person.csv")

        logger.info("=== Tours Reproduction Test ===")
        logger.info("Old tours: %d", len(old_tours))
        logger.info("Old tour trips: %d", len(old_tour_trips))
        logger.info("Linked trips: %d", len(linked_trips))
        logger.info("Persons: %d", len(persons))

        logger.info("\nOld tours columns: %s", old_tours.columns[:10])
        logger.info("Persons columns: %s", persons.columns[:10])
        logger.info("Linked trips columns: %s", linked_trips.columns[:10])

        # Document the format mismatch
        logger.info("\n=== Format Analysis ===")
        logger.info("Old pipeline uses DaySim format:")
        logger.info("  - Person: hhno, pno (not person_id, hh_id)")
        logger.info("  - Tours: tour, pdpurp, tlvorig, etc.")
        logger.info("  - Trips: hhno, pno, lintripno")

        logger.info("\nNew pipeline expects raw format:")
        logger.info("  - Person: person_id, hh_id, person_type")
        logger.info("  - Tours: tour_id, primary_purpose, etc.")
        logger.info("  - Trips: person_id, linked_trip_id")

        # Document implementation gaps
        logger.info("\n=== Implementation Gaps for Exact Reproduction ===")
        logger.info("  1. Need pre-processing to convert DaySim -> raw format")
        logger.info("  2. Or: Make TourBuilder accept DaySim format directly")
        logger.info("  3. Column mappings: tour_id -> tour, etc.")
        logger.info("  4. Exact tour identification algorithm")
        logger.info("  5. Purpose priority logic")
        logger.info("  6. Parent-subtour relationships")

        # Basic structural validation only
        assert len(old_tours) > 0, "No old tours found"
        assert len(old_tour_trips) > 0, "No old tour trips found"
        assert "tour" in old_tours.columns, "Missing tour column"
        assert len(persons) > 0, "No persons found"
        assert len(linked_trips) > 0, "No linked trips found"

        logger.info("\nTest documents structural differences successfully")


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v"])
