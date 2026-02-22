"""Unit tests for Pydantic schemas.

Tests cover:
- Field validation
- Custom validators
- Model configuration
- Serialization/deserialization
"""

import pytest
from pydantic import ValidationError

from api.schemas import (
    PredictionInput,
    BatchPredictionInput,
    ScenarioInput,
    HealthResponse,
    PredictionOutput,
    YearRangeParams,
)


class TestPredictionInput:
    """Tests for PredictionInput schema."""

    def test_valid_minimal_input(self):
        """Test minimal valid input."""
        data = PredictionInput(year=2025)
        assert data.year == 2025
        assert data.water_productivity is None
        assert data.model_name is None

    def test_valid_full_input(self):
        """Test full valid input."""
        data = PredictionInput(
            year=2025,
            water_productivity=8.5,
            freshwater_withdrawals_agriculture=75.0,
            freshwater_withdrawals_industry=5.0,
            renewable_freshwater_resources=4.195,
            model_name="Lasso",
        )
        assert data.year == 2025
        assert data.water_productivity == 8.5
        assert data.model_name == "Lasso"

    def test_year_below_minimum(self):
        """Test year below minimum raises error."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionInput(year=1900)

        errors = exc_info.value.errors()
        assert any("year" in str(e["loc"]) for e in errors)

    def test_year_above_maximum(self):
        """Test year above maximum raises error."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionInput(year=2200)

        errors = exc_info.value.errors()
        assert any("year" in str(e["loc"]) for e in errors)

    def test_water_productivity_negative(self):
        """Test negative water productivity raises error."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionInput(year=2025, water_productivity=-1.0)

        errors = exc_info.value.errors()
        assert any("water_productivity" in str(e["loc"]) for e in errors)

    def test_withdrawal_percentages_exceed_100(self):
        """Test combined withdrawals exceeding 100% raises error."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionInput(
                year=2025,
                freshwater_withdrawals_agriculture=80.0,
                freshwater_withdrawals_industry=30.0,
            )

        errors = exc_info.value.errors()
        assert any("100%" in str(e) for e in errors)

    def test_extra_fields_rejected(self):
        """Test extra fields are rejected."""
        with pytest.raises(ValidationError):
            PredictionInput(year=2025, unknown_field="value")

    def test_model_name_stripped(self):
        """Test model name whitespace is stripped."""
        data = PredictionInput(year=2025, model_name="  Lasso  ")
        assert data.model_name == "Lasso"

    def test_empty_model_name_becomes_none(self):
        """Test empty model name becomes None."""
        data = PredictionInput(year=2025, model_name="   ")
        assert data.model_name is None


class TestBatchPredictionInput:
    """Tests for BatchPredictionInput schema."""

    def test_valid_batch(self):
        """Test valid batch input."""
        data = BatchPredictionInput(
            predictions=[
                PredictionInput(year=2025),
                PredictionInput(year=2026),
            ]
        )
        assert len(data.predictions) == 2

    def test_empty_batch_rejected(self):
        """Test empty batch is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionInput(predictions=[])

        errors = exc_info.value.errors()
        assert any("predictions" in str(e["loc"]) for e in errors)

    def test_batch_too_large(self):
        """Test batch exceeding max size is rejected."""
        predictions = [PredictionInput(year=2000 + i) for i in range(101)]

        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionInput(predictions=predictions)

        errors = exc_info.value.errors()
        assert any("predictions" in str(e["loc"]) for e in errors)


class TestScenarioInput:
    """Tests for ScenarioInput schema."""

    def test_default_values(self):
        """Test default values are set correctly."""
        data = ScenarioInput()
        assert data.target_year == 2030
        assert data.trend_method == "linear"
        assert data.model_name is None

    def test_valid_trend_methods(self):
        """Test all valid trend methods are accepted."""
        for method in ["linear", "exponential", "last_value", "average"]:
            data = ScenarioInput(trend_method=method)
            assert data.trend_method == method

    def test_invalid_trend_method(self):
        """Test invalid trend method raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ScenarioInput(trend_method="invalid")

        errors = exc_info.value.errors()
        assert any("trend_method" in str(e["loc"]) for e in errors)

    def test_target_year_below_minimum(self):
        """Test target year below minimum raises error."""
        with pytest.raises(ValidationError):
            ScenarioInput(target_year=2010)


class TestYearRangeParams:
    """Tests for YearRangeParams schema."""

    def test_valid_range(self):
        """Test valid year range."""
        data = YearRangeParams(start_year=2020, end_year=2030)
        assert data.start_year == 2020
        assert data.end_year == 2030

    def test_start_after_end(self):
        """Test start year after end year raises error."""
        with pytest.raises(ValidationError) as exc_info:
            YearRangeParams(start_year=2030, end_year=2020)

        errors = exc_info.value.errors()
        assert any("start_year" in str(e["msg"]).lower() for e in errors)

    def test_range_too_large(self):
        """Test range exceeding maximum raises error."""
        with pytest.raises(ValidationError) as exc_info:
            YearRangeParams(start_year=1960, end_year=2100)

        errors = exc_info.value.errors()
        assert any("100" in str(e["msg"]) for e in errors)


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_valid_response(self):
        """Test valid health response."""
        data = HealthResponse(status="healthy", version="1.0.0", models_loaded=5)
        assert data.status == "healthy"
        assert data.version == "1.0.0"
        assert data.models_loaded == 5

    def test_invalid_status(self):
        """Test invalid status value raises error."""
        with pytest.raises(ValidationError):
            HealthResponse(status="invalid_status", version="1.0.0", models_loaded=5)

    def test_negative_models_loaded(self):
        """Test negative models_loaded raises error."""
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy", version="1.0.0", models_loaded=-1)


class TestPredictionOutput:
    """Tests for PredictionOutput schema."""

    def test_valid_output(self):
        """Test valid prediction output."""
        data = PredictionOutput(
            year=2025,
            predicted_water_stress=85.5,
            model_used="Lasso",
            input_features={"year": 2025, "water_productivity": 8.5},
        )
        assert data.year == 2025
        assert data.predicted_water_stress == 85.5
        assert "confidence_note" in data.model_dump()

    def test_default_confidence_note(self):
        """Test default confidence note is set."""
        data = PredictionOutput(
            year=2025, predicted_water_stress=85.5, model_used="Lasso", input_features={}
        )
        assert "historical patterns" in data.confidence_note.lower()
