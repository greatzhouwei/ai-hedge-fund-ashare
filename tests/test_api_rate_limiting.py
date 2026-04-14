import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.tools import api as api_module
from src.data.models import Price, FinancialMetrics, LineItem, InsiderTrade, CompanyNews


class TestTushareClient:
    """Test suite for Tushare API client behavior."""

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_missing_token_raises_value_error(self):
        """Test that missing TUSHARE_TOKEN raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            api_module._pro_api = None
            with pytest.raises(ValueError, match="TUSHARE_TOKEN is not set"):
                api_module._get_pro_api()

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_get_prices_empty_dataframe_returns_empty_list(self):
        """Test that empty DataFrame from Tushare returns empty list."""
        mock_pro = Mock()
        mock_pro.daily.return_value = pd.DataFrame()

        with patch.object(api_module, "_pro_api", mock_pro):
            result = api_module.get_prices("000001.SZ", "2024-01-01", "2024-01-10")
            assert result == []
            mock_pro.daily.assert_called_once_with(
                ts_code="000001.SZ",
                start_date="20240101",
                end_date="20240110",
            )

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_get_prices_network_error_returns_empty_list(self):
        """Test that network error returns empty list without crashing."""
        mock_pro = Mock()
        mock_pro.daily.side_effect = Exception("Network timeout")

        with patch.object(api_module, "_pro_api", mock_pro):
            result = api_module.get_prices("000001.SZ", "2024-01-01", "2024-01-10")
            assert result == []

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_get_prices_parses_records_correctly(self):
        """Test that valid DataFrame is parsed into Price objects."""
        mock_pro = Mock()
        mock_pro.daily.return_value = pd.DataFrame([
            {
                "trade_date": "20240102",
                "open": 10.5,
                "close": 11.0,
                "high": 11.2,
                "low": 10.3,
                "vol": 50000,
            }
        ])

        with patch.object(api_module, "_pro_api", mock_pro):
            result = api_module.get_prices("000001.SZ", "2024-01-01", "2024-01-10")
            assert len(result) == 1
            assert isinstance(result[0], Price)
            assert result[0].open == 10.5
            assert result[0].close == 11.0
            assert result[0].volume == 50000
            assert result[0].time == "2024-01-02"

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_get_financial_metrics_empty_dataframe_returns_empty_list(self):
        """Test that empty fina_indicator DataFrame returns empty list."""
        mock_pro = Mock()
        mock_pro.fina_indicator.return_value = pd.DataFrame()
        mock_pro.daily_basic.return_value = pd.DataFrame()

        with patch.object(api_module, "_pro_api", mock_pro):
            result = api_module.get_financial_metrics("000001.SZ", "2024-03-31")
            assert result == []

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_get_market_cap_empty_dataframe_returns_none(self):
        """Test that empty daily_basic DataFrame returns None for market cap."""
        mock_pro = Mock()
        mock_pro.daily_basic.return_value = pd.DataFrame()

        with patch.object(api_module, "_pro_api", mock_pro):
            result = api_module.get_market_cap("000001.SZ", "2024-01-10")
            assert result is None

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_get_market_cap_converts_total_mv(self):
        """Test that total_mv (万元) is converted to yuan (*10000)."""
        mock_pro = Mock()
        mock_pro.daily_basic.return_value = pd.DataFrame([
            {"trade_date": "20240110", "total_mv": 12345.67}
        ])

        with patch.object(api_module, "_pro_api", mock_pro):
            result = api_module.get_market_cap("000001.SZ", "2024-01-10")
            assert result == 12345.67 * 10000

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_search_line_items_unknown_items_return_empty(self):
        """Test that unknown line items return empty list with warning."""
        mock_pro = Mock()
        mock_pro.income.return_value = pd.DataFrame()
        mock_pro.balancesheet.return_value = pd.DataFrame()
        mock_pro.cashflow.return_value = pd.DataFrame()

        with patch.object(api_module, "_pro_api", mock_pro):
            result = api_module.search_line_items(
                "000001.SZ",
                ["totally_unknown_item"],
                "2024-03-31",
            )
            assert result == []

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_search_line_items_maps_known_items(self):
        """Test that known line items are mapped to correct Tushare tables."""
        mock_pro = Mock()
        mock_pro.income.return_value = pd.DataFrame([
            {"ts_code": "000001.SZ", "end_date": "20240331", "revenue": 999.0}
        ])

        with patch.object(api_module, "_pro_api", mock_pro):
            result = api_module.search_line_items(
                "000001.SZ",
                ["revenue"],
                "2024-03-31",
            )
            assert len(result) == 1
            assert result[0].ticker == "000001.SZ"
            assert result[0].revenue == 999.0

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_get_insider_trades_empty_dataframe_returns_empty_list(self):
        """Test that empty stk_holdertrade DataFrame returns empty list."""
        mock_pro = Mock()
        mock_pro.stk_holdertrade.return_value = pd.DataFrame()

        with patch.object(api_module, "_pro_api", mock_pro):
            result = api_module.get_insider_trades("000001.SZ", "2024-01-10")
            assert result == []

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_get_company_news_returns_empty_list(self):
        """Test that get_company_news always returns empty list."""
        result = api_module.get_company_news("000001.SZ", "2024-01-10")
        assert result == []

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_to_tushare_date_conversion(self):
        """Test YYYY-MM-DD to YYYYMMDD conversion."""
        assert api_module._to_tushare_date("2024-01-15") == "20240115"

    @patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}, clear=True)
    def test_from_tushare_date_conversion(self):
        """Test YYYYMMDD to YYYY-MM-DD conversion."""
        assert api_module._from_tushare_date("20240115") == "2024-01-15"


if __name__ == "__main__":
    pytest.main([__file__])
