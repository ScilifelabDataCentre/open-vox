"""Tests for ovox CLI."""

from unittest.mock import AsyncMock, patch

from ovox.cli import build_parser, main


def test_build_parser_has_check_command():
    """Test that the argument parser includes the check command."""
    parser = build_parser()
    # Check if 'check' is in the list of subcommands
    assert "check" in parser._subparsers._group_actions[0].choices

    # Check the help text for the check command
    check_parser = parser._subparsers._group_actions[0].choices["check"]
    assert "STT API is reachable and credentials are valid" in check_parser.description


def test_check_command_success(capsys, stt_env):
    """Test the check command with a successful API connection."""
    # Mock the STTClient.check_connection method to return success
    with patch("ovox.api.STTClient") as mock_client_class:
        # Create a mock client instance
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client_class.return_value = mock_client

        # Configure the check_connection method to return success
        mock_client.check_connection.return_value = {
            "success": True,
            "message": "Successfully connected to STT API",
            "status_code": 200,
        }

        # Run the main function with the check command
        with patch.dict(
            "os.environ",
            {"STT_ENDPOINT": stt_env["STT_ENDPOINT"], "STT_API_KEY": stt_env["STT_API_KEY"]},
        ):
            result = main(["check"])

        # Assert the result
        assert result == 0

        # Capture the output
        captured = capsys.readouterr()
        assert "Successfully connected to STT API" in captured.out

        # Assert the mock was called
        mock_client.check_connection.assert_called_once()


def test_check_command_failure(capsys, stt_env):
    """Test the check command with a failed API connection."""
    # Mock the STTClient.check_connection method to return failure
    with patch("ovox.api.STTClient") as mock_client_class:
        # Create a mock client instance
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client_class.return_value = mock_client

        # Configure the check_connection method to return failure
        mock_client.check_connection.return_value = {
            "success": False,
            "message": "Authentication failed. Check your API key.",
            "status_code": 401,
            "error": "unauthorized",
        }

        # Run the main function with the check command
        with patch.dict(
            "os.environ", {"STT_ENDPOINT": stt_env["STT_ENDPOINT"], "STT_API_KEY": "invalid-key"}
        ):
            result = main(["check"])

        # Assert the result
        assert result == 1

        # Capture the output
        captured = capsys.readouterr()
        assert "Authentication failed. Check your API key." in captured.out

        # Assert the mock was called
        mock_client.check_connection.assert_called_once()


def test_check_command_json_output(capsys, stt_env):
    """Test the check command with JSON output format."""
    # Mock the STTClient.check_connection method to return success
    with patch("ovox.api.STTClient") as mock_client_class:
        # Create a mock client instance
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client_class.return_value = mock_client

        # Configure the check_connection method to return success
        mock_client.check_connection.return_value = {
            "success": True,
            "message": "Successfully connected to STT API",
            "status_code": 200,
        }

        # Run the main function with the check command and --json flag
        with patch.dict(
            "os.environ",
            {"STT_ENDPOINT": stt_env["STT_ENDPOINT"], "STT_API_KEY": stt_env["STT_API_KEY"]},
        ):
            result = main(["check", "--json"])

        # Assert the result
        assert result == 0

        # Capture the output
        captured = capsys.readouterr()
        assert '"success": true' in captured.out
        assert '"message": "Successfully connected to STT API"' in captured.out
        assert '"status_code": 200' in captured.out

        # Assert the mock was called
        mock_client.check_connection.assert_called_once()
