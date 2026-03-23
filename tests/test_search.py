from src.retriever.search import parse_file_references, build_metadata_filter


def test_parse_file_references_from_stacktrace():
    trace = """
    #0      AuthNotifier.login (package:cravv/features/auth/presentation/providers/auth_notifier.dart:44:5)
    #1      OtpScreen._onVerify (package:cravv/features/auth/presentation/screens/otp_screen.dart:87:12)
    """
    refs = parse_file_references(trace)
    assert len(refs) == 2
    assert refs[0]["file"] == "lib/features/auth/presentation/providers/auth_notifier.dart"
    assert refs[0]["line"] == 44
    assert refs[1]["file"] == "lib/features/auth/presentation/screens/otp_screen.dart"


def test_parse_file_references_from_error():
    error = "Error at lib/core/network/dio_client.dart:18"
    refs = parse_file_references(error)
    assert len(refs) == 1
    assert refs[0]["file"] == "lib/core/network/dio_client.dart"


def test_build_metadata_filter_feature():
    sql_str, params = build_metadata_filter(feature="auth")
    assert "feature" in sql_str
    assert params == ["auth"]


def test_build_metadata_filter_empty():
    sql, params = build_metadata_filter()
    assert sql == ""
    assert params == []
