from src.indexer.metadata import extract_dart_metadata, chunk_dart_file

SAMPLE_DART = """
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../data/repositories/auth_repository_impl.dart';

class AuthNotifier extends AsyncNotifier<AuthState> {
  @override
  Future<AuthState> build() async {
    return AuthState.initial();
  }

  Future<void> login(String phone) async {
    state = const AsyncValue.loading();
    state = await AsyncValue.guard(() async {
      final repo = ref.read(authRepositoryProvider);
      return repo.sendOtp(phone);
    });
  }
}

enum AuthState { initial, loading, authenticated, unauthenticated }
""".strip()


def test_extract_imports():
    meta = extract_dart_metadata(SAMPLE_DART, "lib/features/auth/presentation/providers/auth_notifier.dart")
    assert "package:flutter/material.dart" in meta["imports"]
    assert "package:flutter_riverpod/flutter_riverpod.dart" in meta["imports"]
    assert "../data/repositories/auth_repository_impl.dart" in meta["imports"]


def test_extract_exports():
    meta = extract_dart_metadata(SAMPLE_DART, "lib/features/auth/presentation/providers/auth_notifier.dart")
    assert "AuthNotifier" in meta["exports"]
    assert "AuthState" in meta["exports"]


def test_extract_layer():
    meta = extract_dart_metadata(SAMPLE_DART, "lib/features/auth/presentation/providers/auth_notifier.dart")
    assert meta["layer"] == "presentation"


def test_extract_feature():
    meta = extract_dart_metadata(SAMPLE_DART, "lib/features/auth/presentation/providers/auth_notifier.dart")
    assert meta["feature"] == "auth"


def test_extract_file_type():
    meta = extract_dart_metadata(SAMPLE_DART, "lib/features/auth/presentation/providers/auth_notifier.dart")
    assert meta["file_type"] == "notifier"


def test_extract_file_type_screen():
    meta = extract_dart_metadata("class LoginScreen extends StatelessWidget {}", "lib/features/auth/presentation/screens/login_screen.dart")
    assert meta["file_type"] == "screen"


def test_extract_file_type_model():
    meta = extract_dart_metadata("class User {}", "lib/features/auth/data/models/user_model.dart")
    assert meta["file_type"] == "model"


def test_chunk_dart_file():
    chunks = chunk_dart_file(SAMPLE_DART, "lib/features/auth/presentation/providers/auth_notifier.dart")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["feature"] == "auth"
