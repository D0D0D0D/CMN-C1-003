"""Unit tests — ConversationRepository（分離・保存上限）."""

from src.services.conversation_repository import ConversationRepository


def _msgs(n):
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def test_unknown_session_returns_empty():
    repo = ConversationRepository()
    assert repo.load("nope") == []


def test_session_isolation():
    repo = ConversationRepository()
    repo.save("a", [{"role": "user", "content": "A"}])
    repo.save("b", [{"role": "user", "content": "B"}])
    assert repo.load("a") == [{"role": "user", "content": "A"}]
    assert repo.load("b") == [{"role": "user", "content": "B"}]


def test_rolling_window_cap_keeps_most_recent():
    repo = ConversationRepository(max_messages=3)
    repo.save("s", _msgs(10))
    saved = repo.load("s")
    assert len(saved) == 3
    assert [m["content"] for m in saved] == ["m7", "m8", "m9"]


def test_unlimited_when_none():
    repo = ConversationRepository(max_messages=None)
    repo.save("s", _msgs(100))
    assert len(repo.load("s")) == 100


def test_load_returns_copy():
    repo = ConversationRepository()
    repo.save("s", [{"role": "user", "content": "x"}])
    got = repo.load("s")
    got.append({"role": "user", "content": "mutated"})
    assert len(repo.load("s")) == 1  # 内部状態は変わらない
