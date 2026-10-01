"""Unit tests: Algorithm 3 reproduces the paper's preprocessing examples (rows 1-3)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rv2.textnorm import alg3, canonical_key, preprocess  # noqa: E402

CASES = [
    ("يا زووول وين كنت امبارح فقدتك خالص 🥺😢❤️🙏 https://t.me/group1 @ahmed_sd #اخباري_اليوم 🤩",
     "يا زول وين كنت امبارح فقدتك خالص اخباري اليوم"),
    ("الهلال انتصر في المباااراة والحمدلله finally ⚽🎉💚 we won the league!! #الهلال_بطل #Hilal_wins @SudaneseSports",
     "الهلال انتصر في المباراه والحمدلله الهلال بطل"),
    ("أَنتَ صَدِيقِي مِنَ زَمَانٍ وَدَائِماً بَكُونْ مَعَاكْ habibi ya 7abib 💙🥰👍🙏 @user99 https://bit.ly/xyz",
     "انت صديقي من زمان ودايما بكون معاك"),
]


def test_paper_examples():
    for raw, want in CASES:
        got = alg3(raw)
        assert got == want, (got, want)


def test_key_coarser_than_inputs():
    a, b = "ـالسلامُ عليكم", "السلام   عليكم!!"
    assert canonical_key(a) == canonical_key(b)
    assert alg3(alg3(a)) == alg3(a)          # idempotent


def test_modes_exist():
    for m in ("none", "minimal", "alg3", "mhamed"):
        assert isinstance(preprocess("مرحبا يا زول", m), str)


if __name__ == "__main__":
    test_paper_examples(); test_key_coarser_than_inputs(); test_modes_exist()
    print("test_textnorm: OK")
