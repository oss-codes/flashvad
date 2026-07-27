from pathlib import Path

from flashvad.manifest import ManifestItem
from flashvad.region import RegionProfile, balanced_item_weights


def item(language: str, domain: str = "pstn") -> ManifestItem:
    return ManifestItem(Path("unused.wav"), 16_000, language, domain, ())


def test_region_weights_balance_corpus_volume_and_keep_priorities() -> None:
    profile = RegionProfile(
        name="test",
        language_weights={"hi": 1.0, "ml": 2.0},
        domain_weights={"pstn": 1.0, "studio": 0.5},
    )
    items = [item("hi"), item("hi"), item("hi"), item("ml"), item("ml", "studio")]
    weights = balanced_item_weights(items, profile)

    assert weights[:3].tolist() == [1 / 3, 1 / 3, 1 / 3]
    assert weights[3].item() == 1.0
    assert weights[4].item() == 0.5


def test_region_profile_falls_back_from_locale_to_base_language() -> None:
    profile = RegionProfile("test", {"ar": 0.9}, {}, 0.1, 0.2)
    assert profile.language_weight("ar-AE") == 0.9
    assert profile.language_weight("unknown") == 0.1
