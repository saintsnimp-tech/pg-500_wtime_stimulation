from __future__ import annotations

import unittest

from visa_wait.social import extract_timeline, normalize_social_text


class SocialParserTests(unittest.TestCase):
    def test_chinese_timeline_dates(self) -> None:
        lodged, decision, confidence = extract_timeline("2026年3月2日递签，2026年4月15日下签，澳洲500博士")
        self.assertEqual(lodged.isoformat(), "2026-03-02")
        self.assertEqual(decision.isoformat(), "2026-04-15")
        self.assertEqual(confidence, "high")

    def test_normalization_drops_text_and_identity(self) -> None:
        record = normalize_social_text(
            platform="Xiaohongshu",
            source_url="https://example.invalid/post/123",
            text="PhD 国内递交，2026-03-02 lodged，2026-04-15 granted。username=someone",
        )
        self.assertIsNotNone(record)
        self.assertEqual(record.waiting_days, 44)
        self.assertNotIn("someone", repr(record))


if __name__ == "__main__":
    unittest.main()

