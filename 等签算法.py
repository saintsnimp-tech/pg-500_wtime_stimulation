"""兼容入口：实际实现位于 visa_wait 包。"""

from visa_wait.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
