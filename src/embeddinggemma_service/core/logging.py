import logging

_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_FORMAT)