import os
import logging
import pytest
from click.testing import CliRunner
from ultralytics.utils import LOGGER
from yaltai.cli.yaltai import yaltai_cli
from kraken.lib.xml import XMLPage


@pytest.fixture(scope='function')
def custom_logger():
    class CustomLoggingHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            # A list to store log records (level and message)
            self.records = []

        def emit(self, record):
            # Append a tuple of the log level and the message to the records list
            self.records.append((record.levelname, record.getMessage()))

        def clear(self):
            self.records = []

    # Create a logger
    logger = LOGGER
    logger.setLevel(logging.DEBUG)  # Set the logger to handle all log levels

    # Create and add the custom handler to the logger
    custom_handler = CustomLoggingHandler()
    logger.addHandler(custom_handler)

    # Clear the records list before each test
    custom_handler.records.clear()

    # Provide both the logger and handler for access during tests
    yield custom_handler

    custom_handler.records.clear()

    # Optional: Remove the handler after the test to prevent interference
    logger.removeHandler(custom_handler)


def test_yaltai_single_alto_to_xml(custom_logger):
    """Ensures that we can convert to YOLO format"""
    runner = CliRunner()

    # Trigger a warning.
    result = runner.invoke(
        yaltai_cli,
        [
            "kraken",
            "--alto",
            "-i",
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_files", "page1.jpg"),
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_files", "page1.xml"),
            "segment",
            "-y",
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "nano-yolo-ladas.pt")
        ]
    )
    assert result.exit_code == 0
    assert "page1.jpg: 640x352 2 GraphicZones, 1 MainZone-P-Continued, 1 MainZone-Sp, 2 QuireMarksZones" in "\n".join([
        record[1]
        for record in custom_logger.records
    ])

    page = XMLPage(os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_files", "page1.xml"))
    assert {
               region_type: [region.boundary for region in regions]
               for region_type, regions in page.regions.items()
           } == {
               'GraphicZone': [[(614.0, 12.0),
                                (2614.0, 12.0),
                                (2614.0, 819.0),
                                (614.0, 819.0),
                                (614.0, 12.0)],
                               [(720.0, 0.0),
                                (2634.0, 0.0),
                                (2634.0, 343.0),
                                (720.0, 343.0),
                                (720.0, 0.0)]],
               'MainZone-P-Continued': [[(122.0, 1536.0),
                                         (2218.0, 1536.0),
                                         (2218.0, 2888.0),
                                         (122.0, 2888.0),
                                         (122.0, 1536.0)]],
               'MainZone-Sp': [[(95.0, 2974.0),
                                (2201.0, 2974.0),
                                (2201.0, 4854.0),
                                (95.0, 4854.0),
                                (95.0, 2974.0)]],
               'QuireMarksZone': [[(1617.0, 4877.0),
                                   (1825.0, 4877.0),
                                   (1825.0, 4980.0),
                                   (1617.0, 4980.0),
                                   (1617.0, 4877.0)],
                                  [(1531.0, 4841.0),
                                   (1814.0, 4841.0),
                                   (1814.0, 4983.0),
                                   (1531.0, 4983.0),
                                   (1531.0, 4841.0)]]
           }

    assert len([line.baseline for line in page.lines.values()])
    # ToDo: Add a test to check for line being part of regions


def test_alto_to_yolo_with_lines(custom_logger):
    """Test line region detection in ALTO to YOLO conversion"""
    runner = CliRunner()
    test_files_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_files")
    
    result = runner.invoke(
        yaltai_cli,
        [
            "convert", "alto-to-yolo",
            os.path.join(test_files_dir, "page1.xml"),
            "test_output",
            "--line-as-region", "default"
        ]
    )
    
    assert result.exit_code == 0
    assert os.path.exists("test_output/labels/page1.txt")
    
    # Verify line detection
    page = XMLPage(os.path.join(test_files_dir, "page1.xml"))
    assert any(line.tags.get("type") == "default" for line in page.lines.values())
