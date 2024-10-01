import tempfile
import os
from click.testing import CliRunner

from yaltai.cli.yaltai import yaltai_cli


def test_yaltai_single_alto_to_xml():
    """Ensures that we can convert to YOLO format"""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tempdir:
        # Run the Click command
        result = runner.invoke(
            yaltai_cli,
            [
                "convert",
                "alto-to-yolo",
                os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_files", "alto_dataset", "output.xml"),
                tempdir,
            ])

        # Ensure the command ran successfully
        assert result.exit_code == 0
        assert "Found 1 to convert." in result.output
        assert "- 00001 NumberingZone" in result.output, "Correct number of zone types are found"
        assert "- 00001 RunningTitleZone" in result.output, "Correct number of zone types are found"
        assert "- 00001 MainZone-P-Continued" in result.output, "Correct number of zone types are found"
        assert "- 00004 MainZone-P" in result.output, "Correct number of zone types are found"
        with open(os.path.join(tempdir, "labels", "output.txt")) as f:
            data = [line.split() for line in f.read().split("\n")]
        assert data == [
            ['0', '0.345606', '0.117500', '0.037501', '0.020206'],
            ['1', '0.612827', '0.118333', '0.212882', '0.020736'],
            ['2', '0.616390', '0.210833', '0.578172', '0.142851'],
            ['3', '0.616390', '0.365833', '0.577029', '0.162800'],
            ['3', '0.616390', '0.547500', '0.574365', '0.199581'],
            ['3', '0.616390', '0.727500', '0.576050', '0.161861'],
            ['3', '0.611639', '0.819167', '0.583077', '0.022773']
        ]
