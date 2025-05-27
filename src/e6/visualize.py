from e6.ui import show_window
from common.read_dicom_data import read_dicom_data
from pathlib import Path

from e6._module_loader import get_filters


def _main():
    image_data = read_dicom_data(Path("data/series/"))

    show_window(get_filters(), image_data)


if __name__ == "__main__":
    _main()
