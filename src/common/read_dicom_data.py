import pickle
import hashlib
from pathlib import Path
import pydicom


def _get_dicom_pixel(file_path: Path):
    """Load a single file of dicom pixel data."""
    print(f"Loading file {file_path}")
    dicom_file = pydicom.read_file(file_path)
    return dicom_file.pixel_array


def read_dicom_data(path: Path, force_reload: bool = False):
    """Read the dicom content of a file or directory.
    This also creates a pickled cache for the directory for faster reloading.

    Args:
        path (Path): The file or directory to load.
        force_reload (bool, optional): If true, loads the given directory regardless if a cache exists. Defaults to False.

    Returns:
        A list of dicom image sequences.
    """
    # see if we already have a hash for faster loading
    # we need to hash the path string, as that is consistent.
    # just using the path object would result in a new object each time we execute this method
    sha256_file = hashlib.sha256()
    sha256_file.update(str(path).encode("utf-8"))
    hashed_path = Path(sha256_file.hexdigest() + ".bin")
    if not force_reload:
        # a pickled object is available, and we do not want to force reload
        if hashed_path.is_file():
            return pickle.load(open(hashed_path, "rb"))

    # Silently accept a single file, too
    if path.is_file():
        return [_get_dicom_pixel(path)]

    if not path.is_dir():
        raise NotADirectoryError("Path is not a file or directory!")

    result = []

    # get all sub entries in a dir.
    # this can be dirs, files or other things
    # Note: This list is NOT sorted!
    files = list(path.iterdir())
    # select only files, and sort it alphabetically
    files = sorted([file for file in files if file.is_file()])

    result = [_get_dicom_pixel(file) for file in files]

    # Create a hash in case we want to load the dataset later again
    pickle.dump(result, open(hashed_path, "wb"))

    return result
