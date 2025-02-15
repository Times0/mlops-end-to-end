from pathlib import Path
from picsellia import Client, Dataset, DatasetVersion
from picsellia.types.enums import AnnotationFileType
import zipfile
import shutil
from rich.console import Console
from rich.progress import track
import yaml
import random
from config import config
from typing import Optional


# Configuration
DATASET_PATH = Path(r"data/THE-dataset")
console = Console()

client = Client(
    api_token=config.api_token,
    organization_name=config.ORG_NAME,
    host=config.HOST,
)


def ensure_dataset_downloaded(dataset_version: DatasetVersion, dataset_path: Path):
    """Download dataset if not already present"""
    if not dataset_path.exists():
        console.log(f"[yellow]Dataset not found at {dataset_path}, downloading...[/]")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_version.download(dataset_path)
        console.log("[green]Dataset downloaded successfully![/]")
    else:
        console.log("[blue]Dataset already exists, skipping download[/]")


def ensure_annotations_present(dataset_path: Path) -> bool:
    """Check if annotations are present and match the number of images"""

    # check if there is at least one text file in the dataset
    annotation_files = list(dataset_path.glob("*.txt"))
    images_files = list(dataset_path.glob("*.jpg"))

    if not images_files or annotation_files:
        return True
    console.log("[red]It seems like there are no annotations in the dataset![/]")
    return False


def ensure_annotations_downloaded(
    dataset_path: Path, dataset_version: DatasetVersion
) -> Optional[Path]:
    """Download annotations if not already present"""
    annotation_file = dataset_path / "annotations.zip"
    if not ensure_annotations_present(dataset_path):
        console.log("[yellow]Annotations missing or incomplete, downloading...[/]")
        annotation_file = dataset_version.export_annotation_file(
            annotation_file_type=AnnotationFileType.YOLO,
            target_path=str(dataset_path),
        )
        return annotation_file
    return None


def extract_annotations(annotation_file: str, dataset_path: Path):
    """Extract and organize annotation files"""
    # Delete existing txt files before extraction
    for txt_file in dataset_path.glob("*.txt"):
        txt_file.unlink()

    temp_dir = dataset_path / "temp"
    console.log("[yellow]Extracting annotation files...[/]")

    with zipfile.ZipFile(annotation_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    for txt_file in track(
        list(temp_dir.rglob("*.txt")), description="Moving annotation files"
    ):
        shutil.move(str(txt_file), str(dataset_path / txt_file.name))

    shutil.rmtree(temp_dir)

    shutil.rmtree(Path(annotation_file).parent.parent)
    console.log("[green]Annotation files extracted and organized![/]")


def split_dataset(
    dataset_path: Path, train_ratio: float, valid_ratio: float, seed: int = 42
):
    """Split dataset into train and valid sets and creates folders in dataset_path"""
    # Create images and labels folders for each split
    for split in ["train", "valid", "test"]:
        (dataset_path / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_path / split / "labels").mkdir(parents=True, exist_ok=True)
    console.log("[yellow]Splitting dataset...[/]")
    train_path = dataset_path / "train"
    valid_path = dataset_path / "valid"
    test_path = dataset_path / "test"

    # Create train and valid and test folders
    train_path.mkdir(parents=True, exist_ok=True)
    valid_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    formats = ("*.jpg", "*.jpeg", "*.png")
    image_files = [f for pattern in formats for f in dataset_path.glob(pattern)]

    random.seed(seed)
    random.shuffle(image_files)

    # Calculate split indices
    n_images = len(image_files)
    if n_images == 0:
        console.log("[red]No images found in the dataset no spliting...[/]")
        return None
    n_train = int(n_images * train_ratio)
    n_valid = int(n_images * valid_ratio)

    console.log(f"Total images: {n_images}")

    # Split into train and valid sets
    train_files = image_files[:n_train]
    valid_files = image_files[n_train : n_train + n_valid]
    test_files = image_files[n_train + n_valid :]

    def move_files_to_split(
        files, source_path: Path, target_path: Path, split_name: str
    ):
        """Move image files and their corresponding labels to the target split directory"""
        for img_file in track(files, description=f"Moving {split_name} files"):
            img_file = Path(img_file)
            label_file = source_path / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.move(
                    str(label_file), str(target_path / "labels" / label_file.name)
                )
            else:
                pass
                # console.log(f"[red]Label file not found for {img_file}[/]")
            shutil.move(str(img_file), str(target_path / "images" / img_file.name))

    # Move files to their respective splits
    move_files_to_split(train_files, dataset_path, train_path, "train")
    move_files_to_split(valid_files, dataset_path, valid_path, "valid")
    move_files_to_split(test_files, dataset_path, test_path, "test")

    console.log("[green]Dataset split complete![/]")


def create_yaml_yolo(dataset_path: Path, classes: list):
    """Create yaml file for yolo training"""
    console.log("[yellow]Creating yaml file for yolo training...[/]")
    yaml_path = dataset_path / "yolo.yaml"

    # Create dict mapping index to class name
    names_dict = {i: name.name for i, name in enumerate(classes)}

    # Create yaml config dict
    yaml_config = {
        "test": "./test/images",
        "train": "./train/images",
        "val": "./valid/images",
        "nc": len(classes),
        "names": names_dict,
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_config, f, sort_keys=False)

    console.log("[green]YAML file created![/]")


def main():
    console.log("[bold blue]Starting dataset preparation...[/]")
    print(client.list_datasets())
    dataset: Dataset = client.get_dataset_by_id(config.DATASET_ID)
    dataset_version: DatasetVersion = dataset.get_version("initial")

    # Downloading
    ensure_dataset_downloaded(dataset_version, DATASET_PATH)
    annotation_file = ensure_annotations_downloaded(DATASET_PATH, dataset_version)
    if annotation_file:
        extract_annotations(annotation_file, DATASET_PATH)

    # Preprocessing
    split_dataset(DATASET_PATH, 0.6, 0.2)
    classes = dataset_version.list_labels()
    create_yaml_yolo(DATASET_PATH, classes)


if __name__ == "__main__":
    main()
