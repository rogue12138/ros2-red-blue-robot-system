#!/usr/bin/env python3

import json
import random
import shutil
from collections import Counter
from pathlib import Path


SOURCE = Path(
    "/home/ros/robot_ws/datasets/"
    "color_cubes_training/synthetic_200_v1"
)

TARGET = Path(
    "/home/ros/robot_ws/datasets/"
    "color_cubes_training/yolo_200_v1"
)

RANDOM_SEED = 20260815

SOURCE_IMAGES = SOURCE / "images" / "all"
SOURCE_LABELS = SOURCE / "labels" / "all"

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}


def validate_label(label_path):
    class_counts = Counter()

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        fields = line.split()

        if len(fields) != 5:
            raise ValueError(
                f"{label_path}:{line_number} "
                f"字段数量不是5"
            )

        class_id = int(fields[0])
        coordinates = [
            float(value)
            for value in fields[1:]
        ]

        if class_id not in {0, 1}:
            raise ValueError(
                f"{label_path}:{line_number} "
                f"非法类别：{class_id}"
            )

        if not all(
            0.0 <= value <= 1.0
            for value in coordinates
        ):
            raise ValueError(
                f"{label_path}:{line_number} "
                "归一化坐标超出0到1"
            )

        if coordinates[2] <= 0.0:
            raise ValueError("标注宽度必须大于0")

        if coordinates[3] <= 0.0:
            raise ValueError("标注高度必须大于0")

        class_counts[class_id] += 1

    return class_counts


def main():
    if TARGET.exists():
        raise FileExistsError(
            f"目标目录已经存在：{TARGET}"
        )

    images = sorted(
        path
        for path in SOURCE_IMAGES.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )

    if len(images) != 200:
        raise ValueError(
            f"图片数量应为200，实际为{len(images)}"
        )

    pairs = []

    for image_path in images:
        label_path = (
            SOURCE_LABELS
            / f"{image_path.stem}.txt"
        )

        if not label_path.is_file():
            raise FileNotFoundError(
                f"缺少标签：{label_path}"
            )

        validate_label(label_path)

        pairs.append(
            (image_path, label_path)
        )

    random_generator = random.Random(
        RANDOM_SEED
    )
    random_generator.shuffle(pairs)

    split_pairs = {
        "train": pairs[:160],
        "val": pairs[160:180],
        "test": pairs[180:200],
    }

    manifest = {
        "random_seed": RANDOM_SEED,
        "classes": {
            "0": "red_cube",
            "1": "blue_cube",
        },
        "splits": {},
    }

    for split_name, current_pairs in (
        split_pairs.items()
    ):
        image_directory = (
            TARGET / "images" / split_name
        )
        label_directory = (
            TARGET / "labels" / split_name
        )

        image_directory.mkdir(
            parents=True,
            exist_ok=False,
        )
        label_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        class_counts = Counter()
        file_names = []

        for image_path, label_path in (
            current_pairs
        ):
            shutil.copy2(
                image_path,
                image_directory / image_path.name,
            )
            shutil.copy2(
                label_path,
                label_directory / label_path.name,
            )

            class_counts.update(
                validate_label(label_path)
            )
            file_names.append(image_path.name)

        manifest["splits"][split_name] = {
            "image_count": len(current_pairs),
            "red_boxes": class_counts[0],
            "blue_boxes": class_counts[1],
            "files": file_names,
        }

        print(
            f"{split_name}: "
            f"images={len(current_pairs)}, "
            f"red_boxes={class_counts[0]}, "
            f"blue_boxes={class_counts[1]}"
        )

    dataset_yaml = (
        f"path: {TARGET}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: red_cube\n"
        "  1: blue_cube\n"
    )

    (TARGET / "dataset.yaml").write_text(
        dataset_yaml,
        encoding="utf-8",
    )

    (TARGET / "split_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"数据集目录：{TARGET}")
    print("数据集划分完成")


if __name__ == "__main__":
    main()
