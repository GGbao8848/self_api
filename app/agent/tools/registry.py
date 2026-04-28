from app.agent.types import ToolSpec


_TOOL_SPECS = [
    ToolSpec(
        name="xml-to-yolo",
        description="Convert Pascal VOC XML annotations to YOLO labels.",
        async_task=True,
    ),
    ToolSpec(
        name="yolo-sliding-window-crop",
        description="Crop large YOLO images with synchronized sliding-window labels.",
        async_task=True,
    ),
    ToolSpec(
        name="yolo-augment",
        description="Apply YOLO dataset augmentation.",
        async_task=True,
    ),
    ToolSpec(
        name="split-yolo-dataset",
        description="Split a YOLO dataset into train, val, and test subsets.",
        async_task=True,
    ),
    ToolSpec(
        name="annotate-visualize",
        description="Render annotation preview images for a dataset.",
        async_task=True,
    ),
    ToolSpec(
        name="clean-nested-dataset-flat",
        description="Flatten nested image/XML datasets into images and xmls folders.",
        async_task=True,
    ),
    ToolSpec(
        name="publish-incremental-yolo-dataset",
        description="Publish incremental YOLO data to a remote dataset location.",
        async_task=True,
    ),
    ToolSpec(
        name="scan-yolo-label-indices",
        description="Scan YOLO label index usage and counts.",
    ),
    ToolSpec(
        name="rewrite-yolo-label-indices",
        description="Rewrite or remap YOLO label indices.",
    ),
]


def get_tool_specs() -> list[ToolSpec]:
    return list(_TOOL_SPECS)
