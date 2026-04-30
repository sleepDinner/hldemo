import warnings


def suppress_pil_exif_warnings():
    """Hide noisy PIL warnings from broken EXIF metadata while keeping real read errors visible."""
    warnings.filterwarnings(
        "ignore",
        message=r"Corrupt EXIF data.*",
        category=UserWarning,
        module=r"PIL\.TiffImagePlugin",
    )
