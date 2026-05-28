import logging, warnings


def quiet_startup() -> None:
    warnings.filterwarnings("ignore", message=r".*predict_dataloader.*num_workers.*")
    warnings.filterwarnings(
        "ignore", message=r".*tensorboardX.*removed as a dependency.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated",
        category=FutureWarning,
        module=r"torch\.cuda",
    )

    # Some transitive deps still configure the pytorch_lightning logger even
    # though we no longer use it; silence it defensively.
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
