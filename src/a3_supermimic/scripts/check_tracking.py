"""Validate the actual launcher environment and authentication before CUDA startup."""
from a3_supermimic.training.tracking import (
    load_tracking_credentials, tracking_error_message, tracking_settings,
)


def main():
    try:
        import os
        import swanlab
        load_tracking_credentials("online")
        tracking_settings()
        if not swanlab.login(api_key=os.environ["SWANLAB_API_KEY"], save=False):
            raise RuntimeError("SwanLab authentication failed")
    except Exception as error:
        raise SystemExit(tracking_error_message(error)) from None
    print("SwanLab settings/authentication preflight OK", flush=True)


if __name__ == "__main__":
    main()
