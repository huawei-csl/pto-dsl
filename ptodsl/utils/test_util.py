import os


DEVICE_ENV_VAR = "PTODSL_TEST_DEVICE_ID"
DEFAULT_DEVICE_ID = "0"
DEVICE_PREFIX = "npu:"


def get_test_device() -> str:
    device_id = os.getenv(DEVICE_ENV_VAR)
    if not device_id:
        print(
            f"Warning: {DEVICE_ENV_VAR} is not set; defaulting to {DEFAULT_DEVICE_ID}."
        )
        device_id = DEFAULT_DEVICE_ID

    if device_id.startswith(DEVICE_PREFIX):
        return device_id
    return f"{DEVICE_PREFIX}{device_id}"
