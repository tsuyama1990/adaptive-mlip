import os

from pyacemaker.domain_models.constants import UV_PROJECT_ENVIRONMENT_DEFAULT

os.environ["UV_PROJECT_ENVIRONMENT"] = os.environ.get(
    "UV_PROJECT_ENVIRONMENT", UV_PROJECT_ENVIRONMENT_DEFAULT
)
