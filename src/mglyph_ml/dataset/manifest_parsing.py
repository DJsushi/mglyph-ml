import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from pyparsing import cached_property
from typing import TypeVar, Generic


class ManifestSample(BaseModel):
    x: Decimal
    id: int


# TODO: this shouldn't be here, but rather a user-defined class inside the experiment
class ManifestSampleShape(Enum):
    SQUARE = "s"
    TRIANGLE = "t"
    CIRCLE = "c"


class ShapedManifestSample(ManifestSample):
    shape: ManifestSampleShape


T = TypeVar("T", bound=ManifestSample)


class Manifest(BaseModel, Generic[T]):
    model_config = ConfigDict(strict=True)

    name: str
    creation_time: datetime.datetime
    train_samples: list[T]
    test_samples: list[T]

    def get_sample_filename(self, id: Decimal) -> str:
        return f"{id}.png"
