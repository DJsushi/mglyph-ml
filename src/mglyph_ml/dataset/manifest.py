import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict


class ManifestSample(BaseModel):
    x: float
    filename: str
    metadata: dict


T = TypeVar("T", bound=ManifestSample)


class DatasetManifest(BaseModel, Generic[T]):
    model_config = ConfigDict(strict=True)

    name: str
    creation_time: datetime.datetime
    samples: dict[str, list[T]]

    @staticmethod
    def get_sample_filename(id: float) -> str:
        return f"{id}.png"
