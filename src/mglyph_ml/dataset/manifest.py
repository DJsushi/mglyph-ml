import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict


class ManifestSample(BaseModel):
    x: float
    filename: str
    metadata: dict


class DatasetManifest(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    creation_time: datetime.datetime
    samples: dict[str, list[ManifestSample]]

    @staticmethod
    def get_sample_filename(id: float) -> str:
        return f"{id}.png"
