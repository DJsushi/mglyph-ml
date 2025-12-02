import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from pyparsing import cached_property


class ManifestImage(BaseModel):
    filename: str
    x: Decimal


class Manifest(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    short_name: str
    author_public: bool
    creation_time: datetime.datetime
    images: list[ManifestImage]

    @cached_property
    def __images_dict(self) -> dict[Decimal, str]:
        return {Decimal(img.x): img.filename for img in self.images}

    @field_validator("images", mode="before")
    @classmethod
    def parse_image_list(cls, value: Any):
        # If already parsed into ManifestImage objects, return as list of dicts
        if all(isinstance(i, ManifestImage) for i in value):
            return [{"filename": img.filename, "x": img.x} for img in value]
        # If it's a list of dicts (already in correct format), return as is
        if all(isinstance(i, dict) for i in value):
            return value
        # If it's a list of [filename, x] tuples
        return [{"filename": filename, "x": Decimal(x)} for filename, x in value]

    def get_glyph_filename(self, x: Decimal) -> str:
        return self.__images_dict[x]
