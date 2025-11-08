import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from pyparsing import cached_property


class ManifestImage(BaseModel):
    filename: str
    x: float


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
        # If already parsed into ManifestImage objects, keep as is
        if all(isinstance(i, ManifestImage) for i in value):
            return value
        # If it's a list of [filename, x]
        return [ManifestImage(filename=fn, x=x) for fn, x in value]

    def get_glyph_filename(self, x: Decimal) -> str:
        return self.__images_dict[x]
