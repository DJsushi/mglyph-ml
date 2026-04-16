"""Color conversion utilities for dataset generation."""


def hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
	"""Convert HSL values to a hex color string.

	Args:
		hue: Hue in degrees (any real number, wrapped into [0, 360)).
		saturation: Saturation in percent (expected 0-100).
		lightness: Lightness in percent (expected 0-100).

	Returns:
		A hex color string in the form "#RRGGBB".
	"""

	h = float(hue) % 360.0
	s = max(0.0, min(100.0, float(saturation))) / 100.0
	l = max(0.0, min(100.0, float(lightness))) / 100.0

	c = (1.0 - abs(2.0 * l - 1.0)) * s
	hp = h / 60.0
	x = c * (1.0 - abs(hp % 2.0 - 1.0))

	if 0.0 <= hp < 1.0:
		r1, g1, b1 = c, x, 0.0
	elif 1.0 <= hp < 2.0:
		r1, g1, b1 = x, c, 0.0
	elif 2.0 <= hp < 3.0:
		r1, g1, b1 = 0.0, c, x
	elif 3.0 <= hp < 4.0:
		r1, g1, b1 = 0.0, x, c
	elif 4.0 <= hp < 5.0:
		r1, g1, b1 = x, 0.0, c
	else:
		r1, g1, b1 = c, 0.0, x

	m = l - c / 2.0
	r = int(round((r1 + m) * 255.0))
	g = int(round((g1 + m) * 255.0))
	b = int(round((b1 + m) * 255.0))

	r = max(0, min(255, r))
	g = max(0, min(255, g))
	b = max(0, min(255, b))

	return f"#{r:02X}{g:02X}{b:02X}"
