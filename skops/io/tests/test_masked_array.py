import io
import json
from zipfile import ZipFile

import numpy as np
import pytest

from skops.io import dump, dumps, get_untrusted_types, load, loads


@pytest.mark.parametrize("use_file", [False, True])
@pytest.mark.parametrize("mask", [[False, True, False], np.ma.nomask])
@pytest.mark.parametrize(
    "data, fill_value",
    [([1, 2, 3], -99), ([1.5, 2.5, 3.5], -0.5), (["a", "b", "c"], "?")],
)
def test_masked_array_fill_value_roundtrip(data, fill_value, mask, use_file, tmp_path):
    original = np.ma.array(data, mask=mask, fill_value=fill_value)
    if use_file:
        path = tmp_path / "masked.skops"
        dump(original, path)
        restored = load(path)
    else:
        saved = dumps(original)
        assert get_untrusted_types(data=saved) == []
        restored = loads(saved)

    assert restored.fill_value == original.fill_value
    assert restored.dtype == original.dtype
    np.testing.assert_array_equal(restored.data, original.data)
    np.testing.assert_array_equal(restored.mask, original.mask)
    np.testing.assert_array_equal(restored.filled(), original.filled())


def test_masked_array_without_saved_fill_value():
    original = np.ma.array([1, 2, 3], mask=[False, True, False])
    saved = io.BytesIO()
    with ZipFile(io.BytesIO(dumps(original))) as source, ZipFile(saved, "w") as target:
        for name in source.namelist():
            content = source.read(name)
            if name == "schema.json":
                schema = json.loads(content)
                schema["content"].pop("fill_value", None)
                content = json.dumps(schema).encode()
            target.writestr(name, content)

    restored = loads(saved.getvalue())
    np.testing.assert_array_equal(restored.filled(), original.filled())
    assert restored.fill_value == original.fill_value
