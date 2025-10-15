import cbor2

SENML_KEYMAP = {
    0: "n",   # Name
    1: "u",   # Unit
    2: "v",   # Value
    3: "vs",  # String value
    4: "vb",  # Boolean value
    5: "vd",  # Data value
    6: "s",   # Sum
    7: "t",   # Time
    8: "ut",  # Update time
    -1: "bu", # Base unit
    -2: "bn", # Base name
    -3: "bt", # Base time
    -4: "bv", # Base value
    -5: "bs"  # Base sum
}

def _senml_object_hook(decoder, value):
    if isinstance(value, dict):
        new_value = {}
        for k, v in value.items():
            if isinstance(k, int) and k in SENML_KEYMAP:
                new_value[SENML_KEYMAP[k]] = v
            else:
                new_value[k] = v
        return new_value
    return value

def decode_senml_cbor(payload: bytes):
    return cbor2.loads(payload, object_hook=_senml_object_hook)

