from cbor2 import CBORDecoder, CBORDecodeError
import io

def decode_stream(buffer: bytearray):
    decoded_objects = []
    stream = io.BytesIO(buffer)
    decoder = CBORDecoder(stream)

    try:
        while True:
            obj = decoder.decode()
            if isinstance(obj, dict):
                decoded_objects.append(obj)
    #        else:
     #           print(f"[decode_stream] Ignored non-dict object: {obj}")
    except EOFError:
        # Incomplete frame: keep leftovers for next round
        pass
    except CBORDecodeError as e:
        consumed = stream.tell()
        print(f"[decode_stream] CBOR decode error, dropping {consumed} bytes: {e}")
        return decoded_objects, bytearray()  # flush buffer on error

    # Figure out how much of buffer was consumed
    consumed = stream.tell()
    remaining = buffer[consumed:]
    return decoded_objects, bytearray(remaining)
