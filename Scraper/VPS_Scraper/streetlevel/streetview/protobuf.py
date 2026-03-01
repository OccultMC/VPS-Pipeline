# URL-encoded protobuf used by Google Maps.
from enum import Enum
from decimal import Decimal


class ProtobufType(Enum):
    MESSAGE = "m"
    BOOL = "b"
    DOUBLE = "d"
    ENUM = "e"
    INT = "i"
    STRING = "s"


class ProtobufEnum:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"ProtobufEnum({str(self.value)})"

    def __str__(self):
        return f"ProtobufEnum({str(self.value)})"


def to_protobuf_url(fields):
    return _to_protobuf_url(fields)[1]


def _to_protobuf_url(fields):
    serialized = ""
    child_count = 0
    for field in fields.items():
        tag = field[0]
        value = field[1]
        sub_child_count, sub_serialized = _field_to_string(tag, value)
        serialized += sub_serialized
        child_count += sub_child_count
    return child_count, serialized


def _message_to_string(tag, value):
    sub_child_count, sub_serialized = _to_protobuf_url(value)
    serialized = f"!{tag}m{sub_child_count}" + sub_serialized
    return sub_child_count + 1, serialized


def _list_to_string(tag, value):
    serialized = ""
    child_count = 0
    for entry in value:
        sub_child_count, sub_serialized = _field_to_string(tag, entry)
        serialized += sub_serialized
        child_count += sub_child_count
    return child_count, serialized


def _field_to_string(tag, value):
    if isinstance(value, list):
        return _list_to_string(tag, value)
    else:
        datatype = _get_datatype_str(value)
        if datatype is ProtobufType.MESSAGE:
            return _message_to_string(tag, value)
        elif datatype is ProtobufType.BOOL:
            value = 1 if value else 0
        elif datatype is ProtobufType.ENUM:
            value = value.value
        return 1, f"!{tag}{datatype.value}{value}"


def _get_datatype_str(value):
    if isinstance(value, str):
        return ProtobufType.STRING
    elif isinstance(value, bool):
        return ProtobufType.BOOL
    elif isinstance(value, ProtobufEnum):
        return ProtobufType.ENUM
    elif isinstance(value, int):
        return ProtobufType.INT
    elif isinstance(value, (float, Decimal)):
        return ProtobufType.DOUBLE
    elif isinstance(value, dict):
        return ProtobufType.MESSAGE
    else:
        raise NotImplementedError(value)
