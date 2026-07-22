"""A real ``CompileOptionsProto`` encoder.

The famous 6 bytes ``1a 04 20 01 28 01``, rediscovered independently by nearly
every PJRT client, are not a magic constant. They are
``build_options { num_replicas: 1, num_partitions: 1 }``. Treating them as a
literal hard-codes single-device execution -- which is how a sharding story ends
up never running.

XLA's C++ defaults for these are 0 and ``ParseDeviceAssignmentCompileOptions``
CHECK-fails on ``replica_count > 0``, so *something* must be encoded; it just
should not be a constant.

Field numbers come from the generated ABI module, parsed from the vendored
``compile_options.proto`` -- never hand-typed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _varint(n: int) -> bytes:
    if n < 0:  # sign-extend to 64 bits, per protobuf wire format
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _tag(field_no: int, wire: int) -> bytes:
    return _varint((field_no << 3) | wire)


def _int_field(field_no: int, v: int) -> bytes:
    return _tag(field_no, 0) + _varint(v)


def _bool_field(field_no: int, v: bool) -> bytes:
    return _tag(field_no, 0) + _varint(1 if v else 0)


def _bytes_field(field_no: int, v: bytes) -> bytes:
    return _tag(field_no, 2) + _varint(len(v)) + v


def _packed_bools(field_no: int, vs: list[bool]) -> bytes:
    return _bytes_field(field_no, bytes(1 if v else 0 for v in vs))


@dataclass
class CompileOptions:
    """The subset of ``CompileOptionsProto`` a PJRT client actually needs.

    ``num_replicas``/``num_partitions`` default to 1 so the encoding of a
    default instance is byte-identical to the well-known 6 bytes -- which the
    test suite pins.
    """

    num_replicas: int = 1
    num_partitions: int = 1
    use_spmd_partitioning: bool = False
    use_shardy_partitioner: bool = False
    alias_passthrough_params: bool = False
    allow_spmd_sharding_propagation_to_output: list[bool] = field(default_factory=list)
    device_assignment: bytes | None = None  # serialized xla.DeviceAssignmentProto
    parameter_is_tupled_arguments: bool = False
    compile_portable_executable: bool = False

    def build_options_bytes(self, F: dict[str, int]) -> bytes:
        out = bytearray()
        # Emitted in ascending field order: not required by protobuf, but it
        # makes the encoding canonical, so a compile cache keyed on these bytes
        # is stable.
        out += _int_field(F["num_replicas"], self.num_replicas)
        out += _int_field(F["num_partitions"], self.num_partitions)
        if self.use_spmd_partitioning:
            out += _bool_field(F["use_spmd_partitioning"], True)
        if self.device_assignment is not None:
            out += _bytes_field(F["device_assignment"], self.device_assignment)
        if self.alias_passthrough_params:
            out += _bool_field(F["alias_passthrough_params"], True)
        if self.allow_spmd_sharding_propagation_to_output:
            out += _packed_bools(F["allow_spmd_sharding_propagation_to_output"],
                                 self.allow_spmd_sharding_propagation_to_output)
        if self.use_shardy_partitioner:
            out += _bool_field(F["use_shardy_partitioner"], True)
        return bytes(out)

    def encode(self, abi) -> bytes:
        """Serialize to a ``CompileOptionsProto``."""
        co = abi.PROTO_FIELDS["CompileOptionsProto"]
        bo = abi.PROTO_FIELDS["ExecutableBuildOptionsProto"]
        out = bytearray()
        if self.parameter_is_tupled_arguments:
            out += _bool_field(co["parameter_is_tupled_arguments"], True)
        out += _bytes_field(co["executable_build_options"], self.build_options_bytes(bo))
        if self.compile_portable_executable:
            out += _bool_field(co["compile_portable_executable"], True)
        return bytes(out)


#: What every prior port hardcoded. Kept only so the test suite can prove the
#: encoder reproduces it.
WELL_KNOWN_DEFAULT = bytes([0x1A, 0x04, 0x20, 0x01, 0x28, 0x01])
