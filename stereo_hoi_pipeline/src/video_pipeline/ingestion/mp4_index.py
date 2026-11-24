"""Read the frame index of an MP4 file without reading its video data, and turn its HEVC samples
into Annex-B packets.

Used by the robocap and EGO-CAM live readers. Only the box headers and the ``moov`` box are read
(ranged reads on a seekable stream, e.g. a Cloud Storage object), so indexing a 300 MB file reads
about 100 kB. The video data is streamed later, sample by sample, from the offsets found here.

Facts this relies on (checked on the op-datoric files, see docs/ego_qa.md "Recording formats"):
one video track, ``moov`` after ``mdat``, no B-frames (no ``ctts`` needed, but it is honoured),
HEVC in ``hvc1`` boxes with the parameter sets in ``hvcC`` and 4-byte NAL lengths.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import IO

import numpy as np

from video_pipeline.core.errors import IngestionError

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta"}
_VPS, _SPS, _PPS = 32, 33, 34


@dataclass
class Mp4Track:
    codec: str                       # sample entry type, e.g. 'hvc1'
    width: int
    height: int
    timescale: int
    offsets: np.ndarray              # int64, byte offset of every sample in the file
    sizes: np.ndarray                # int64
    keys: np.ndarray                 # bool (sync samples)
    pts: np.ndarray                  # int64, presentation time in timescale units
    length_size: int = 4             # bytes of each NAL length prefix (hvcC)
    parameter_sets: bytes = b""      # VPS/SPS/PPS from hvcC as Annex-B
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def samples(self) -> int:
        return int(len(self.sizes))


def _box_header(data: bytes, off: int, end: int) -> tuple[bytes, int, int]:
    """(type, header length, total size) of the box at `off`."""
    if off + 8 > end:
        raise IngestionError(f"truncated MP4 box header at {off}")
    size, typ = struct.unpack_from(">I4s", data, off)
    header = 8
    if size == 1:
        size = struct.unpack_from(">Q", data, off + 8)[0]
        header = 16
    elif size == 0:
        size = end - off
    if size < header or off + size > end:
        raise IngestionError(f"bad MP4 box '{typ!r}' size {size} at {off}")
    return typ, header, size


def _children(data: bytes, start: int, end: int):
    off = start
    while off + 8 <= end:
        typ, header, size = _box_header(data, off, end)
        yield typ, off + header, off + size
        off += size


def _find(data: bytes, start: int, end: int, path: list[bytes]) -> tuple[int, int] | None:
    for typ, body, stop in _children(data, start, end):
        if typ == path[0]:
            return (body, stop) if len(path) == 1 else _find(data, body, stop, path[1:])
    return None


def _full_box(data: bytes, body: int) -> tuple[int, int]:
    """(version, offset after version+flags)."""
    return data[body], body + 4


def _parse_hvcc(data: bytes, body: int, end: int) -> tuple[int, bytes]:
    length_size = (data[body + 21] & 0x03) + 1
    arrays = data[body + 22]
    pos = body + 23
    out = bytearray()
    for _ in range(arrays):
        count = struct.unpack_from(">H", data, pos + 1)[0]
        pos += 3
        for _ in range(count):
            n = struct.unpack_from(">H", data, pos)[0]
            pos += 2
            if pos + n > end:
                raise IngestionError("truncated hvcC box")
            out += b"\x00\x00\x00\x01" + data[pos:pos + n]
            pos += n
    return length_size, bytes(out)


def _ilst_tags(data: bytes, start: int, end: int) -> dict[str, str]:
    """iTunes-style tags (udta/meta/ilst), e.g. '\xa9cmt' -> 'comment' as FFmpeg writes them."""
    names = {b"\xa9cmt": "comment", b"\xa9too": "encoder", b"\xa9nam": "title"}
    tags: dict[str, str] = {}
    meta = _find(data, start, end, [b"meta"])
    if meta is None:
        return tags
    ilst = _find(data, meta[0] + 4, meta[1], [b"ilst"])         # meta is a full box
    if ilst is None:
        return tags
    # 'mdta' style (FFmpeg -movflags use_metadata_tags): ilst entries are 1-based indexes into 'keys'
    key_names: dict[int, str] = {}
    keys = _find(data, meta[0] + 4, meta[1], [b"keys"])
    if keys is not None:
        count = struct.unpack_from(">I", data, keys[0] + 4)[0]
        pos = keys[0] + 8
        for index in range(1, count + 1):
            size = struct.unpack_from(">I", data, pos)[0]
            key_names[index] = data[pos + 8:pos + size].decode("utf-8", "replace")
            pos += size
    for typ, body, stop in _children(data, *ilst):
        for dtyp, dbody, dstop in _children(data, body, stop):
            if dtyp == b"data" and dstop - dbody >= 8:
                index = int.from_bytes(typ, "big")
                key = key_names.get(index) or names.get(typ, typ.decode("latin-1"))
                tags[key] = data[dbody + 8:dstop].decode("utf-8", "replace")
    return tags


def parse_moov(data: bytes, base_tags: dict[str, str] | None = None) -> Mp4Track:
    """Index of the first video track of a ``moov`` box (`data` = the whole moov box)."""
    _typ, header, size = _box_header(data, 0, len(data))
    start, end = header, size
    tags = dict(base_tags or {})
    udta = _find(data, start, end, [b"udta"])
    if udta:
        tags.update(_ilst_tags(data, *udta))
    for typ, body, stop in _children(data, start, end):
        if typ != b"trak":
            continue
        hdlr = _find(data, body, stop, [b"mdia", b"hdlr"])
        if hdlr is None or data[hdlr[0] + 8:hdlr[0] + 12] != b"vide":
            continue
        mdhd = _find(data, body, stop, [b"mdia", b"mdhd"])
        version, p = _full_box(data, mdhd[0])
        timescale = struct.unpack_from(">I", data, p + (16 if version == 1 else 8))[0]
        stbl = _find(data, body, stop, [b"mdia", b"minf", b"stbl"])
        if stbl is None:
            raise IngestionError("MP4 video track without sample table")
        boxes = {t: (b, s) for t, b, s in _children(data, *stbl)}
        # sample description: codec, size, hvcC
        stsd_body, stsd_end = boxes[b"stsd"]
        entry_typ, entry_body, entry_end = next(_children(data, stsd_body + 8, stsd_end))
        width, height = struct.unpack_from(">HH", data, entry_body + 24)
        length_size, parameter_sets = 4, b""
        hvcc = _find(data, entry_body + 78, entry_end, [b"hvcC"])
        if hvcc:
            length_size, parameter_sets = _parse_hvcc(data, *hvcc)
        # sizes
        b, _ = boxes[b"stsz"]
        _v, p = _full_box(data, b)
        uniform, count = struct.unpack_from(">II", data, p)
        sizes = (np.full(count, uniform, dtype=np.int64) if uniform else
                 np.frombuffer(data, dtype=">u4", count=count, offset=p + 8).astype(np.int64))
        # chunk offsets
        if b"stco" in boxes:
            b, _ = boxes[b"stco"]
            _v, p = _full_box(data, b)
            n = struct.unpack_from(">I", data, p)[0]
            chunk_offsets = np.frombuffer(data, dtype=">u4", count=n, offset=p + 4).astype(np.int64)
        else:
            b, _ = boxes[b"co64"]
            _v, p = _full_box(data, b)
            n = struct.unpack_from(">I", data, p)[0]
            chunk_offsets = np.frombuffer(data, dtype=">u8", count=n, offset=p + 4).astype(np.int64)
        # sample -> chunk
        b, _ = boxes[b"stsc"]
        _v, p = _full_box(data, b)
        n = struct.unpack_from(">I", data, p)[0]
        stsc = np.frombuffer(data, dtype=">u4", count=3 * n, offset=p + 4).astype(np.int64).reshape(n, 3)
        offsets = np.empty(count, dtype=np.int64)
        sample = 0
        for i, (first_chunk, per_chunk, _desc) in enumerate(stsc):
            last_chunk = stsc[i + 1][0] - 1 if i + 1 < n else len(chunk_offsets)
            for chunk in range(first_chunk - 1, last_chunk):
                pos = chunk_offsets[chunk]
                for _ in range(per_chunk):
                    if sample >= count:
                        break
                    offsets[sample] = pos
                    pos += sizes[sample]
                    sample += 1
        if sample != count:
            raise IngestionError(f"MP4 sample table maps {sample} of {count} samples")
        # decode times (stts) + composition offsets (ctts)
        b, _ = boxes[b"stts"]
        _v, p = _full_box(data, b)
        n = struct.unpack_from(">I", data, p)[0]
        stts = np.frombuffer(data, dtype=">u4", count=2 * n, offset=p + 4).astype(np.int64).reshape(n, 2)
        deltas = np.repeat(stts[:, 1], stts[:, 0])
        if len(deltas) != count:
            raise IngestionError(f"MP4 stts covers {len(deltas)} of {count} samples")
        pts = np.concatenate([[0], np.cumsum(deltas)[:-1]]).astype(np.int64)
        if b"ctts" in boxes:
            b, _ = boxes[b"ctts"]
            version, p = _full_box(data, b)
            n = struct.unpack_from(">I", data, p)[0]
            ctts = np.frombuffer(data, dtype=">i4" if version else ">u4", count=2 * n,
                                 offset=p + 4).astype(np.int64).reshape(n, 2)
            pts = pts + np.repeat(ctts[:, 1], ctts[:, 0])
        # sync samples
        keys = np.ones(count, dtype=bool)
        if b"stss" in boxes:
            b, _ = boxes[b"stss"]
            _v, p = _full_box(data, b)
            n = struct.unpack_from(">I", data, p)[0]
            keys[:] = False
            keys[np.frombuffer(data, dtype=">u4", count=n, offset=p + 4).astype(np.int64) - 1] = True
        return Mp4Track(codec=entry_typ.decode("latin-1"), width=int(width), height=int(height), timescale=int(timescale),
                        offsets=offsets, sizes=sizes, keys=keys, pts=pts, length_size=length_size,
                        parameter_sets=parameter_sets, tags=tags)
    raise IngestionError("MP4 file has no video track")


def read_mp4_index(stream: IO[bytes], file_size: int) -> Mp4Track:
    """Find the ``moov`` box by walking the top-level box headers, read it, and index it."""
    off = 0
    while off + 8 <= file_size:
        stream.seek(off)
        head = stream.read(16)
        if len(head) < 8:
            break
        size, typ = struct.unpack_from(">I4s", head, 0)
        if size == 1:
            size = struct.unpack_from(">Q", head, 8)[0]
        elif size == 0:
            size = file_size - off
        if size < 8:
            raise IngestionError(f"bad MP4 top-level box at {off}")
        if typ == b"moov":
            stream.seek(off)
            data = stream.read(size)
            if len(data) != size:
                raise IngestionError("truncated moov box")
            return parse_moov(data)
        off += size
    raise IngestionError("no moov box (the MP4 file is incomplete or not an MP4)")


def nal_types_length_prefixed(sample: bytes, length_size: int) -> list[int]:
    types = []
    pos = 0
    while pos + length_size <= len(sample):
        n = int.from_bytes(sample[pos:pos + length_size], "big")
        pos += length_size
        if n <= 0 or pos + n > len(sample):
            raise IngestionError("corrupt MP4 sample (NAL length outside the sample)")
        types.append((sample[pos] >> 1) & 0x3F)
        pos += n
    return types


def to_annexb(sample: bytes, length_size: int, parameter_sets: bytes = b"") -> bytes:
    """Length-prefixed HEVC sample -> Annex-B packet. `parameter_sets` (VPS/SPS/PPS, Annex-B) are put
    in front when given and the sample does not carry its own VPS, so a keyframe decodes alone."""
    out = bytearray()
    if parameter_sets and _VPS not in nal_types_length_prefixed(sample, length_size):
        out += parameter_sets
    pos = 0
    while pos + length_size <= len(sample):
        n = int.from_bytes(sample[pos:pos + length_size], "big")
        pos += length_size
        if n <= 0 or pos + n > len(sample):
            raise IngestionError("corrupt MP4 sample (NAL length outside the sample)")
        out += b"\x00\x00\x00\x01"
        out += sample[pos:pos + n]
        pos += n
    if pos != len(sample):
        raise IngestionError("corrupt MP4 sample (trailing bytes)")
    return bytes(out)
