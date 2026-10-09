"""Write-Ahead Log (WAL) engine for atomic transactions and crash consistency."""

import json
import os
import struct
import zlib
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
import numpy as np


class WALRecordType:
    BEGIN = 1
    INSERT = 2
    COMMIT = 3
    CHECKPOINT = 4
    DELETE = 5


class WALEntry(NamedTuple):
    tx_id: int
    op_type: int
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    vector_bytes: bytes


class WALManager:
    """Manages write-ahead logging with CRC32 integrity validation and fsync durability.
    
    Binary Record Format:
        [MAGIC: 4B (0x57414C52 'WALR')]
        [RECORD_LEN: 4B uint32]
        [TX_ID: 8B uint64]
        [OP_TYPE: 1B uint8]
        [PAYLOAD: RECORD_LEN bytes]
        [CRC32: 4B uint32 (checksum over PAYLOAD)]
    """

    MAGIC_HEADER = b"POPWAL01"
    RECORD_MAGIC = b"WALR"
    HEADER_SIZE = len(MAGIC_HEADER)
    # 4B magic + 4B len + 8B tx_id + 1B op_type = 17B prefix + 4B CRC suffix = 21B framing
    FRAME_OVERHEAD = 4 + 4 + 8 + 1 + 4

    def __init__(self, wal_path: str):
        self.wal_path = wal_path
        self._file = None
        self._current_tx_id = 0
        self._init_wal_file()

    def _init_wal_file(self) -> None:
        """Initialize WAL file, scan max tx_id, and open handle."""
        if not os.path.exists(self.wal_path) or os.path.getsize(self.wal_path) == 0:
            with open(self.wal_path, "wb") as f:
                f.write(self.MAGIC_HEADER)
                f.flush()
                os.fsync(f.fileno())

        max_tx = 0
        for tx_id, _, _ in self.read_all_records():
            if tx_id > max_tx:
                max_tx = tx_id
        self._current_tx_id = max_tx

        self._file = open(self.wal_path, "a+b")

    def next_tx_id(self) -> int:
        """Generate monotonically increasing transaction ID."""
        self._current_tx_id += 1
        return self._current_tx_id

    def _write_record(self, tx_id: int, op_type: int, payload: bytes) -> None:
        """Write framed, checksummed record and execute fsync."""
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        record_len = len(payload)
        header = struct.pack(">4sIQB", self.RECORD_MAGIC, record_len, tx_id, op_type)
        footer = struct.pack(">I", crc)

        self._file.write(header)
        self._file.write(payload)
        self._file.write(footer)
        self._file.flush()
        os.fsync(self._file.fileno())

    def begin_transaction(self, tx_id: int, record_count: int) -> None:
        """Log transaction beginning."""
        payload = struct.pack(">I", record_count)
        self._write_record(tx_id, WALRecordType.BEGIN, payload)

    def log_insert(
        self,
        tx_id: int,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
        vector: np.ndarray,
    ) -> None:
        """Log record insert with vector bytes."""
        doc_id_bytes = doc_id.encode("utf-8")
        text_bytes = text.encode("utf-8")
        meta_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        vec_bytes = vector.astype(np.float32).tobytes()

        # Payload format:
        # [id_len: uint16][id_bytes][text_len: uint32][text_bytes][meta_len: uint32][meta_bytes][vec_len: uint32][vec_bytes]
        payload = (
            struct.pack(">HI", len(doc_id_bytes), len(text_bytes))
            + doc_id_bytes
            + text_bytes
            + struct.pack(">II", len(meta_bytes), len(vec_bytes))
            + meta_bytes
            + vec_bytes
        )
        self._write_record(tx_id, WALRecordType.INSERT, payload)

    def log_delete(self, tx_id: int, doc_id: str) -> None:
        """Log soft-delete tombstone record."""
        doc_id_bytes = doc_id.encode("utf-8")
        payload = struct.pack(">H", len(doc_id_bytes)) + doc_id_bytes
        self._write_record(tx_id, WALRecordType.DELETE, payload)

    def commit_transaction(self, tx_id: int) -> None:
        """Log transaction commit."""
        payload = struct.pack(">Q", tx_id)
        self._write_record(tx_id, WALRecordType.COMMIT, payload)

    def checkpoint(self) -> None:
        """Log checkpoint and truncate WAL file back to header."""
        if self._file:
            self._file.close()

        with open(self.wal_path, "wb") as f:
            f.write(self.MAGIC_HEADER)
            f.flush()
            os.fsync(f.fileno())

        self._file = open(self.wal_path, "a+b")

    def read_all_records(self) -> Iterator[Tuple[int, int, bytes]]:
        """Scan WAL file and yield validated (tx_id, op_type, payload) tuples."""
        if not os.path.exists(self.wal_path):
            return

        with open(self.wal_path, "rb") as f:
            header = f.read(len(self.MAGIC_HEADER))
            if header != self.MAGIC_HEADER:
                return

            while True:
                prefix = f.read(17)  # 4B magic + 4B len + 8B tx + 1B op
                if len(prefix) < 17:
                    break

                magic, record_len, tx_id, op_type = struct.unpack(">4sIQB", prefix)
                if magic != self.RECORD_MAGIC:
                    break

                payload = f.read(record_len)
                if len(payload) < record_len:
                    break

                footer = f.read(4)
                if len(footer) < 4:
                    break

                expected_crc, = struct.unpack(">I", footer)
                actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
                if actual_crc != expected_crc:
                    break

                yield (tx_id, op_type, payload)

    @staticmethod
    def parse_insert_payload(payload: bytes) -> Tuple[str, str, Dict[str, Any], np.ndarray]:
        """Deserialize raw payload from INSERT record into components."""
        id_len, text_len = struct.unpack_from(">HI", payload, 0)
        offset = 6

        doc_id = payload[offset : offset + id_len].decode("utf-8")
        offset += id_len

        text = payload[offset : offset + text_len].decode("utf-8")
        offset += text_len

        meta_len, vec_len = struct.unpack_from(">II", payload, offset)
        offset += 8

        meta_json = payload[offset : offset + meta_len].decode("utf-8")
        offset += meta_len

        metadata = json.loads(meta_json)
        vec_bytes = payload[offset : offset + vec_len]

        vector = np.frombuffer(vec_bytes, dtype=np.float32).copy()
        return doc_id, text, metadata, vector

    def recover(self) -> List[Tuple[str, str, Dict[str, Any], np.ndarray]]:
        """Recover all COMMITTED records from WAL.
        
        Ignores uncommitted transactions to maintain strict atomicity.
        Returns:
            List of (doc_id, text, metadata, vector) tuples for committed transactions.
        """
        committed_tx_ids = set()
        transactions: Dict[int, List[bytes]] = {}
        max_tx_id = 0

        for tx_id, op_type, payload in self.read_all_records():
            max_tx_id = max(max_tx_id, tx_id)
            if op_type == WALRecordType.BEGIN:
                transactions[tx_id] = []
            elif op_type == WALRecordType.INSERT:
                if tx_id not in transactions:
                    transactions[tx_id] = []
                transactions[tx_id].append(payload)
            elif op_type == WALRecordType.COMMIT:
                committed_tx_ids.add(tx_id)

        self._current_tx_id = max_tx_id

        recovered_records = []
        for tx_id in sorted(transactions.keys()):
            if tx_id in committed_tx_ids:
                for payload in transactions[tx_id]:
                    recovered_records.append(self.parse_insert_payload(payload))

        return recovered_records

    def close(self) -> None:
        """Close WAL file descriptor."""
        if self._file and not self._file.closed:
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except OSError:
                pass
            self._file.close()
            self._file = None
