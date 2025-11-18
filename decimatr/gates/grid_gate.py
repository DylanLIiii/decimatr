import logging
import uuid

import imagehash
from PIL import Image

from decimatr.gates.base_gate import BaseGate
from decimatr.scheme import VideoFramePacket


class GridGate(BaseGate[str]):
    # FIXME: these are not used by buffer. Because it should be a one shot check instead of a sliding window compare.
    DEFAULT_GRID_ROWS = 4
    DEFAULT_GRID_COLS = 4
    DEFAULT_CELL_HASH_SIZE = 8
    DEFAULT_SIMILARITY_THRESHOLD = 5

    def __init__(
        self,
        grid_rows: int = DEFAULT_GRID_ROWS,
        grid_cols: int = DEFAULT_GRID_COLS,
        cell_hash_size: int = DEFAULT_CELL_HASH_SIZE,
        similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
        session_id: str = "default_session",
    ):
        """
        Initializes the GridGate component.

        Args:
            grid_rows (int): Number of rows in the grid.
            grid_cols (int): Number of columns in the grid.
            cell_hash_size (int): Hash size for perceptual hashing of each cell.
            similarity_threshold (int): Max Hamming distance for grid signatures to be considered similar.
            session_id (str): Identifier for the current session, used in logging.
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cell_hash_size = cell_hash_size
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(f"Decimatr.{self.__class__.__name__}")
        self.session_id = session_id

        # Store signatures to compare against
        self.stored_signatures = []

    def _get_pil_image(self, packet: VideoFramePacket) -> Image.Image:
        """
        Convert the frame data to a PIL Image.

        Args:
            packet (VideoFramePacket): The packet containing the frame data

        Returns:
            Image.Image: The PIL Image representation of the frame
        """
        return Image.fromarray(packet.frame_data)

    def _extract_grid_signature(self, packet: VideoFramePacket) -> str:
        """
        Extracts a grid-based signature from a frame.

        Args:
            packet (VideoFramePacket): The packet containing the frame to analyze

        Returns:
            str: The concatenated string of cell hashes.
        """
        pil_image = self._get_pil_image(packet)
        img_width, img_height = pil_image.size

        if self.grid_cols <= 0 or self.grid_rows <= 0:
            return str(imagehash.average_hash(pil_image, hash_size=self.cell_hash_size))

        base_cell_width = img_width // self.grid_cols
        base_cell_height = img_height // self.grid_rows

        cell_hashes = []
        placeholder_hex_len = (self.cell_hash_size * self.cell_hash_size) // 4
        placeholder_hash = "0" * placeholder_hex_len

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                left = c * base_cell_width
                upper = r * base_cell_height
                right = min((c + 1) * base_cell_width, img_width)
                lower = min((r + 1) * base_cell_height, img_height)

                if right > left and lower > upper:
                    cell_image = pil_image.crop((left, upper, right, lower))
                    try:
                        cell_hash_obj = imagehash.average_hash(
                            cell_image, hash_size=self.cell_hash_size
                        )
                        cell_hashes.append(str(cell_hash_obj))
                    except Exception:
                        cell_hashes.append(placeholder_hash)
                else:
                    cell_hashes.append(placeholder_hash)

        return "".join(cell_hashes)

    def _is_similar_signature(self, signature: str) -> bool:
        """
        Check if the given signature is similar to any stored signatures.

        Args:
            signature (str): The signature to check.

        Returns:
            bool: True if similar to any stored signature, False otherwise.
        """
        for stored_signature in self.stored_signatures:
            if len(signature) != len(stored_signature):
                # Consider signatures with different lengths as different
                continue

            # Calculate Hamming distance
            distance = sum(
                1 for i in range(len(signature)) if signature[i] != stored_signature[i]
            )
            # If the distance is within the threshold, signatures are considered similar
            if distance <= self.similarity_threshold:
                return True

        # If no similar signature found, return False
        return False

    def process_frame(self, packet: VideoFramePacket) -> bool:
        """
        Process a single video frame packet using grid-based signatures to determine if
        it is similar to previously seen frames.

        Args:
            packet (VideoFramePacket): The packet containing the frame to analyze

        Returns:
            bool: True if the frame is unique (passes the gate), False if it's a duplicate (filtered)
        """
        # Extract grid signature for the current frame
        signature = self._extract_grid_signature(packet)

        # Check if similar to any stored signatures
        is_duplicate = self._is_similar_signature(signature)

        # Logging
        log_metadata = {
            "frame_number": packet.frame_number,
            "timestamp": packet.timestamp,
            "source_video_id": packet.source_video_id,
            "signature_generated": signature[:20] + "..."
            if len(signature) > 20
            else signature,  # Truncate for logging
            "decision_is_duplicate": bool(is_duplicate),
        }
        log_dict = {
            "component_name": self.__class__.__name__,
            "operation": "process_frame_grid_check",
            "outcome": "duplicate" if is_duplicate else "unique",
            "event_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "relevant_metadata": log_metadata,
        }
        self.logger.info(
            f"Frame {packet.frame_number} processed with grid signature.",
            extra=log_dict,
        )

        # If not a duplicate, store the signature for future comparisons
        if not is_duplicate:
            self.stored_signatures.append(signature)

        # Return True if the frame is unique (passes the gate), False if duplicate (filtered)
        return not is_duplicate

    def clear_signatures(self) -> None:
        """
        Clears the stored signatures.
        """
        self.stored_signatures = []
