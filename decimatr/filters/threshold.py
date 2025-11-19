"""
Generic threshold-based filter for stateless frame filtering.

This module provides a flexible ThresholdFilter that can filter frames based on
any numeric tag value using configurable comparison operators.
"""

from decimatr.filters.base import StatelessFilter
from decimatr.scheme import VideoFramePacket


class ThresholdFilter(StatelessFilter):
    """
    Generic threshold-based filter for numeric tag values.

    This filter evaluates a single tag value against a threshold using a
    configurable comparison operator. It's a flexible building block for
    creating specific filters like BlurFilter or EntropyFilter.

    Supported operators:
        - '>': Greater than
        - '<': Less than
        - '>=': Greater than or equal to
        - '<=': Less than or equal to
        - '==': Equal to
        - '!=': Not equal to

    Attributes:
        tag_key: The tag key to evaluate
        threshold: The threshold value to compare against
        operator: The comparison operator to use

    Example:
        >>> # Filter frames with blur_score > 100.0
        >>> filter = ThresholdFilter('blur_score', 100.0, '>')
        >>> packet.tags = {'blur_score': 150.0}
        >>> filter.should_pass(packet)
        True

        >>> # Filter frames with entropy >= 4.0
        >>> filter = ThresholdFilter('entropy', 4.0, '>=')
        >>> packet.tags = {'entropy': 3.5}
        >>> filter.should_pass(packet)
        False
    """

    VALID_OPERATORS = {">", "<", ">=", "<=", "==", "!="}

    def __init__(self, tag_key: str, threshold: float, operator: str = ">"):
        """
        Initialize threshold filter.

        Args:
            tag_key: The tag key to evaluate (e.g., 'blur_score', 'entropy')
            threshold: The threshold value to compare against
            operator: Comparison operator (one of: '>', '<', '>=', '<=', '==', '!=')
                     Default is '>' (greater than)

        Raises:
            ValueError: If operator is not one of the valid operators
        """
        if operator not in self.VALID_OPERATORS:
            raise ValueError(f"operator must be one of {self.VALID_OPERATORS}, got '{operator}'")

        self.tag_key = tag_key
        self.threshold = threshold
        self.operator = operator

    def should_pass(self, packet: VideoFramePacket) -> bool:
        """
        Determine if frame passes the threshold filter.

        Evaluates the specified tag value against the threshold using the
        configured comparison operator. If the tag is missing, the frame
        is filtered out (returns False).

        Args:
            packet: VideoFramePacket containing frame data and tags

        Returns:
            True if the tag value satisfies the threshold condition,
            False if the tag is missing or doesn't satisfy the condition
        """
        # Get tag value, return False if missing
        tag_value = packet.get_tag(self.tag_key)
        if tag_value is None:
            return False

        # Apply comparison operator
        if self.operator == ">":
            return tag_value > self.threshold
        elif self.operator == "<":
            return tag_value < self.threshold
        elif self.operator == ">=":
            return tag_value >= self.threshold
        elif self.operator == "<=":
            return tag_value <= self.threshold
        elif self.operator == "==":
            return tag_value == self.threshold
        elif self.operator == "!=":
            return tag_value != self.threshold

        # Should never reach here due to validation in __init__
        return False

    @property
    def required_tags(self) -> list[str]:
        """
        Return list of required tag keys.

        Returns:
            List containing the single tag key this filter evaluates
        """
        return [self.tag_key]

    def __repr__(self) -> str:
        """String representation of the filter."""
        return (
            f"ThresholdFilter(tag_key='{self.tag_key}', "
            f"threshold={self.threshold}, operator='{self.operator}')"
        )
