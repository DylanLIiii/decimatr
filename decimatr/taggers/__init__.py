"""
Taggers module for stateless frame analysis.

Taggers compute metrics and assign tags to frames without maintaining state.
"""

from decimatr.taggers.base import Tagger
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.entropy import EntropyTagger
from decimatr.taggers.hash import HashTagger

__all__ = ["Tagger", "BlurTagger", "EntropyTagger", "HashTagger"]
