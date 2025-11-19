"""
Taggers module for stateless frame analysis.

Taggers compute metrics and assign tags to frames without maintaining state.
"""

from decimatr.taggers.base import Tagger
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.entropy import EntropyTagger
from decimatr.taggers.hash import HashTagger

# CLIPTagger is optional and requires GPU dependencies
# Import it only if dependencies are available
try:
    from decimatr.taggers.clip import CLIPTagger

    __all__ = ["Tagger", "BlurTagger", "EntropyTagger", "HashTagger", "CLIPTagger"]
except ImportError:
    # GPU dependencies not available, CLIPTagger not exported
    __all__ = ["Tagger", "BlurTagger", "EntropyTagger", "HashTagger"]
