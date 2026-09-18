"""Neural text classification architectures."""

from fs_lrtc.models.classic import BiLSTMClassifier, TextCNNClassifier, WordAveragingClassifier

__all__ = ["WordAveragingClassifier", "TextCNNClassifier", "BiLSTMClassifier"]
