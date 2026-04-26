"""Base ForceField class. Subclasses provide concrete parsers/writers.

Logic ported verbatim from the pre-Phase-1 top-level `ForceField.py`. Once
the rest of Phase 1 is wired up the legacy module becomes a thin re-export
of this one and is removed in Phase 2.
"""


class ForceField:
    """Common interface for every force-field type.

    Attributes
    ----------
    param_min_max_delta : dict[tuple, dict[str, float]]
        Mapping from a `(section, entry, item)` tuple to the move-bounds
        dict `{'delta': ..., 'min': ..., 'max': ...}` for the optimizer.
    """

    def __init__(self, ff_filepath_input, ParamSelect_filePath_input):
        self.params = {}
        self.param_selection = []
        self.param_min_max_delta = {}
        self._param_selected = 0
        self.ff_filePath = ff_filepath_input
        self.ParamSelect_filePath = ParamSelect_filePath_input

    def write_forcefield(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError

    def parseParamSelectionFile(self):  # pragma: no cover - abstract
        raise NotImplementedError


def list_to_dict(input_list):
    """[a, b, c] → {1: a, 2: b, 3: c} — used to keep ReaxFF item indexing 1-based."""
    return {(i + 1): input_list[i] for i in range(len(input_list))}
