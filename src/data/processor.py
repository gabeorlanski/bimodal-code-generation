from src.common import Registrable


class Preprocessor(Registrable):
    """
    Just a wrapper for the registrable so that preprocessing
    functions can be registered.
    """
    pass


class Postprocessor(Registrable):
    """
    Just a wrapper for the registrable so that postprocessing functions can be
    registered.
    """
    pass
