import logging


class SplitStreamLogger(logging.Logger):
    """
    Dual Logger for log_util info messages and below to a certain handler. Then
    Warning and above are sent to a different one.
    """

    def __init__(self, name):
        super(SplitStreamLogger, self).__init__(name)

        self.error_handlers = []
        self.normal_handlers = []
        parent = self.parent
        while parent is not None and parent.name != "root":
            parent = parent.parent

        if isinstance(parent, SplitStreamLogger):
            self.error_handlers = getattr(parent, "error_handlers")
            self.normal_handlers = getattr(parent, "normal_handlers")

    def addErrorHandler(self, handler):
        self.error_handlers.append(len(self.handlers))
        self.addHandler(handler)

    def addNormalHandler(self, handler):
        self.normal_handlers.append(len(self.handlers))
        self.addHandler(handler)

    def addNormalFileHandler(self, handler):
        self.error_handlers.append(len(self.handlers))
        self.normal_handlers.append(len(self.handlers))
        self.addHandler(handler)

    def callHandlers(self, record):
        handlers_used = []
        if record.levelno >= logging.WARNING:
            if not self.error_handlers:
                raise ValueError(
                    f"Trying to log error with SplitStreamLogger, but no error handlers defined."
                )
            for handler_index in self.error_handlers:
                self.handlers[handler_index].handle(record)
                handlers_used.append(handler_index)
        else:
            if not self.normal_handlers:
                raise ValueError(
                    f"Trying to log with SplitStreamLogger, but no normal loggers defined."
                )
            for handler_index in self.normal_handlers:
                if record.levelno >= self.handlers[handler_index].level:
                    if handler_index not in handlers_used:
                        self.handlers[handler_index].handle(record)
                        handlers_used.append(handler_index)
