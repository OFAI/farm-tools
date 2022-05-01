"""
Various utilities that could be useful in several modules.
"""
import numbers
import sys
import os
import logging
import logging.config
import datetime
import time
from functools import wraps


start = 0
LOGGING_FORMAT = "%(asctime)s|%(levelname)s|%(name)s|%(message)s"


def init_logger(name=None, file=None, lvl=None, config=None, debug=False, args=None, fmt=None):
    """
    Configure the root logger (this only works the very first time, all subsequent
    invocations will not modify the root logger). The root logger is initialized
    with a standard format the given log level and, if specified the outputs to the
    given file.

    The get a new logger for the given name is retrieved using the given name or
    the invoking command if None. It is also set to the given logging leve and returned.

    TODO: If file is not given but args is given and has "outpref" parameter, log to
    file "outpref.DATETIME.log" as well.

    Args:
        name: name to use in the log, if None, __name__
        file: if given, log to this destination in addition to stderr
        lvl: set logging level
        config: if specified, set logger config from this file
        debug: if true, set the level to DEBUG
        args: not used yet
        fmt: logging format to use, if None uses a default format

    Returns:
        A logger instance for name (always the same instance for the same name)
    """

    if name is None:
        name = sys.argv[0]
    if fmt is None:
        fmt = LOGGING_FORMAT
    if lvl is None:
        if debug:
            lvl = logging.DEBUG
        else:
            lvl = logging.INFO
    if config:
        # NOTE we could also configure from a yaml file or a dictionary, see
        # http://zetcode.com/python/logging/
        # see doc on logging.config
        logging.config.fileConfig(fname=config)
    # get the root logger
    rl = logging.getLogger()
    rl.setLevel(lvl)
    # NOTE: basicConfig does nothing if there is already a handler, so it only runs once, but we create the additional
    # handler for the file, if needed, only if the root logger has no handlers yet as well
    addhandlers = []
    fmt = logging.Formatter(fmt)
    hndlr = logging.StreamHandler(sys.stderr)
    hndlr.setFormatter(fmt)
    addhandlers.append(hndlr)
    if file and len(logging.getLogger().handlers) == 0:
        hndlr = logging.FileHandler(file)
        hndlr.setFormatter(fmt)
        addhandlers.append(hndlr)
    logging.basicConfig(level=lvl, handlers=addhandlers)
    # now get the handler for name
    logger = logging.getLogger(name)
    return logger


def run_start(logger=None, name=None, lvl=None):
    """
    Define time when running starts.

    Returns:
        system time in seconds
    """
    global start
    if logger is None:
        logger = init_logger(name=name, lvl=lvl)
    logger.info(
        "Started: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M%S"))
    )
    start = time.time()
    return start


def run_stop(logger=None, name=None):
    """
    Log and return formatted elapsed run time.

    Returns:
        tuple of formatted run time, run time in seconds
    """
    if logger is None:
        logger = init_logger(name=name)
    logger.info(
        "Stopped: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M%S"))
    )
    if start == 0:
        logger.warning("Run timing not set up properly, no time!")
        return "", 0
    stop = time.time()
    delta = stop - start
    deltastr = str(datetime.timedelta(seconds=delta))
    logger.info(f"Runtime: {deltastr}")
    return deltastr, delta


def file4logger(thelogger, noext=False):
    """
    Return the first logging file found for this logger or None if there is no file handler.

    Args:
        thelogger: logger

    Returns:
        file path (string)
    """
    lpath = None
    for h in thelogger.handlers:
        if isinstance(h, logging.FileHandler):
            lpath = h.baseFilename
            if noext:
                lpath = os.path.splitext(lpath)[0]
            break
    return lpath

