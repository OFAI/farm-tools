#!/usr/bin/env python
"""
BERT classification/ordinal regression model hyper parameter search

Allow to specify the same parameters as for estimation, plus a config file to
specify parameter ranges to explore. The ones specified on the command line serve as a
single value fallback
if the config does not specify a value/range for them.

This creates a tsv file that contains one column for all parameters specified in any
parameter exploration set in the config file.
"""
import sys
import socket
from copy import deepcopy, copy
import toml
import farm_lib
import random
import socket
from gatenlp.utils import init_logger
from farm_lib import run_estimate

logger = init_logger()

def expand_all(hcfg):
    """
    Given several configs, expands all combinations of parameters, adds parameter "set"
    """

    all_names = set()
    all_names.add("set")
    all_names.add("idx")
    all_settings = []   # a list of dictionaries
    cursettings = []
    for k, v in hcfg.items():
        # k is the name of the set, v is a dict with the parm settings for that set (scalar or list)
        setparms = list(v.items())
        # for each list of parameters, take the existing list of settings and create a new
        # list of settings which has all the dicts of the existing list expanded with the parameter and one value
        # we start with a list that contains one empty dict
        cursettings = [{"set": k}]
        for setparm_name, setparm_values in setparms:
            all_names.add(setparm_name)
            if not isinstance(setparm_values, list):  # assume it is a scalar!
                setparm_values = [setparm_values]
            newsettings = []
            for cursetting in cursettings:
                for setparm_value in setparm_values:
                    newsetting = cursetting.copy()
                    newsetting[setparm_name] = setparm_value
                    newsettings.append(newsetting)
            cursettings = newsettings
        all_settings.extend(cursettings)
    all_names = list(all_names)
    all_names.sort()
    logger.info(f"HSEARCH: all parms: {all_names}")
    idx = 0
    for setting in all_settings:
        setting["idx"] = idx
        idx += 1
        logger.info(f"HSEARCH: setting: {dict2list(setting, all_names)}")
    return all_settings, all_names


def dict2list(thedict, parmnames, default=None):
    """
    Convert the values in thedict for the given list of parmnames to a list of values.
    """
    return [thedict.get(n, default) for n in parmnames]


def ret2dict(estimationresult):
    """
    Convert the estimationresult to a single dictionary of key/value pairs.

    The estimationresult is a list of dicts, where each dict corresponds to a head and has a `task_name`
    key. The return value is a merged dict which has all keys prefixed with the task name and the task name
    keys removed, as well as the keys "report" and "confusion" and all other entries where the value is a list or dict.
    """
    new = {}
    for d in estimationresult:
        name = d["task_name"]
        for k, v in d.items():
            if k != "task_name" and k != "report" and not k.startswith("confusion") and not isinstance(v, (list, dict)):
                new[name+"_"+k] = v
    return new


def cfgval2str(val):
    if "object at" in str(val):
        return type(val).__name__
    else:
        return str(val)


def add_config(cfg, toadd):
    for k, v in toadd.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (list, tuple)):
            cfg[k] = ";".join(cfgval2str(x) for x in v)
        else:
            cfg[k] = v
    return cfg


if __name__ == "__main__":

    cfg = farm_lib.getargs(*farm_lib.argparser_hsearch())
    configfile = cfg["hcfg"]
    hcfg = toml.load(configfile)
    # check that the config file ONLY contains settings with a group header, so hconfig may only
    # contain dictionaries
    for k,v in hcfg.items():
        if not isinstance(v, dict):
            raise Exception(f"TOML config file {configfile}: all entries must be grouped, but found {k}")
    outpref = cfg["outpref"]
    # the format of the config file must be:
    # either just a list of parameters, each with a scalar or list of values
    # or several named blocks of such values which will then get combined to form the whole search space
    logger.info(f"CONFIG={hcfg}")
    halg = cfg["halg"]
    assert halg in ["greedy", "grid", "random", "beam"]
    assert cfg["est_cmp"] in ["min", "max"]
    logger.info(f"Using algorithm: {halg}")
    if halg == "greedy":
        # here we need to make sure we use the settings in the same order as they occur in the
        # config file settings.
        # For each set
        #   Pass one: collect the first or only value of all settings in a set and start with that
        #   For each parameter, go through all settings and pick the best, fix that parameter and do
        #      the same for the next
        #   At the end we found the optimal greedy search combination of parameters for that set
        #   That result gets stored, then process all sets in the same way
        for setname, setdata in hcfg.items():
            setparms = list(setdata.items())
            # process the set
            start_settings = {}
            for setparm_name, setparm_values in setparms:
                if isinstance(setparm_values, list):
                    start_settings[setparm_name] = setparm_values[0]
                else:
                    start_settings[setparm_name] = setparm_values
            # now do the actual greedy search
            for setparm_name, setparm_values in setparms:
                if isinstance(setparm_values, list):
                    # evaluate with each of the values, find best
                    raise Exception("Greedy not yet fully implemented")
                    pass
                else:
                    best_val = setparm_values

    elif halg == "grid":
        all_settings, all_names = expand_all(hcfg)
        # to perform grid search we simple add the parameters from all_settings to the initial cfg and run
        # estimation for this, then combine the hsearch parameters with the result we got from the
        # estimation run into a data line
        datas = []
        n_avail = len(all_settings)
        logger.info(f"Running strategy random, number of settings: {n_avail}")
        tmpfp = open(cfg["outpref"] + ".tmp.tsv", "wt", encoding="utf-8")
        have_hdr = False
        for i, setting in enumerate(all_settings):
            logger.info(f"HSEARCH GRID RUN for settings: {setting}")
            tmp_cfg = copy(cfg)
            tmp_cfg.update(setting)
            try:
                ret = run_estimate(tmp_cfg, logger)
            except Exception as ex:
                logger.error(f"HSEARCH ERROR: {ex} for {setting}", stack_info=True)
                continue
            # for debugging
            # ret = [{"task_name": "taskname", "head0_f1_macro_mean": 0.1}]
            tmp_data = {k: "" for k in all_names}
            tmp_data["hostname"] = socket.gethostname()
            add_config(tmp_data, tmp_cfg)
            tmp_data.update(setting)
            retasdict = ret2dict(ret)
            tmp_data.update(retasdict)
            logger.info(f"Have a data: {tmp_data}")
            datas.append(tmp_data)
            logger.info(f"Completed estimation run {i+1} of {n_avail}")
            logger.info(f"Used parameters: {setting}")
            logger.info(f"Estimation results: {retasdict}")
            tmp_names = sorted(list(tmp_data.keys()))
            if not have_hdr:
                print("\t".join(tmp_names), file=tmpfp)
                have_hdr = True
            print("\t".join([str(x) for x in dict2list(tmp_data, tmp_names, default="")]), file=tmpfp)
            tmpfp.flush()
        tmpfp.close()
        if len(datas) == 0:
            logger.error("All trials failed, nothing to write out, aborting program!")
            sys.exit()
        # now datas contains all dictionaries
        # get all the parameters from the first entry
        #parmnames = list(datas[0].keys())
        #parmnames.sort()
        # sort the datas by the target estimation variable
        reverse = (cfg["est_cmp"] == "max")
        est_var = cfg["est_var"]
        tmp = datas[0].get(est_var)
        if tmp is None:
            # we forgot to specify the correct estimation variable name, fall back to an existing one
            # if possible, because having this crash after running a long time would be annoying
            logger.warning(f"PROBLEM: the estimation variable (--est_var) {est_var} is not found in the combined results")
            logger.warning(f"Possible names: {list(datas[0].keys())}")
            logger.warning(f"Trying to find one automatically")
            cands = [x for x in datas[0].keys() if x.endswith("f1_macro_mean")]
            if len(cands) == 0:
                cands = [x for x in datas[0].keys() if x.endswith("acc_mean")]
            if len(cands) > 0:
                est_var = cands[0]
                logger.warning(f"Using {est_var}")
            else:
                est_var = None
                logger.warning(f"No replacement found, not sorting the output file!")
        if est_var is not None:
            datas.sort(key=lambda x: x[est_var], reverse=reverse)
        tmp_names = sorted(list(datas[0].keys()))
        with open(cfg["outpref"]+".tsv", "wt", encoding="utf-8") as outfp:
            print("\t".join(tmp_names), file=outfp)
            for data in datas:
                ldata = [str(x) for x in dict2list(data, tmp_names, default="")]
                print("\t".join(ldata), file=outfp)
    elif halg == "random":
        all_settings, all_names = expand_all(hcfg)
        # to perform random search we execute N estimation runs on one randomly chosen setting, then
        # remove the setting for the next round
        datas = []
        n_avail = len(all_settings)
        n_run = cfg["halg_random_n"]
        n_actual = min(n_avail, n_run)
        logger.info(f"Running strategy random, available settings: {n_avail}, requested: {n_run}, actual: {n_actual}")
        tmpfp = open(cfg["outpref"] + ".tmp.tsv", "wt", encoding="utf-8")
        have_hdr = False
        for i in range(n_actual):
            tmp_cfg = copy(cfg)
            choose_idx = random.randint(0, len(all_settings)-1)
            setting = all_settings[choose_idx]
            logger.info(f"HSEARCH RANDOM RUN for settings: {setting}")
            tmp_cfg.update(setting)
            try:
                ret = run_estimate(tmp_cfg, logger)
            except Exception as ex:
                logger.error(f"HSEARCH ERROR: {ex} for {setting}", stack_info=True)
                continue
            tmp_data = {k: "" for k in all_names}
            tmp_data["hostname"] = socket.gethostname()
            add_config(tmp_data, tmp_cfg)
            tmp_data.update(setting)
            retasdict = ret2dict(ret)
            tmp_data.update(retasdict)
            datas.append(tmp_data)
            del all_settings[choose_idx]
            tmp_names = sorted(list(tmp_data.keys()))
            if not have_hdr:
                print("\t".join(tmp_names), file=tmpfp)
                have_hdr = True
            print("\t".join([str(x) for x in dict2list(tmp_data, tmp_names, default="")]), file=tmpfp)
            tmpfp.flush()
            logger.info(f"Completed estimation run {i+1} of {n_actual}")
            logger.info(f"Used parameters: {setting}")
            logger.info(f"Estimation results: {retasdict}")
        tmpfp.close()
        if len(datas) == 0:
            logger.error("All trials failed, nothing to write out, aborting program!")
            sys.exit()
        # now datas contains all dictionaries
        # get all the parameters from the first entry
        #parmnames = list(datas[0].keys())
        #parmnames.sort()
        # sort the datas by the target estimation variable
        reverse = (cfg["est_cmp"] == "max")
        est_var = cfg["est_var"]
        tmp_names = sorted(list(datas[0].keys()))
        datas.sort(key=lambda x: x[est_var], reverse=reverse)
        with open(cfg["outpref"]+".tsv", "wt", encoding="utf-8") as outfp:
            print("\t".join(tmp_names), file=outfp)
            for data in datas:
                ldata = [str(x) for x in dict2list(data, tmp_names, default="")]
                print("\t".join(ldata), file=outfp)
    elif halg == "beam":
        raise Exception("Alg beam not yet implemented")