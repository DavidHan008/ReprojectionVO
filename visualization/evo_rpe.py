from evo import main_rpe
import argparse
import logging

logger = logging.getLogger(__name__)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def merge_config(args):
    """
    merge .json config file with the command line args (if --config was defined)
    :param args: parsed argparse NameSpace object
    :return: merged argparse NameSpace object
    """
    import json
    if args.config:
        with open(args.config) as config:
            merged_config_dict = vars(args).copy()
            merged_config_dict.update(json.loads(config.read()))  # merge both parameter dicts
            args = argparse.Namespace(**merged_config_dict)  # override args the hacky way
    return args


def launch():
    args = Namespace(align=False,
                     all_pairs=False,
                     config=None,
                     correct_scale=False,
                     debug=False,
                     delta=1,
                     delta_tol=0.1,
                     delta_unit='f',
                     est_file='logs/pos_results_ceres.txt',
                     no_warnings=False,
                     plot=True,
                     plot_colormap_max=None,
                     plot_colormap_max_percentile=None,
                     plot_colormap_min=None,
                     plot_mode='xyz',
                     pose_relation='trans_part',
                     ref_file='logs/groundtruth.txt',
                     save_plot=None,
                     save_results='logs/evo_rpe.zip',
                     serialize_plot=None,
                     silent=False,
                     subcommand='tum',
                     t_max_diff=0.01,
                     t_offset=0.0,
                     verbose=True)
    # args = parser.parse_args()
    if hasattr(args, "config"):
        args = merge_config(args)
    import sys
    from evo.tools import settings
    try:
        main_rpe.run(args)
    except SystemExit as e:
        sys.exit(e.code)
    except:
        logger.exception("Unhandled error in " + main_rpe.__name__)
        print("")
        err_msg = "evo module " + main_rpe.__name__ + " crashed"
        if settings.SETTINGS.logfile_enabled:
            err_msg += " - see " + settings.DEFAULT_LOGFILE_PATH
        else:
            err_msg += " - no logfile written (disabled)"
        logger.error(err_msg)
        from evo.tools import user
        if not args.no_warnings:
            if settings.SETTINGS.logfile_enabled and user.confirm("Open logfile? (y/n)"):
                import webbrowser
                webbrowser.open(settings.DEFAULT_LOGFILE_PATH)
        sys.exit(1)