import sys
import os
from common import utils, import_tilespecs, bounding_box, trans_models
from stitching import create_features, match_features, optimize_2d
from alignment import pre_match, block_match, optimize_3d, normalize_coordinates
import renderer


if __name__ == '__main__':
    # Step 0: Notification of the argument settings.
    print(utils.to_red('Step 0 -- All the running arguments are stored in the folder "arguments".'
                       'Have you confirmed that all the arguments are set as you want? (yes/no)'))
    ans = input()
    if ans == 'yes':
        print(utils.to_green('Arguments confirmed, start.'))
    elif ans == 'no':
        sys.exit('Please double check the arguments in the "arguments" folder and run again.')
    else:
        sys.exit('Please enter yes or no.')
    overall_args = utils.load_json_file('../arguments/overall_args.json')
    utils.create_dir(overall_args["base"]["workspace"])
    log_controller = utils.LogController('main', os.path.join(overall_args["base"]["workspace"], 'log'),
                                         overall_args["base"]["running_mode"])

    # Step 1: Check the data and generate tilespecs files.
    