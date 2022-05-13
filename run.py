import sys
import os
from tqdm import tqdm
import shutil
from common import utils, import_tilespecs, bounding_box, trans_models
# from stitching import create_features, match_features, optimize_2d
# from alignment import pre_match, block_match, optimize_3d, normalize_coordinates
# import renderer


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
    overall_args = utils.load_json_file('arguments/overall_args.json')
    utils.create_dir(overall_args["base"]["workspace"])
    shutil.copy('arguments/overall_args.json', os.path.join(overall_args["base"]["workspace"], 'overall_args.json'))
    log_controller = utils.LogController('main', os.path.join(overall_args["base"]["workspace"], 'log'))

    # Step 1: Check the data and generate tilespecs files.
    print(utils.to_red('Step 1 -- Check the data and generate tilespecs files.'))
    tilespecs_folder_path = os.path.join(overall_args["base"]["workspace"], 'tilespecs')
    utils.create_dir(tilespecs_folder_path)
    section_folder_paths = utils.ls_sub_folder_paths(overall_args["base"]["sample_folder"])
    if overall_args["base"]["EM_type"] == 'singlebeam':
        for section_folder_path in tqdm(section_folder_paths):
            if not utils.check_EM_type(section_folder_path) == 'singlebeam':
                log_controller.error('The EM type you declared in the overall_args.json(singlebeam) is different '
                                     'from the parsing result({}).'.format(utils.check_EM_type(section_folder_path)))
                raise AssertionError
            import_tilespecs.parse_section_singlebeam_save(section_folder_path, tilespecs_folder_path)
    elif overall_args["base"]["EM_type"] == 'multibeam':
        for section_folder_path in tqdm(section_folder_paths):
            if not utils.check_EM_type(section_folder_path) == 'multibeam':
                log_controller.error('The EM type you declared in the overall_args.json(multibeam) is different '
                                     'from the parsing result({}).'.format(utils.check_EM_type(section_folder_path)))
                raise AssertionError
            import_tilespecs.parse_section_multibeam_save(section_folder_path, tilespecs_folder_path)
    else:
        log_controller.error('Wrong EM type in overall_args.json. It should be singlebeam or multibeam.')
        raise AssertionError

    # Step 3: Stitching
    print(utils.to_red('Step 3 -- Stitch the data according to the tilespecs and output new tilespecs.'))
