import sys
import os
import time
import math
from tqdm import tqdm
import shutil
import itertools
import multiprocessing as mp
from common import utils, import_tilespecs, bounding_box
from stitching import create_features, match_features, optimize_2d
from alignment import pre_match, block_match, optimize_3d, normalize_coordinates
from renderer import render_driver


def check_overall_args(args):
    assert args["base"]["running_mode"] in ["debug", "release"]
    assert args["base"]["EM_type"] in ["singlebeam", "multibeam"]
    if args["base"]["running_mode"] == 'debug':
        print(utils.to_green('[debug mode] multiprocessing will be disabled.'))
    elif args["base"]["running_mode"] == 'release':
        print(utils.to_green('[release mode] multiprocessing will be enabled.'))


if __name__ == '__main__':
    # Step 0: Notification of the argument settings.
    print(utils.to_red('Step 0 -- All the running arguments are stored in the folder "arguments". '
                       'Have you confirmed that all the arguments are set as you want?'))
    ans = input('Please enter yes or no: ')
    if ans == 'yes':
        print(utils.to_green('Arguments confirmed, start.'))
    elif ans == 'no':
        sys.exit('Please double check the arguments in the "arguments" folder and run again.')
    else:
        sys.exit('Please enter yes or no.')
    overall_args = utils.load_json_file('arguments/overall_args.json')
    check_overall_args(overall_args)
    utils.create_dir(overall_args["base"]["workspace"])
    workspace = os.path.abspath(overall_args["base"]["workspace"])
    shutil.copy('arguments/overall_args.json', os.path.join(workspace, 'overall_args.json'))
    log_controller = utils.LogController('main', 'run', os.path.join(workspace, 'log'))

    # Step 1: Check the data and generate tilespecs files.
    print(utils.to_red('Step 1 -- Check the data and generate tilespecs files.'))
    time.sleep(0.1)
    tilespecs_folder_path = os.path.join(workspace, 'tilespecs')
    utils.create_dir(tilespecs_folder_path)
    section_folder_paths = utils.ls_sub_folder_paths(overall_args["base"]["sample_folder"])
    included_layers = []
    skipped_layers = utils.parse_layer_range(overall_args["base"]["skipped_layers"])
    if overall_args["base"]["EM_type"] == 'singlebeam':
        for section_folder_path in tqdm(section_folder_paths):
            if not utils.check_EM_type(section_folder_path) == 'singlebeam':
                log_controller.error('The EM type you declared in the overall_args.json(singlebeam) is different '
                                     'from the parsing result({}).'.format(utils.check_EM_type(section_folder_path)))
                raise AssertionError
            cur_layer = import_tilespecs.parse_section_singlebeam_save(section_folder_path, tilespecs_folder_path)
            if cur_layer not in skipped_layers:
                included_layers.append(cur_layer)
    elif overall_args["base"]["EM_type"] == 'multibeam':
        for section_folder_path in tqdm(section_folder_paths):
            if not utils.check_EM_type(section_folder_path) == 'multibeam':
                log_controller.error('The EM type you declared in the overall_args.json(multibeam) is different '
                                     'from the parsing result({}).'.format(utils.check_EM_type(section_folder_path)))
                raise AssertionError
            cur_layer = import_tilespecs.parse_section_multibeam_save(section_folder_path, tilespecs_folder_path)
            if cur_layer not in skipped_layers:
                included_layers.append(cur_layer)
    print(utils.to_green('Exclude the skipped layers and get {} layers in total.'.format(len(included_layers))))
    layers_data = {}

    # Step 2: Stitching
    print(utils.to_red('Step 2 -- Stitch the data according to the tilespecs and output new tilespecs.'))
    time.sleep(0.1)
    stitch_workspace = os.path.join(workspace, '2d')
    utils.create_dir(stitch_workspace)
    features_folder_path = os.path.join(stitch_workspace, 'features')
    utils.create_dir(features_folder_path)
    match_folder_path = os.path.join(stitch_workspace, 'matched_features')
    utils.create_dir(match_folder_path)
    optimized_2d_folder_path = os.path.join(stitch_workspace, 'optimized_2d')
    utils.create_dir(optimized_2d_folder_path)
    tilespecs_json_files = utils.ls_absolute_paths(tilespecs_folder_path)
    for layer in tqdm(included_layers):
        layers_data[str(layer)] = {}
        # Step 2.1: Preparation
        tilespecs_json_file = os.path.join(tilespecs_folder_path, 'Sec_{}.json'.format(str(layer).zfill(4)))
        layers_data[str(layer)]["tilespecs_json_file_path"] = tilespecs_json_file
        log_controller.debug('Start stitching section {}/{}'.format(layer, len(included_layers)))

        # Skip the section if already have the optimized result
        layer_basename_no_ext = os.path.basename(tilespecs_json_file).replace('.json', '')
        opt_montage_json = os.path.join(optimized_2d_folder_path, "{0}_montaged.json".format(layer_basename_no_ext))
        layers_data[str(layer)]["stitched_tilespecs_json_file_path"] = opt_montage_json

        # Step 2.2: Creating features
        log_controller.debug('Start extracting features')
        section_features_folder_path = os.path.join(features_folder_path, 'Sec_{}'.format(str(layer).zfill(4)))
        utils.create_dir(section_features_folder_path)
        layers_data[str(layer)]["features_folder_path"] = section_features_folder_path
        tilespecs = utils.load_json_file(tilespecs_json_file)
        layers_data[str(layer)]["all_mfovs"] = set([tilespec["mfov"] for tilespec in tilespecs])
        features_args = utils.load_json_file('arguments/features_args.json')

        if os.path.exists(opt_montage_json):
            log_controller.debug('Previously optimized layer: {0}, skipping all pre-computations'.format(layer))
            continue

        if overall_args["base"]["running_mode"] == 'debug':
            for i, tilespec in enumerate(tilespecs):
                log_controller.debug('Extracting {}/{}'.format(i + 1, len(tilespecs)))
                create_features.compute_and_save_tile_features(tilespec, section_features_folder_path,
                                                               overall_args["stitching"]["features_type"],
                                                               features_args)
        elif overall_args["base"]["running_mode"] == 'release':
            pool_extract = mp.Pool(overall_args["multiprocess"]["extract_features"])
            for tilespec in tilespecs:
                pool_extract.apply_async(create_features.compute_and_save_tile_features,
                                         (tilespec, section_features_folder_path,
                                          overall_args["stitching"]["features_type"], features_args))
            pool_extract.close()
            pool_extract.join()

        # Step 2.3: Matching features
        log_controller.debug('Start matching features')
        section_match_folder_path = os.path.join(match_folder_path, 'Sec_{}'.format(str(layer).zfill(4)))
        utils.create_dir(section_match_folder_path)
        intra_match_folder_path = os.path.join(section_match_folder_path, 'intra')
        utils.create_dir(intra_match_folder_path)
        inter_match_folder_path = os.path.join(section_match_folder_path, 'inter')
        utils.create_dir(inter_match_folder_path)

        # Collect matching arguments
        match_file_paths_list = []
        match_features_params = []
        log_controller.debug('Looking for the overlapped tiles and collecting arguments for matching function.')
        for pair in itertools.combinations(range(len(tilespecs)), 2):
            idx1 = pair[0]
            idx2 = pair[1]
            ts1 = tilespecs[idx1]
            ts2 = tilespecs[idx2]
            bbox1 = bounding_box.BoundingBox.fromList(ts1["bbox"])
            bbox2 = bounding_box.BoundingBox.fromList(ts2["bbox"])
            if bbox1.overlap(bbox2):
                img_file_path1 = ts1["mipmapLevels"]["0"]["imageUrl"]
                img_file_path2 = ts2["mipmapLevels"]["0"]["imageUrl"]
                img_basename_no_ext1 = os.path.basename(img_file_path1).split('.')[0]
                img_basename_no_ext2 = os.path.basename(img_file_path2).split('.')[0]
                feature_path1 = os.path.join(os.path.join(section_features_folder_path, str(ts1["mfov"]).zfill(6)),
                                             '{}_{}.h5py'.format(overall_args["stitching"]["features_type"],
                                                                 img_basename_no_ext1))
                feature_path2 = os.path.join(os.path.join(section_features_folder_path, str(ts2["mfov"]).zfill(6)),
                                             '{}_{}.h5py'.format(overall_args["stitching"]["features_type"],
                                                                 img_basename_no_ext2))
                if ts1["mfov"] == ts2["mfov"]:
                    # Intra mfov
                    cur_match_folder_path = os.path.join(intra_match_folder_path, str(ts1["mfov"]))
                    utils.create_dir(cur_match_folder_path)
                else:
                    # Inter mfov
                    cur_match_folder_path = inter_match_folder_path
                match_path = os.path.join(cur_match_folder_path,
                                          "{0}_matches_{1}_{2}.json".format(overall_args["stitching"]["features_type"],
                                                                            img_basename_no_ext1, img_basename_no_ext2))
                # Add to lists
                match_file_paths_list.append(match_path)
                if not os.path.exists(match_path):
                    match_features_params.append({'feature_path1': feature_path1,
                                                  'feature_path2': feature_path2,
                                                  'tilespecs1': ts1,
                                                  'tilespecs2': ts2,
                                                  'match_path': match_path})
        matching_pairs_num = len(match_features_params)
        log_controller.debug('Found {} pairs to be matched.'.format(matching_pairs_num))

        # Start matching function
        if overall_args["base"]["running_mode"] == 'debug':
            for i in range(matching_pairs_num):
                log_controller.debug('Matching {}/{}'.format(i + 1, matching_pairs_num))
                match_features.match_and_save(match_features_params[i]['feature_path1'],
                                              match_features_params[i]['feature_path2'],
                                              match_features_params[i]['tilespecs1'],
                                              match_features_params[i]['tilespecs2'],
                                              match_features_params[i]['match_path'],
                                              features_args, overall_args["stitching"]["matching_type"])
        elif overall_args["base"]["running_mode"] == 'release':
            pool_extract = mp.Pool(overall_args["multiprocess"]["match_features"])
            for i in range(matching_pairs_num):
                pool_extract.apply_async(match_features.match_and_save,
                                         (match_features_params[i]['feature_path1'],
                                          match_features_params[i]['feature_path2'],
                                          match_features_params[i]['tilespecs1'],
                                          match_features_params[i]['tilespecs2'],
                                          match_features_params[i]['match_path'],
                                          features_args, overall_args["stitching"]["matching_type"]))
            pool_extract.close()
            pool_extract.join()

        # Step 2.4 Optimization
        log_controller.debug('Start optimizing for layer {}'.format(layer))
        optimize_2d.optimize_2d_stitching(tilespecs_json_file, match_file_paths_list, opt_montage_json)
    cur_tilespecs_folder_path = optimized_2d_folder_path

    # Step 3: Alignment
    print(utils.to_red('Step 3 -- Align the data according to the stitched tilespecs and output new tilespecs.'))
    time.sleep(0.1)
    align_args = utils.load_json_file('arguments/align_args.json')
    align_workspace = os.path.join(workspace, '3d')
    utils.create_dir(stitch_workspace)
    pre_matches_dir = os.path.join(align_workspace, "pre_matches")
    utils.create_dir(pre_matches_dir)
    matched_pmcc_dir = os.path.join(align_workspace, "matched_pmcc")
    utils.create_dir(matched_pmcc_dir)
    post_optimization_dir = os.path.join(align_workspace, "post_optimization")
    utils.create_dir(post_optimization_dir)
    stitched_tilespecs_json_files = utils.ls_absolute_paths(optimized_2d_folder_path)
    all_pmcc_file_paths = []
    all_ts_file_paths = []
    for layer1 in tqdm(included_layers):
        log_controller.debug('Start aligning section {}/{}'.format(layer1, len(included_layers)))
        all_ts_file_paths.append(layers_data[str(layer1)]["stitched_tilespecs_json_file_path"])
        matched_after_layers = 0
        j = 1
        layers_data[str(layer1)]['pre_matched_mfovs'] = {}
        while matched_after_layers < overall_args["alignment"]["max_layer_distance"]:
            if layer1 + j > included_layers[-1]:
                break
            layer2 = layer1 + j
            if layer2 in skipped_layers:
                log_controller.debug("Skipping matching of layers {} and {}, "
                                     "because {} should be skipped".format(layer1, layer2, layer2))
                j += 1
                continue

            layer1_basename_no_ext = 'Sec_{}'.format(str(layer1).zfill(4))
            layer2_basename_no_ext = 'Sec_{}'.format(str(layer2).zfill(4))

            # Step 3.1 Pre-match
            pre_match_file_path = os.path.join(pre_matches_dir, '{}_{}_pre_matches.json'.format(layer1_basename_no_ext,
                                                                                                layer2_basename_no_ext))
            layers_data[str(layer1)]['pre_matched_mfovs'][str(layer2)] = pre_match_file_path
            if not os.path.exists(pre_match_file_path):
                log_controller.debug("Pre-Matching layers: {} and {}".format(layer1, layer2))
                pre_match.pre_match_layers(layers_data[str(layer1)]['stitched_tilespecs_json_file_path'],
                                           layers_data[str(layer1)]['features_folder_path'],
                                           layers_data[str(layer2)]['stitched_tilespecs_json_file_path'],
                                           layers_data[str(layer2)]['features_folder_path'],
                                           pre_match_file_path, align_args)

            # Step 3.2 PMCC Match
            match_pmcc_params = []
            # Collect pmcc match parameters
            for mfov1 in layers_data[str(layer1)]["all_mfovs"]:
                mfov_pmcc_file_path1 = os.path.join(matched_pmcc_dir,
                                                    "{}_{}_match_pmcc_mfov_{}.json".format(layer1_basename_no_ext,
                                                                                           layer2_basename_no_ext,
                                                                                           mfov1))
                all_pmcc_file_paths.append(mfov_pmcc_file_path1)
                if not os.path.exists(mfov_pmcc_file_path1):
                    match_pmcc_params.append({'ts1': layers_data[str(layer1)]['stitched_tilespecs_json_file_path'],
                                              'ts2': layers_data[str(layer2)]['stitched_tilespecs_json_file_path'],
                                              'pre': layers_data[str(layer1)]['pre_matched_mfovs'][str(layer2)],
                                              'pmcc_file_path': mfov_pmcc_file_path1,
                                              'mfov': int(mfov1)})
            for mfov2 in layers_data[str(layer2)]["all_mfovs"]:
                mfov_pmcc_file_path2 = os.path.join(matched_pmcc_dir,
                                                    "{}_{}_match_pmcc_mfov_{}.json".format(layer2_basename_no_ext,
                                                                                           layer1_basename_no_ext,
                                                                                           mfov2))
                all_pmcc_file_paths.append(mfov_pmcc_file_path2)
                if not os.path.exists(mfov_pmcc_file_path2):
                    match_pmcc_params.append({'ts1': layers_data[str(layer2)]['stitched_tilespecs_json_file_path'],
                                              'ts2': layers_data[str(layer1)]['stitched_tilespecs_json_file_path'],
                                              'pre': layers_data[str(layer1)]['pre_matched_mfovs'][str(layer2)],
                                              'pmcc_file_path': mfov_pmcc_file_path2,
                                              'mfov': int(mfov2)})
            if overall_args["base"]["running_mode"] == 'debug':
                for i in range(len(match_pmcc_params)):
                    block_match.match_layers_pmcc_matching(match_pmcc_params[i]['ts1'],
                                                           match_pmcc_params[i]['ts2'],
                                                           match_pmcc_params[i]['pre'],
                                                           match_pmcc_params[i]['pmcc_file_path'],
                                                           match_pmcc_params[i]['mfov'], align_args)
            elif overall_args["base"]["running_mode"] == 'release':
                pool_extract = mp.Pool(overall_args["multiprocess"]["pmcc_match"])
                for i in range(len(match_pmcc_params)):
                    pool_extract.apply_async(block_match.match_layers_pmcc_matching,
                                             (match_pmcc_params[i]['ts1'],
                                              match_pmcc_params[i]['ts2'],
                                              match_pmcc_params[i]['pre'],
                                              match_pmcc_params[i]['pmcc_file_path'],
                                              match_pmcc_params[i]['mfov'], align_args))
                pool_extract.close()
                pool_extract.join()
            j += 1
            matched_after_layers += 1

    ts_list_file = os.path.join(align_workspace, "all_ts_files.txt")
    utils.write_list_to_file(ts_list_file, all_ts_file_paths)
    pmcc_list_file = os.path.join(align_workspace, "all_pmcc_files.txt")
    utils.write_list_to_file(pmcc_list_file, all_pmcc_file_paths)

    # Step 3.3 Optimize all layers to a single 3d image
    sections_opt_outputs = []
    for layer in included_layers:
        postfix = os.path.basename(layers_data[str(layer)]['stitched_tilespecs_json_file_path'])
        out_section = os.path.join(post_optimization_dir, '{}_{}'.format(str(layer).zfill(4), postfix))
        sections_opt_outputs.append(out_section)

    print('Start optimizing 3d...')
    output_folder_path = os.path.join(workspace, "final_tilespecs")
    utils.create_dir(output_folder_path)
    existing_aligned_tilespecs_files = utils.ls_absolute_paths(output_folder_path)
    aligned_flag = True
    for layer in included_layers:
        aligned_tilespecs_file = os.path.join(output_folder_path, 'Sec_{}_aligned.json'.format(str(layer).zfill(4)))
        if not aligned_tilespecs_file in existing_aligned_tilespecs_files:
            aligned_flag = False
            break
    if not aligned_flag:
        optimize_3d.optimize_layers_elastic([ts_list_file], [pmcc_list_file], post_optimization_dir, align_args)
        sections_outputs = []
        for section_opt_output in sections_opt_outputs:
            out_section = os.path.join(output_folder_path, os.path.basename(section_opt_output))
            sections_outputs.append(out_section)
        # Normalize the output files to the (0, 0) coordinates
        normalize_coordinates.normalize_coordinates(post_optimization_dir, output_folder_path)
    cur_tilespecs_folder_path = output_folder_path

    # Step 4: Render the tilespecs to images.
    print(utils.to_red('Step 4 -- Render the tilespecs to images.'))
    mip_level = overall_args["renderer"]["mip"]
    scale = 1 / (2**mip_level)
    render_folder_path = os.path.join(workspace, 'rendered')
    utils.create_dir(render_folder_path)
    mip_folder_path = os.path.join(render_folder_path, 'mip' + str(mip_level))
    utils.create_dir(mip_folder_path)

    tilespecs_paths_list = utils.ls_absolute_paths(cur_tilespecs_folder_path)

    entire_image_bbox = bounding_box.BoundingBox.read_bbox_from_ts(utils.load_json_file(tilespecs_paths_list[0]))
    for tilespecs_path in tilespecs_paths_list[1:]:
        entire_image_bbox.extend(bounding_box.BoundingBox.read_bbox_from_ts(utils.load_json_file(tilespecs_path)))
    log_controller.debug("Final bbox for the 3d image: {}".format(entire_image_bbox))

    from_x = overall_args["renderer"]["from_x"]
    from_y = overall_args["renderer"]["from_y"]
    to_x = overall_args["renderer"]["to_x"]
    to_y = overall_args["renderer"]["to_y"]
    if from_x == 0:
        from_x = int(math.floor(entire_image_bbox[0]))
    if from_y == 0:
        from_y = int(math.floor(entire_image_bbox[2]))
    if to_x == -1:
        to_x = int(math.ceil(entire_image_bbox[1]))
    if to_y == -1:
        to_y = int(math.ceil(entire_image_bbox[3]))
    in_bbox = [from_x, to_x, from_y, to_y]

    render_params = []
    # Collect rendering parameters
    for i in range(len(tilespecs_paths_list)):
        layer = int(os.path.basename(tilespecs_paths_list[i]).split('_')[1])
        render_params.append({
            "tilespecs_file_path": tilespecs_paths_list[i],
            "layer": layer,
            "save_folder_path": os.path.join(mip_folder_path, 'Sec_{}'.format(str(layer).zfill(4)))
        })

    if overall_args["base"]["running_mode"] == 'debug':
        for i in tqdm(range(len(render_params))):
            render_driver.render_tilespec(render_params[i]["tilespecs_file_path"],
                                          render_params[i]["save_folder_path"],
                                          render_params[i]["layer"],
                                          scale,
                                          overall_args["renderer"]["file_type"],
                                          in_bbox,
                                          overall_args["renderer"]["tile_size"],
                                          overall_args["renderer"]["invert_image"],
                                          overall_args["renderer"]["blend_type"])
    elif overall_args["base"]["running_mode"] == 'release':
        pool_render = mp.Pool(overall_args["multiprocess"]["render"])
        for i in range(len(render_params)):
            pool_render.apply_async(render_driver.render_tilespec, (render_params[i]["tilespecs_file_path"],
                                                                    render_params[i]["save_folder_path"],
                                                                    render_params[i]["layer"],
                                                                    scale,
                                                                    overall_args["renderer"]["file_type"],
                                                                    in_bbox,
                                                                    overall_args["renderer"]["tile_size"],
                                                                    overall_args["renderer"]["invert_image"],
                                                                    overall_args["renderer"]["blend_type"]))
        pool_render.close()
        pool_render.join()
print(utils.to_red('Finished.'))
