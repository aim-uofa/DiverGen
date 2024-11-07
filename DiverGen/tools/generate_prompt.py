import os
import json

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_json_path', type=str, default='datasets/metadata/lvis_v1_train_cat_info.json')
    parser.add_argument('--out_txt_dir', type=str, default='input/lvis_prompt')
    args = parser.parse_args()

    if not os.path.exists(args.out_txt_dir):
        print('>>> create {}'.format(args.out_txt_dir))
        os.makedirs(args.out_txt_dir)

    print('>>> read json from {}'.format(args.in_json_path))
    with open(args.in_json_path, 'r') as f:
        categories = json.load(f)
    

    for category in categories:
        category_id = category['id']
        print('>>> process {}'.format(category_id))
        with open(os.path.join(args.out_txt_dir, '{}.txt'.format(category_id)), 'w') as f:
            f.write('a photo of a single {}, {}, in a white background\n'.format(category['name'].replace('_', ' '), category['def']))
    