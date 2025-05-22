import argparse


def get_mage_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='2019')
    # parser.add_argument('--years', type=str, default='2019')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=3028)

    parser.add_argument('--bs', type=int, default=64)
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--recur_num', type=int, default=16)
    parser.add_argument('--feature_dim', type=int, default=1)
    parser.add_argument('--node_dim', type=int, default=32)
    parser.add_argument('--blocknum', type=int, default=3)
    parser.add_argument('--head', type=int, default=16)
    parser.add_argument('--topk', type=int, default=4)
    

    parser.add_argument('--backbone', type=str, default='MAGE')

    parser.add_argument('--outer', type=int, default=1)

    parser.add_argument('--bshare', type=int, default=0)
    parser.add_argument('--lshare', type=int, default=0)

    

    parser.add_argument('--second', type=int, default=0)
    parser.add_argument('--minute', type=int, default=288)
    parser.add_argument('--hour', type=int, default=0)
    parser.add_argument('--day', type=int, default=7)
    parser.add_argument('--week', type=int, default=0)
    parser.add_argument('--weekday', type=int, default=0)
    parser.add_argument('--month', type=int, default=0)
    parser.add_argument('--quarter', type=int, default=0)
    parser.add_argument('--year', type=int, default=0)


    


    parser.add_argument('--ct', type=int, default=0) # continue learning
    return parser

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')