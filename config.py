import argparse
import pathlib

def get_config():

    parser = argparse.ArgumentParser(
        prog='VQVAE MAPPING',
        description='This program maps a chosen modality [image, video, pose, music] to text, utilizing the in-context learning of a LLM to compress sequences',
    )

    # GPU
    parser.add_argument("--device", default="cuda:0")

    # Experiment Settings
    parser.add_argument("--algo", default="rl", help="the algorithm used for training")
    parser.add_argument("--image_encoder", default="dalle")
    parser.add_argument("--exp_type", default="image", help="which modality is used. [Image, Music, Pose, Video]")
    parser.add_argument("--llm", default="gpt2", help="The LLM we are mapping to")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--image_size", default=128, type=int, help="The size of image that encoder takes as input")

    # Dataset
    parser.add_argument("--dataset_path", default="./data/lsun", type=pathlib.Path)
    parser.add_argument('--dataset_classes', nargs='+', help='List of LSUN datasets', default=['living_room_train'])

    # HyperParameters
    parser.add_argument("--batch", default=20, type=int)
    parser.add_argument("--lr", default=5e-5, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--gamma", default=0.95, type=int, help="discount factor for REINFORCE")
    parser.add_argument("--decay", default=0.9, type=int, help="how learning rate decays for each epoch")

    return parser