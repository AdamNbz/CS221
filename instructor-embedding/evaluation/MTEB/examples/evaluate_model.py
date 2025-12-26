import os
import sys
import logging
import argparse
import torch
from mteb import MTEB
from InstructorEmbedding import INSTRUCTOR
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None,type=str)
    parser.add_argument('--output_dir', default=None,type=str)
    parser.add_argument('--task_name', default=None,type=str)
    parser.add_argument('--cache_dir', default=None,type=str)
    parser.add_argument('--result_file', default=None,type=str)
    parser.add_argument('--prompt', default=None,type=str)
    parser.add_argument('--split', default='test',type=str)
    parser.add_argument('--batch_size', default=32,type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',type=str)
    args = parser.parse_args()
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Using device: {args.device}")
    else:
        print("CUDA not available, using CPU")

    if not args.result_file.endswith('.txt') and not os.path.isdir(args.result_file):
        os.makedirs(args.result_file,exist_ok=True)

    print(f"Loading model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    model = INSTRUCTOR(args.model_name,cache_folder=args.cache_dir,device=args.device)
    print("Model loaded successfully!")
    
    print(f"Starting evaluation on task: {args.task_name}")
    evaluation = MTEB(tasks=[args.task_name],task_langs=["en"])
    evaluation.run(model, output_folder=args.output_dir, eval_splits=[args.split],args=args,)

    print("--DONE--")
