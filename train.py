import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from process_data import Dataset
from loss import Reinforce_Loss, CrossEntropySG_Loss

from Mapper.mapper import TokenMapper
from Encoder.text_encoder import GPT2
from Encoder.dalle_encoder import DalleEncoder



def main():
    parser = get_config()
    args = parser.parse_args()
    device = args.device
    batch_size = args.batch
    
    print("===============================================================")
    print(f"          MAPPING FROM {args.image_encoder} ({args.exp_type}) TO {args.llm}")
    print("===============================================================")
    
    experiment_name = f"{args.exp_type}/{args.algo}/model={args.llm}_lr={args.lr}"
    
    writer = SummaryWriter(f'runs/{experiment_name}')
    
    print(f"loading {args.llm} model ...")
    if "gpt2" in args.llm:
        llm = GPT2(args)
    
    print("finish loading model\n")
    
    if args.image_encoder == "dalle":
        image_encoder = DalleEncoder(args)
    
    mapper = TokenMapper(image_encoder.codebook_len, llm.codebook_len, args)
    
    print("\nloading dataset......")
    dataset = Dataset(args, image_encoder)
    dataloader = DataLoader(dataset.dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"Dataset Length: {len(dataloader)}")
    print("finish loading dataset\n")
    
    if args.algo == 'rl':
        print("USING ALGORITHM: REINFORCE\n")
    elif args.algo == 'base':
        print("USING ALGORITHM: SUPERVISED\n")
        
    
    print(f"training {args.epoch} epoch(s) with learning rate={args.lr}\n")
    
    optimizer = optim.Adam(mapper.model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)
            
    writer.add_hparams(
            {
                "llm_model": args.llm,
                "modality": args.exp_type,
                "encoder": args.image_encoder,
                "learning_rate": args.lr,
            },
            {}
    )

    # def train_on_lsun(dataloader, epochs=2):
    for epoch in range(args.epoch):
        for i, (images, _) in enumerate(dataloader):
            
            optimizer.zero_grad()

            # for vqgan
            # images = images.permute(0, 2, 3, 1)
            
            # Process each image through DALL-E encoder to get image tokens
            one_hot_image_tokens = image_encoder.encode(images)
    
            flattened_tokens = one_hot_image_tokens.reshape(one_hot_image_tokens.size(0), -1, image_encoder.codebook_len)

            # Map tokens and get ground truth from LLM
            mapped_feature_vector = mapper.model(flattened_tokens)
            ground_truth_tokens = llm.get_ground_truth(mapped_feature_vector).to(device)
            # Calculate Base Loss
            ground_truth_tokens = ground_truth_tokens.reshape(-1)
            # loss = CELoss_SG(mapped_feature_vector, ground_truth_tokens)
            
            # RL Loss
            rl_loss, prediction_logits = CrossEntropySG_Loss(llm, mapped_feature_vector, ground_truth_tokens, reduction='none')
            prediction_logits = prediction_logits.reshape(batch_size, -1, llm.vocab_len)
            ground_truth_tokens = ground_truth_tokens.reshape(batch_size, -1)
            rl_loss = rl_loss.reshape(batch_size, -1)

            loss = Reinforce_Loss(prediction_logits, ground_truth_tokens, rl_loss, gamma=args.gamma)
            
            # Backward pass and update
            loss.backward()
            optimizer.step()

            # Log the losses
            writer.add_scalars(
                "Training Metrics",
                {
                    "loss": loss.item(),
                    "cross_entropy": rl_loss[:,0].mean().item(),
                },
                epoch * len(dataloader) + i
            )
                
            if i % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}")

        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epoch} completed.")
        
    Path(f"models/{exp_type}/{experiment}").mkdir(parents=True, exist_ok=True)
    torch.save(mapper.state_dict(), f"models/{experiment_name}")
    writer.close()


if __name__=="__main__":
    main()
    
    