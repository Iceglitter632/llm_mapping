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
    
    writer = SummaryWriter(log_dir = f'runs/{experiment_name}')
    
    print(f"loading {args.llm} model ...")
    if "gpt2" in args.llm:
        llm = GPT2(args)
    
    print("finish loading model\n")
    
    if args.image_encoder == "dalle":
        image_encoder = DalleEncoder(args)
    
    mapper = TokenMapper(image_encoder.vocab_len, llm.feature_dim, args)
    
    print("\nloading dataset......")
    dataset = Dataset(args, image_encoder)
    trainloader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"Train Dataset Length: {len(trainloader)}")
    print(f"Validation Dataset Length: {len(valloader)}")
    print("finish loading dataset\n")
    
    if args.algo == 'rl':
        print("USING ALGORITHM: REINFORCE\n")
    elif args.algo == 'base':
        print("USING ALGORITHM: SUPERVISED\n")
        
    
    print(f"training {args.epoch} epoch(s) with learning rate={args.lr}\n")
    
    optimizer = optim.Adam(mapper.model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)
    criterion = nn.CrossEntropyLoss()
    rl_criterion = nn.CrossEntropyLoss(reduction='none')
            
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
        
        mapper.train()
        for i, (images, _) in enumerate(trainloader):
            
            optimizer.zero_grad()

            # for vqgan
            # images = images.permute(0, 2, 3, 1)
            
            # Process each image through DALL-E encoder to get image tokens
            image_token_logits = image_encoder.encode(images)
            ground_truth_tokens = torch.argmax(image_token_logits, dim=1)
            one_hot_image_tokens = F.one_hot(ground_truth_tokens, num_classes=image_encoder.vocab_len).permute(0,3,1,2).float()
    
            ground_truth_tokens = ground_truth_tokens.reshape(-1)
            flattened_tokens = one_hot_image_tokens.reshape(one_hot_image_tokens.size(0), -1, image_encoder.vocab_len)

            # Map tokens and get ground truth from LLM
            mapped_feature_vector = mapper.model(flattened_tokens)
            translated_text_tokens = image_encoder.translate(mapped_feature_vector, llm.embeddings)

            final_layer_fv = llm.generate_next_token_predictions(translated_text_tokens)

            action_logits = torch.matmul(final_layer_fv, mapper.model.weight)
            _logits = action_logits.reshape(-1, image_encoder.vocab_len)


            # Calculate Base Loss
            # loss = criterion(_logits, ground_truth_tokens)
            
            # RL Loss
            ce_loss = rl_criterion(_logits, ground_truth_tokens)
            ground_truth_tokens = ground_truth_tokens.reshape(batch_size, -1)
            ce_loss = ce_loss.reshape(batch_size, -1)

            loss = Reinforce_Loss(action_logits, ground_truth_tokens, ce_loss, gamma=args.gamma)
            
            # Backward pass and update
            loss.backward()
            optimizer.step()

            # Log the losses
            if 'rl' in args.algo:
                writer.add_scalars(
                    "Training Metrics",
                    {
                        "loss": loss.item(),
                        "cross_entropy": ce_loss.mean().item(),
                    },
                    epoch * len(trainloader) + i
                )
            elif 'base' in args.algo:
                writer.add_scalar("Training/cross_entropy", loss.item(), epoch*len(trainloader)+i)
                
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}")

        
        if args.val:
            mapper.eval()
            for i, (images, _) in enumerate(valloader):

                # for vqgan
                # images = images.permute(0, 2, 3, 1)
                
                # Process each image through DALL-E encoder to get image tokens
                image_token_logits = image_encoder.encode(images)
                ground_truth_tokens = torch.argmax(image_token_logits, dim=1)
                one_hot_image_tokens = F.one_hot(ground_truth_tokens, num_classes=image_encoder.vocab_len).permute(0,3,1,2).float()
        
                ground_truth_tokens = ground_truth_tokens.reshape(-1)
                flattened_tokens = one_hot_image_tokens.reshape(one_hot_image_tokens.size(0), -1, image_encoder.vocab_len)

                # Map tokens and get ground truth from LLM
                mapped_feature_vector = mapper.model(flattened_tokens)
                translated_text_tokens = image_encoder.translate(mapped_feature_vector, llm.embeddings)

                final_layer_fv = llm.generate_next_token_predictions(translated_text_tokens)

                action_logits = torch.matmul(final_layer_fv, mapper.model.weight)
                _logits = action_logits.reshape(-1, image_encoder.vocab_len)


                # Calculate Base Loss
                # loss = criterion(_logits, ground_truth_tokens)
                
                ce_loss = rl_criterion(_logits, ground_truth_tokens)
                prediction_logits = prediction_logits.reshape(batch_size, -1, llm.vocab_len)
                ground_truth_tokens = ground_truth_tokens.reshape(batch_size, -1)
                ce_loss = ce_loss.reshape(batch_size, -1)

                loss = Reinforce_Loss(action_logits, ground_truth_tokens, ce_loss, gamma=args.gamma)
                
                if 'rl' in args.algo:
                    writer.add_scalars(
                        "Validation Metrics",
                        {
                            "loss": loss.item(),
                            "cross_entropy": ce_loss.mean().item(),
                        },
                        epoch * len(valloader) + i
                    )
                
                elif 'base' in args.algo:
                    writer.add_scalar("Validation/cross_entropy", loss.item(), epoch*len(valloader)+i)

        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epoch} completed.")
        
    Path(f"models/{experiment_name}").mkdir(parents=True, exist_ok=True)
    torch.save(mapper.state_dict(), f"models/{experiment_name}")
    writer.close()


if __name__=="__main__":
    main()
    
    