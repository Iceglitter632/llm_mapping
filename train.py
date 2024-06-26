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
from Encoder.vqgan_encoder import VQGanEncoder



def main(args):
    device = args.device
    batch_size = args.batch
    
    print("===============================================================")
    print(f"          MAPPING FROM {args.image_encoder} ({args.exp_type}) TO {args.llm}")
    print("===============================================================")
    
    print(args.name)
    
    if args.name != "none":
        experiment_name = f"{args.exp_type}/{args.algo}/{args.name}/model={args.llm}_lr={args.lr}"
    else:
        experiment_name = f"{args.exp_type}/{args.algo}/model={args.llm}_lr={args.lr}"
    
    writer = SummaryWriter(log_dir = f'runs/{experiment_name}')
    
    print(f"loading {args.llm} model ...")
    if "gpt2" in args.llm:
        llm = GPT2(args)
    
    print("finish loading model\n")
    
    if args.image_encoder == "dalle":
        image_encoder = DalleEncoder(args)
    elif args.image_encoder == "vqgan":
        image_encoder = VQGanEncoder(args)
    
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
    ce_criterion = nn.CrossEntropyLoss()
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
    
    '''
    為了加速寫的 之後可以拿掉
    '''
    seq_len = 255
    discount_matrix = torch.zeros((seq_len, seq_len)).to(device)
    # Fill the matrix according to the given pattern
    for i in range(seq_len):
        for j in range(i, seq_len):
            discount_matrix[i, j] = args.gamma ** (j - i)
    
    normalize_factor = discount_matrix.sum(dim=1)

    '''
    為了加速寫的 之後可以拿掉
    '''

    for epoch in range(args.epoch):
        
        mapper.train()
        for i, (images, _) in enumerate(trainloader):
            
            optimizer.zero_grad()
            
            # Process each image through DALL-E encoder to get image tokens
            ground_truth_tokens = image_encoder.get_ground_truth(images)
            one_hot_image_tokens = image_encoder.get_onehot(ground_truth_tokens)
    
            ground_truth_tokens = ground_truth_tokens[:,1:].reshape(-1)
            one_hot_image_tokens = one_hot_image_tokens.reshape(one_hot_image_tokens.size(0), -1, image_encoder.vocab_len)

            # Map tokens and get ground truth from LLM
            mapped_feature_vector = mapper.model(one_hot_image_tokens)
            translated_feature_vector, translated_logits, translated_text_tokens = image_encoder.translate(
                mapped_feature_vector, llm.embeddings, temperature=args.temperature
            )

            final_layer_fv = llm.generate_next_token_predictions_withfv(translated_text_tokens)

            # final_layer_fv = F.normalize(final_layer_fv, dim=-1)
            # mapper_embeds = F.normalize(mapper.model.weight, dim=0)
            mapper_embeds = mapper.model.weight
            logits = torch.matmul(final_layer_fv, mapper_embeds)
            _logits = logits[:,:-1].reshape(-1, image_encoder.vocab_len)
            
            ce_loss = ce_criterion(_logits, ground_truth_tokens)
            ce_loss.backward()
            optimizer.step()

            if 'base' in args.algo:
                writer.add_scalar("Training/cross_entropy", loss.item(), epoch*len(trainloader)+i)
            
            elif 'rl' in args.algo:
                optimizer.zero_grad()
                mapped_feature_vector = mapper.model(flattened_tokens)
                # action_logits = torch.matmul(mapped_feature_vector, llm.embeddings.T)
                translated_feature_vector, translated_logits, translated_text_tokens = image_encoder.translate(
                    mapped_feature_vector, llm.embeddings.detach(), temperature=args.temperature
                )
                cloned_mapper_weight = mapper.model.weight.clone().detach()
                
                with torch.no_grad():
                    final_layer_fv = generate_next_token_predictions_withfv(translated_feature_vector)
                    final_layer_fv = F.normalize(final_layer_fv, dim=-1)
                    cloned_mapper_weight = F.normalize(cloned_mapper_weight, dim=0)
                    logits = torch.matmul(final_layer_fv, cloned_mapper_weight)
                    logits = logits[:,:-1]
                    _logits = logits.reshape(-1, image_encoder.vocab_len)
                    ce_loss = rl_criterion(_logits, ground_truth_tokens)
                    ce_loss = ce_loss.reshape(-1, logits.size(1))

                # loss = Reinforce_Loss(action_logits[1:], translated_text_tokens[1:], ce_loss, gamma=args.gamma, alpha=args.alpha, device=args.device)
                rl_loss = Reinforce_Loss(
                    translated_logits[:,1:], translated_text_tokens[:,1:].detach(), ce_loss, discount_matrix, normalize_factor, 
                    gamma=args.gamma, alpha=args.alpha, temperature=args.temperature, device=args.device
                )
                
                rl_loss.backward()
                optimizer.step()
                
                # Log the losses
                writer.add_scalars(
                    "Training",
                    {
                        "reinforce_loss": rl_loss.item(),
                        "cross_entropy": ce_loss.mean().item(),
                    },
                    epoch * len(trainloader) + i
                )
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}")

        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epoch} completed.")
        
    Path(f"models/{experiment_name}").mkdir(parents=True, exist_ok=True)
    torch.save(mapper.state_dict(), f"models/{experiment_name}")
    writer.close()


if __name__=="__main__":
    parser = get_config()
    args = parser.parse_args()
    
    if args.use_seed:
        torch.manual_seed(args.seed)
        
    main(args)
    