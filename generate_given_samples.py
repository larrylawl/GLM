import os
from generate_samples import *
from typing import List

# set these!
# hacky fix as idw break args
data_dir="/home/jovyan/datasets/zeno/data"
dataset_fp = os.path.join(data_dir, "infill_dataset.txt")
exp_name = "infill"
out_dir = "./samples"
out_fp = os.path.join(out_dir, f"{exp_name}.txt")

default_mask = "[MASK]"
model_mask = "[MASK]"
delim = "[delim]"

def clean_output(output):
    cleaned_output = output.split("<|startofpiece|>", 1)[1]
    return cleaned_output.strip()

def generate_given_samples(model, tokenizer, args, device, samples: List[str]):
    model.eval()
    all_outputs = []

    with torch.no_grad():
        for sample in samples:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())

            terminate_runs, raw_text, context_tokens_tensor, context_length = read_context(tokenizer, args, raw_text=sample)
            if terminate_runs == 1:
                return
            start_time = time.time()
            if args.block_lm:
                mems = []
                tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)
                mask_tokens = ['MASK', 'sMASK', 'gMASK'] if args.task_mask else ['MASK']
                mask_tokens = [tokenizer.get_command(token).Id for token in mask_tokens]
                end_tokens = [tokenizer.get_command('eop').Id, args.eod_token]
                mask_positions = []
                for token in mask_tokens:
                    mask_positions += (context_tokens_tensor == token).nonzero(as_tuple=True)[0].tolist()
                mask_positions.sort()
                if args.no_block_position:
                    for mask_position in mask_positions:
                        position_ids[0, mask_position + 1:] += args.out_seq_length
                _, *mems = model(tokens, position_ids, attention_mask, *mems)
                for mask_position in mask_positions:
                    if args.no_block_position:
                        position = position_ids[0, mask_position].item()
                    else:
                        position = mask_position
                    tokens, mems = sample_sequence(model, tokenizer, tokens, position,
                                                   args, device, mems=mems, end_tokens=end_tokens)
            else:
                tokens, _ = sample_sequence(model, tokenizer, context_tokens_tensor, context_length, args, device)
            output_tokens_list = tokens.view(-1).contiguous()
            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", raw_text, flush=True)
                decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
                trim_decode_tokens = clean_output(decode_tokens)
                print("\nGLM:", trim_decode_tokens, flush=True)
                all_outputs.append(trim_decode_tokens)

            torch.distributed.barrier(group=mpu.get_model_parallel_group())
        return all_outputs

def load_dataset(dataset_fp):
    dataset = []
    with open(dataset_fp, "r") as ipf:
        for line in ipf:
            informal, formal, *examples = line.split(delim)
            informal = informal.strip()
            formal = formal.strip()
            examples = [eg.replace(default_mask, model_mask).strip() for eg in examples]  # replace with model's mask
            # dataset.append((informal, formal, examples))
            dataset.append(examples)
    return dataset

def main():
    print("Generating samples for zeno!")
    
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()
    args.mem_length = args.seq_length + args.mem_length - 1

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)
    
    # getting infill dataset
    dataset = load_dataset(dataset_fp)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    # setting default batch size to 1
    args.batch_size = 1
    
    # generate samples
    all_outputs: List[str] = []
    for examples in dataset:
        output: List[str] = generate_given_samples(model, tokenizer, args, torch.cuda.current_device(), examples)
        output = delim.join(output)
        all_outputs.append(output)
    
    # write out samples
    with open(out_fp, "w", encoding="utf-8") as opf:
        opf.write("\n".join(all_outputs))
       
        

if __name__ == "__main__":
    main()